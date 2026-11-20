// Photometric views of a plate / VIN crop (glare, shading, low contrast, blur). Port of
// `ml/vinplate/enhance.py`. Every view has EXACTLY the size of its input, so the recogniser's
// time steps stay aligned and the views' posteriors can be fused step by step.
//
// What this can and cannot do (same honesty rule as the Python): it recovers contrast that is still
// in the pixels (veiling glare, shadows, a weak colour channel, mild blur). Pixels clipped to pure
// white are gone; `deglare` only removes the hard white blob so it is not read as a stroke.
//
// Numerics: the luminance is computed with float32 rounding like numpy so the 8-bit truncation
// lands on the same level as Python; the blurs run in float64 (so views differ from the float32
// Python by at most a couple of grey levels, see the golden tests). Pure Dart, isolate-safe.
//
// The small ndimage-style helpers (gaussian, percentile, labelling, ...) are public because
// quality.dart and rows.dart share them.

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/types.dart';

const clipLevel = 247; // all three channels at or above this = specular clipping

// ======================================================================= shared helpers
final Float32List _f1 = Float32List(1);

/// numpy's `x[...,0]*0.299 + x[...,1]*0.587 + x[...,2]*0.114` on float32 data: every product and
/// every sum is rounded to float32.
Float32List luminanceF32(RgbImage img) {
  final n = img.width * img.height;
  final out = Float32List(n);
  final p = img.rgb;
  _f1[0] = 0.299;
  final c0 = _f1[0];
  _f1[0] = 0.587;
  final c1 = _f1[0];
  _f1[0] = 0.114;
  final c2 = _f1[0];
  for (var i = 0; i < n; i++) {
    _f1[0] = p[3 * i] * c0;
    final x0 = _f1[0];
    _f1[0] = p[3 * i + 1] * c1;
    final x1 = _f1[0];
    _f1[0] = x0 + x1;
    final s = _f1[0];
    _f1[0] = p[3 * i + 2] * c2;
    final x2 = _f1[0];
    _f1[0] = s + x2;
    out[i] = _f1[0];
  }
  return out;
}

/// np.percentile (linear interpolation) of [data] for each q in 0..100.
List<double> percentiles(List<double> data, List<double> qs) {
  final s = Float64List.fromList(data)..sort();
  final n = s.length;
  if (n == 0) return [for (final _ in qs) 0.0];
  return [
    for (final q in qs)
      () {
        final v = q / 100 * (n - 1);
        final lo = v.floor(), hi = math.min(lo + 1, n - 1);
        return s[lo] + (s[hi] - s[lo]) * (v - lo);
      }()
  ];
}

/// scipy 'reflect' boundary (d c b a | a b c d | d c b a) for any index, however far outside.
int reflectIndex(int i, int n) {
  if (n == 1) return 0;
  final p = 2 * n;
  var m = i % p;
  if (m < 0) m += p;
  return m < n ? m : p - 1 - m;
}

List<double> _gaussKernel(double sigma) {
  final r = (4.0 * sigma + 0.5).toInt();
  final k = [for (var x = -r; x <= r; x++) math.exp(-0.5 / (sigma * sigma) * x * x)];
  final s = k.fold(0.0, (a, b) => a + b);
  return [for (final v in k) v / s];
}

/// scipy.ndimage.gaussian_filter(image, sigma) (truncate 4, mode 'reflect') on a w x h plane.
Float64List gaussianFilter(Float64List src, int w, int h, double sigma) {
  final k = _gaussKernel(sigma);
  final r = k.length ~/ 2;
  final tmp = Float64List(w * h);
  final line = Float64List(math.max(w, h) + 2 * r);
  // rows
  for (var y = 0; y < h; y++) {
    for (var i = 0; i < w + 2 * r; i++) {
      line[i] = src[y * w + reflectIndex(i - r, w)];
    }
    for (var x = 0; x < w; x++) {
      var s = 0.0;
      for (var j = 0; j < k.length; j++) {
        s += line[x + j] * k[j];
      }
      tmp[y * w + x] = s;
    }
  }
  final out = Float64List(w * h);
  // columns
  for (var x = 0; x < w; x++) {
    for (var i = 0; i < h + 2 * r; i++) {
      line[i] = tmp[reflectIndex(i - r, h) * w + x];
    }
    for (var y = 0; y < h; y++) {
      var s = 0.0;
      for (var j = 0; j < k.length; j++) {
        s += line[y + j] * k[j];
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

/// ndimage.label with the default 4-connectivity; returns the size of every component.
List<int> componentSizes4(Uint8List mask, int w, int h) {
  final seen = Uint8List(mask.length);
  final stack = Int32List(mask.length);
  final sizes = <int>[];
  for (var start = 0; start < mask.length; start++) {
    if (mask[start] == 0 || seen[start] == 1) continue;
    var sp = 0, count = 0;
    stack[sp++] = start;
    seen[start] = 1;
    while (sp > 0) {
      final p = stack[--sp];
      count++;
      final y = p ~/ w, x = p - y * w;
      void push(int q) {
        if (mask[q] == 1 && seen[q] == 0) {
          seen[q] = 1;
          stack[sp++] = q;
        }
      }

      if (x > 0) push(p - 1);
      if (x < w - 1) push(p + 1);
      if (y > 0) push(p - w);
      if (y < h - 1) push(p + w);
    }
    sizes.add(count);
  }
  return sizes;
}

/// scipy ndimage.laplace (second difference along each axis, 'reflect' border).
Float64List laplace(Float64List f, int w, int h) {
  final out = Float64List(w * h);
  for (var y = 0; y < h; y++) {
    final ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
    for (var x = 0; x < w; x++) {
      final xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
      final c = f[y * w + x];
      out[y * w + x] = f[y * w + xm] - 2 * c + f[y * w + xp] + f[ym * w + x] - 2 * c + f[yp * w + x];
    }
  }
  return out;
}

RgbImage _rep3(Float64List g, int w, int h) {
  final out = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    final v = g[i];
    final b = v <= 0 ? 0 : (v >= 255 ? 255 : v.toInt()); // astype(uint8) truncates after the clip
    out[3 * i] = b;
    out[3 * i + 1] = b;
    out[3 * i + 2] = b;
  }
  return RgbImage(w, h, out);
}

/// Percentile stretch of a grey plane (flat planes are returned unchanged: no noise amplification).
Float64List _stretch(Float64List g, {double lo = 2, double hi = 98}) {
  final p = percentiles(g, [lo, hi]);
  if (p[1] - p[0] < 8) return g;
  final scale = 255.0 / (p[1] - p[0]);
  final out = Float64List(g.length);
  for (var i = 0; i < g.length; i++) {
    out[i] = (g[i] - p[0]) * scale;
  }
  return out;
}

Float64List _f64(Float32List f) => Float64List.fromList(f);

// ============================================================================== clipping
/// Pixels clipped to white in every channel, with isolated noise removed: the union of every 2x2
/// block that is entirely clipped (= binary opening with a 2x2 structure, border counts as unset).
Uint8List clipMask(RgbImage img, {int level = clipLevel}) {
  final w = img.width, h = img.height, p = img.rgb;
  final m = Uint8List(w * h);
  for (var i = 0; i < w * h; i++) {
    m[i] = (p[3 * i] >= level && p[3 * i + 1] >= level && p[3 * i + 2] >= level) ? 1 : 0;
  }
  final out = Uint8List(w * h);
  for (var y = 0; y + 1 < h; y++) {
    for (var x = 0; x + 1 < w; x++) {
      final i = y * w + x;
      if (m[i] == 1 && m[i + 1] == 1 && m[i + w] == 1 && m[i + w + 1] == 1) {
        out[i] = 1;
        out[i + 1] = 1;
        out[i + w] = 1;
        out[i + w + 1] = 1;
      }
    }
  }
  return out;
}

/// {clipped_frac, largest_blob}, both rounded to 4 decimals like the Python.
Map<String, double> clipStats(RgbImage img) {
  final m = clipMask(img);
  final n = m.length;
  var set = 0;
  for (final v in m) {
    set += v;
  }
  final sizes = componentSizes4(m, img.width, img.height);
  final blob = sizes.isEmpty ? 0.0 : sizes.reduce(math.max) / n;
  double r4(double v) => (v * 1e4).roundToDouble() / 1e4;
  return {'clipped_frac': r4(set / n), 'largest_blob': r4(blob)};
}

// ================================================================================ views
/// Luminance, percentile contrast stretch, replicated to 3 channels. Fixes a veil over the whole
/// crop (haze, mild glare, backlight) and removes colour as a nuisance.
RgbImage stretch(RgbImage img) => _rep3(_stretch(_f64(luminanceF32(img))), img.width, img.height);

/// The single colour channel with the widest contrast (yellow / green plates have almost none in
/// blue; a tinted flash changes which channel carries the strokes).
RgbImage bestChannel(RgbImage img) {
  final n = img.width * img.height;
  final ch = [for (var c = 0; c < 3; c++) Float64List(n)];
  for (var i = 0; i < n; i++) {
    for (var c = 0; c < 3; c++) {
      ch[c][i] = img.rgb[3 * i + c].toDouble();
    }
  }
  var best = 0;
  var bestSpread = -double.infinity;
  for (var c = 0; c < 3; c++) {
    final p = percentiles(ch[c], [95, 5]);
    final spread = p[0] - p[1];
    if (spread > bestSpread) {
      bestSpread = spread;
      best = c;
    }
  }
  return _rep3(_stretch(ch[best]), img.width, img.height);
}

/// Single-scale retinex on luminance: removes slowly varying illumination (shadow edge, glare
/// gradient) and keeps the strokes. Not in the default view set.
RgbImage retinex(RgbImage img, {double sigmaFrac = 0.5}) {
  final w = img.width, h = img.height, n = w * h;
  final g = _f64(luminanceF32(img));
  for (var i = 0; i < n; i++) {
    g[i] += 1.0;
  }
  final blur = gaussianFilter(g, w, h, math.max(3.0, sigmaFrac * h));
  final r = Float64List(n);
  for (var i = 0; i < n; i++) {
    r[i] = math.log(g[i]) - math.log(blur[i] + 1.0);
  }
  return _rep3(_stretch(r, lo: 1, hi: 99), w, h);
}

/// Unsharp mask on the stretched luminance, for defocus / motion blur (soft recovery only).
RgbImage sharpen(RgbImage img, {double amount = 1.4}) {
  final w = img.width, h = img.height, n = w * h;
  final g = _stretch(_f64(luminanceF32(img)));
  final blur = gaussianFilter(g, w, h, math.max(1.0, h / 16.0));
  final out = Float64List(n);
  for (var i = 0; i < n; i++) {
    out[i] = g[i] + amount * (g[i] - blur[i]);
  }
  return _rep3(out, w, h);
}

/// Remove hard-clipped hot-spots by diffusing in the surrounding (unclipped) values (normalised
/// convolution), then stretch. Null when there is nothing to do: no clipped region, or a crop so
/// overexposed that the "blob" is just the white plate.
RgbImage? deglare(RgbImage img) {
  final w = img.width, h = img.height, n = w * h;
  final m = clipMask(img);
  var set = 0;
  for (final v in m) {
    set += v;
  }
  final frac = set / n;
  if (frac < 0.004 || frac > 0.45) return null;
  // hole = binary_dilation(m, iterations=2) with the default cross structure
  var hole = m;
  for (var it = 0; it < 2; it++) {
    final nx = Uint8List.fromList(hole);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (hole[y * w + x] == 0) continue;
        if (x > 0) nx[y * w + x - 1] = 1;
        if (x < w - 1) nx[y * w + x + 1] = 1;
        if (y > 0) nx[(y - 1) * w + x] = 1;
        if (y < h - 1) nx[(y + 1) * w + x] = 1;
      }
    }
    hole = nx;
  }
  final g = _f64(luminanceF32(img));
  final wt = Float64List(n);
  var wsum = 0.0;
  for (var i = 0; i < n; i++) {
    wt[i] = hole[i] == 0 ? 1.0 : 0.0;
    wsum += wt[i];
  }
  if (wsum < 0.1 * n) return null;
  final sigma = math.max(3.0, h / 6.0);
  final gw = Float64List(n);
  for (var i = 0; i < n; i++) {
    gw[i] = g[i] * wt[i];
  }
  final num = gaussianFilter(gw, w, h, sigma);
  final den = gaussianFilter(wt, w, h, sigma);
  final out = Float64List(n);
  for (var i = 0; i < n; i++) {
    out[i] = hole[i] == 1 ? num[i] / math.max(den[i], 1e-3) : g[i];
  }
  return _rep3(_stretch(out), w, h);
}

/// The views the recogniser reads, 'orig' first. Views that do not apply (`deglare` with no clipped
/// region) are omitted. Order of the rest is fixed: stretch, retinex, sharpen, channel, deglare.
List<(String, RgbImage)> variants(RgbImage crop, {Set<String> views = const {'orig', 'stretch', 'deglare'}}) {
  final out = <(String, RgbImage)>[('orig', crop)];
  if (views.contains('stretch')) out.add(('stretch', stretch(crop)));
  if (views.contains('retinex')) out.add(('retinex', retinex(crop)));
  if (views.contains('sharpen')) out.add(('sharpen', sharpen(crop)));
  if (views.contains('channel')) out.add(('channel', bestChannel(crop)));
  if (views.contains('deglare')) {
    final d = deglare(crop);
    if (d != null) out.add(('deglare', d));
  }
  return out;
}
