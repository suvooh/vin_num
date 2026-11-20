// Parallax / planarity check, ported from ml/vinplate/authenticity/active.py (`planarity`,
// `phase_shift`, `align_translation`, `_dlt`, `_apply`, `_lum_linear`). Pure Dart: isolate-safe,
// no Flutter imports, no packages.
//
// What it says (and does not): between two frames taken from DIFFERENT positions, the motion of a
// flat scene (a screen, a print) is one homography; a real car has depth, which breaks it. It is
// weak evidence - it needs real camera translation (pure rotation also looks planar), a textured
// scene, and it does not cover a screen held in front of a real background. Thresholds are the
// Python's PROVISIONAL ones, exercised only on toy physics and simulated frames.
//
// Differences from the Python reference (deliberate, listed so nobody assumes bit-equality):
//  * RANSAC uses a fixed-seed xorshift32 generator (Python: numpy PCG64, seed 0), so the sampled
//    4-tuples differ; with 300 draws the best inlier fraction (and therefore the verdict) agrees.
//  * The 4-point DLT solves the homogeneous system by Jacobi eigen-decomposition of AᵀA (the right
//    singular vector of the smallest singular value) instead of numpy's SVD: same null vector.
//  * Resampling is linear with aligned corners like `scipy.ndimage.zoom(order=1)`, Gaussian blur is
//    scipy's (sigma 6, truncate 4, 'reflect'), the Hann window is numpy's symmetric one.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/types.dart';

// ------------------------------------------------------------------------------- result
class PlanarityResult {
  const PlanarityResult(this.verdict, {this.residualPx, this.baselinePx, this.tiles = 0, this.inlierFrac, this.reasons = const []});

  /// planar | non_planar | inconclusive
  final String verdict;
  final double? residualPx, baselinePx, inlierFrac;
  final int tiles;
  final List<String> reasons;

  Map<String, dynamic> toJson() => {
        'verdict': verdict,
        'residual_px': residualPx,
        'baseline_px': baselinePx,
        'tiles': tiles,
        'inlier_frac': inlierFrac,
        'reasons': reasons,
      };
}

// ----------------------------------------------------------------------------- helpers
/// Python's round(): halves go to the even neighbour (`int(round(x))`).
int pyRound(double x) {
  final f = x.floorToDouble();
  final d = x - f;
  if (d < 0.5) return f.toInt();
  if (d > 0.5) return f.toInt() + 1;
  return f.toInt().isEven ? f.toInt() : f.toInt() + 1;
}

double _round(double v, int places) {
  final p = math.pow(10, places).toDouble();
  return (v * p).roundToDouble() / p;
}

double median(List<double> v) {
  if (v.isEmpty) return double.nan;
  final s = [...v]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

final Float64List _lin22 = Float64List.fromList([for (var i = 0; i < 256; i++) math.pow(i / 255.0, 2.2).toDouble()]);

/// sqrt of the approximate linear luminance (`_lum_linear(rgb) ** 0.5`), row-major h*w.
Float64List sqrtLinearLuminance(RgbImage im) {
  final n = im.width * im.height;
  final out = Float64List(n);
  final p = im.rgb;
  for (var i = 0; i < n; i++) {
    out[i] = math.sqrt(_lin22[p[3 * i]] * 0.2126 + _lin22[p[3 * i + 1]] * 0.7152 + _lin22[p[3 * i + 2]] * 0.0722);
  }
  return out;
}

/// `scipy.ndimage.zoom(g, sc, order=1)`: output size round(n*sc); the first and last samples map to
/// the first and last input samples (aligned corners).
(Float64List, int, int) zoomLinear(Float64List g, int h, int w, double sc) {
  final oh = math.max(1, pyRound(h * sc)), ow = math.max(1, pyRound(w * sc));
  final out = Float64List(oh * ow);
  final ry = oh > 1 ? (h - 1) / (oh - 1) : 0.0, rx = ow > 1 ? (w - 1) / (ow - 1) : 0.0;
  for (var y = 0; y < oh; y++) {
    final fy = y * ry;
    final y0 = math.min(h - 1, fy.floor()), y1 = math.min(h - 1, y0 + 1);
    final wy = fy - y0;
    for (var x = 0; x < ow; x++) {
      final fx = x * rx;
      final x0 = math.min(w - 1, fx.floor()), x1 = math.min(w - 1, x0 + 1);
      final wx = fx - x0;
      final a = g[y0 * w + x0] * (1 - wx) + g[y0 * w + x1] * wx;
      final b = g[y1 * w + x0] * (1 - wx) + g[y1 * w + x1] * wx;
      out[y * ow + x] = a * (1 - wy) + b * wy;
    }
  }
  return (out, oh, ow);
}

/// `scipy.ndimage.gaussian_filter(g, sigma)`: truncate 4.0, 'reflect' boundary, separable.
Float64List gaussianFilter(Float64List g, int h, int w, double sigma) {
  final r = (4.0 * sigma + 0.5).toInt();
  final k = Float64List(2 * r + 1);
  var sum = 0.0;
  for (var i = -r; i <= r; i++) {
    k[i + r] = math.exp(-0.5 * i * i / (sigma * sigma));
    sum += k[i + r];
  }
  for (var i = 0; i < k.length; i++) {
    k[i] /= sum;
  }
  int refl(int i, int n) {
    while (i < 0 || i >= n) {
      i = i < 0 ? -i - 1 : 2 * n - i - 1;
    }
    return i;
  }

  final tmp = Float64List(h * w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var s = 0.0;
      for (var i = -r; i <= r; i++) {
        s += g[y * w + refl(x + i, w)] * k[i + r];
      }
      tmp[y * w + x] = s;
    }
  }
  final out = Float64List(h * w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var s = 0.0;
      for (var i = -r; i <= r; i++) {
        s += tmp[refl(y + i, h) * w + x] * k[i + r];
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------ FFT
bool _isPow2(int n) => n > 0 && (n & (n - 1)) == 0;

void _fftPow2(Float64List re, Float64List im, int n, {required bool inverse}) {
  // bit reversal
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    for (; j & bit != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      final tr = re[i], ti = im[i];
      re[i] = re[j];
      im[i] = im[j];
      re[j] = tr;
      im[j] = ti;
    }
  }
  for (var len = 2; len <= n; len <<= 1) {
    final ang = 2 * math.pi / len * (inverse ? 1 : -1);
    final wr = math.cos(ang), wi = math.sin(ang);
    for (var i = 0; i < n; i += len) {
      var cr = 1.0, ci = 0.0;
      for (var k = 0; k < len ~/ 2; k++) {
        final a = i + k, b = i + k + len ~/ 2;
        final xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        final ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/// Unnormalised DFT of any length n (radix-2 when n is a power of two, Bluestein otherwise).
void fft1d(Float64List re, Float64List im, int n, {bool inverse = false}) {
  if (n <= 1) return;
  if (_isPow2(n)) {
    _fftPow2(re, im, n, inverse: inverse);
    return;
  }
  var m = 1;
  while (m < 2 * n - 1) {
    m <<= 1;
  }
  final sgn = inverse ? 1.0 : -1.0;
  final wr = Float64List(n), wi = Float64List(n);
  for (var k = 0; k < n; k++) {
    final a = sgn * math.pi * ((k * k) % (2 * n)) / n;
    wr[k] = math.cos(a);
    wi[k] = math.sin(a);
  }
  final ar = Float64List(m), ai = Float64List(m), br = Float64List(m), bi = Float64List(m);
  for (var k = 0; k < n; k++) {
    ar[k] = re[k] * wr[k] - im[k] * wi[k];
    ai[k] = re[k] * wi[k] + im[k] * wr[k];
    br[k] = wr[k];
    bi[k] = -wi[k];
    if (k > 0) {
      br[m - k] = wr[k];
      bi[m - k] = -wi[k];
    }
  }
  _fftPow2(ar, ai, m, inverse: false);
  _fftPow2(br, bi, m, inverse: false);
  for (var i = 0; i < m; i++) {
    final r = ar[i] * br[i] - ai[i] * bi[i], q = ar[i] * bi[i] + ai[i] * br[i];
    ar[i] = r;
    ai[i] = q;
  }
  _fftPow2(ar, ai, m, inverse: true);
  for (var k = 0; k < n; k++) {
    final cr = ar[k] / m, ci = ai[k] / m;
    re[k] = cr * wr[k] - ci * wi[k];
    im[k] = cr * wi[k] + ci * wr[k];
  }
}

/// In-place 2-D DFT of an h x w complex array (row-major), unnormalised.
void fft2d(Float64List re, Float64List im, int h, int w, {bool inverse = false}) {
  final rr = Float64List(w), ri = Float64List(w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      rr[x] = re[y * w + x];
      ri[x] = im[y * w + x];
    }
    fft1d(rr, ri, w, inverse: inverse);
    for (var x = 0; x < w; x++) {
      re[y * w + x] = rr[x];
      im[y * w + x] = ri[x];
    }
  }
  final cr = Float64List(h), ci = Float64List(h);
  for (var x = 0; x < w; x++) {
    for (var y = 0; y < h; y++) {
      cr[y] = re[y * w + x];
      ci[y] = im[y * w + x];
    }
    fft1d(cr, ci, h, inverse: inverse);
    for (var y = 0; y < h; y++) {
      re[y * w + x] = cr[y];
      im[y * w + x] = ci[y];
    }
  }
}

Float64List _hanning(int n) => Float64List.fromList([for (var k = 0; k < n; k++) n == 1 ? 1.0 : 0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1))]);

/// Sub-pixel displacement (dx, dy) of content from [a] to [b] (a feature at p in a is at p+d in b)
/// by phase correlation, and the correlation peak height (0..1). [a] and [b] are h x w, row-major.
(double, double, double) phaseShift(Float64List a, Float64List b, int h, int w) {
  final win = _hanning(h);
  final wx = _hanning(w);
  double mean(Float64List v) {
    var s = 0.0;
    for (final x in v) {
      s += x;
    }
    return s / v.length;
  }

  final ma = mean(a), mb = mean(b);
  final ar = Float64List(h * w), ai = Float64List(h * w), br = Float64List(h * w), bi = Float64List(h * w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final k = y * w + x, wv = win[y] * wx[x];
      ar[k] = (a[k] - ma) * wv;
      br[k] = (b[k] - mb) * wv;
    }
  }
  fft2d(ar, ai, h, w);
  fft2d(br, bi, h, w);
  // r = fb * conj(fa), normalised to unit magnitude (+1e-6 like the reference)
  final rr = Float64List(h * w), ri = Float64List(h * w);
  for (var k = 0; k < h * w; k++) {
    final re = br[k] * ar[k] + bi[k] * ai[k], im = bi[k] * ar[k] - br[k] * ai[k];
    final mag = math.sqrt(re * re + im * im) + 1e-6;
    rr[k] = re / mag;
    ri[k] = im / mag;
  }
  fft2d(rr, ri, h, w, inverse: true);
  final inv = 1.0 / (h * w);
  var best = -double.infinity;
  var iy = 0, ix = 0;
  for (var k = 0; k < h * w; k++) {
    final v = rr[k] * inv;
    if (v > best) {
      best = v;
      iy = k ~/ w;
      ix = k % w;
    }
  }
  double c(int y, int x) => rr[((y % h + h) % h) * w + ((x % w + w) % w)] * inv;
  double refine(double m1, double m0, double p1) {
    final den = m1 - 2 * m0 + p1;
    return den.abs() < 1e-9 ? 0.0 : 0.5 * (m1 - p1) / den;
  }

  var dx = ix + refine(c(iy, ix - 1), c(iy, ix), c(iy, ix + 1));
  var dy = iy + refine(c(iy - 1, ix), c(iy, ix), c(iy + 1, ix));
  if (dx > w / 2) dx -= w;
  if (dy > h / 2) dy -= h;
  return (dx, dy, best);
}

/// Whole-pixel roll (dx, dy) that lines [mov] up with [ref]: `np.roll(np.roll(mov, dy, 0), dx, 1)`.
/// Compensates hand-shake between a flash-off and a flash-on frame (phase correlation on sqrt of
/// linear luminance, like `align_translation`). The app's flash test (`checks.dart`
/// `retroreflection`) does its own SAD alignment; this is the reference-faithful alternative.
(int, int) alignTranslation(RgbImage ref, RgbImage mov) {
  final ga = sqrtLinearLuminance(ref), gb = sqrtLinearLuminance(mov);
  final (dx, dy, _) = phaseShift(gb, ga, ref.height, ref.width);
  return (pyRound(dx), pyRound(dy));
}

// ----------------------------------------------------------------------------- homography
typedef Mat3 = List<double>; // row-major 3x3

/// Symmetric eigen-decomposition by cyclic Jacobi. Returns (eigenvalues, eigenvectors as columns).
(List<double>, List<List<double>>) _jacobi(List<List<double>> a0, int n) {
  final a = [for (final r in a0) [...r]];
  final v = [for (var i = 0; i < n; i++) [for (var j = 0; j < n; j++) i == j ? 1.0 : 0.0]];
  for (var sweep = 0; sweep < 60; sweep++) {
    var off = 0.0;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        off += a[i][j] * a[i][j];
      }
    }
    if (off < 1e-22) break;
    for (var p = 0; p < n - 1; p++) {
      for (var q = p + 1; q < n; q++) {
        if (a[p][q].abs() < 1e-300) continue;
        final theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        final t = (theta >= 0 ? 1.0 : -1.0) / (theta.abs() + math.sqrt(theta * theta + 1));
        final c = 1 / math.sqrt(t * t + 1), s = t * c;
        for (var k = 0; k < n; k++) {
          final akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (var k = 0; k < n; k++) {
          final apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (var k = 0; k < n; k++) {
          final vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return ([for (var i = 0; i < n; i++) a[i][i]], v);
}

Mat3? _inv3(Mat3 m) {
  final a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  final det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (det.abs() < 1e-300 || det.isNaN) return null;
  return [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
}

Mat3 _mul3(Mat3 x, Mat3 y) => [
      for (var r = 0; r < 3; r++)
        for (var c = 0; c < 3; c++) x[3 * r] * y[c] + x[3 * r + 1] * y[3 + c] + x[3 * r + 2] * y[6 + c]
    ];

/// 4-point DLT homography mapping p -> q (Hartley-normalised), or null if degenerate. Port of `_dlt`.
Mat3? dlt(List<(double, double)> p, List<(double, double)> q) {
  (List<(double, double)>, Mat3) norm(List<(double, double)> x) {
    var cx = 0.0, cy = 0.0;
    for (final (a, b) in x) {
      cx += a;
      cy += b;
    }
    cx /= x.length;
    cy /= x.length;
    var md = 0.0;
    for (final (a, b) in x) {
      md += math.sqrt((a - cx) * (a - cx) + (b - cy) * (b - cy));
    }
    md /= x.length;
    final s = math.sqrt(2) / (md + 1e-9);
    return ([for (final (a, b) in x) (s * (a - cx), s * (b - cy))], [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1]);
  }

  final (pn, tp) = norm(p);
  final (qn, tq) = norm(q);
  final a = <List<double>>[];
  for (var i = 0; i < pn.length; i++) {
    final (x, y) = pn[i];
    final (u, v) = qn[i];
    a.add([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    a.add([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  final ata = [for (var i = 0; i < 9; i++) [for (var j = 0; j < 9; j++) 0.0]];
  for (final r in a) {
    for (var i = 0; i < 9; i++) {
      for (var j = 0; j < 9; j++) {
        ata[i][j] += r[i] * r[j];
      }
    }
  }
  final (vals, vecs) = _jacobi(ata, 9);
  var k = 0;
  for (var i = 1; i < 9; i++) {
    if (vals[i] < vals[k]) k = i;
  }
  final hn = <double>[for (var i = 0; i < 9; i++) vecs[i][k]];
  final tqi = _inv3(tq);
  if (tqi == null) return null;
  final h = _mul3(_mul3(tqi, hn), tp);
  final z = h[8].abs() > 1e-12 ? h[8] : 1.0;
  return [for (final v in h) v / z];
}

(double, double) _apply(Mat3 h, double x, double y) {
  final w = h[6] * x + h[7] * y + h[8];
  return ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w);
}

// ------------------------------------------------------------------------------ RANSAC rng
/// xorshift32, fixed seed: deterministic across runs and platforms (all values masked to 32 bits).
class _Rng {
  _Rng(this._s);
  int _s;
  int next() {
    _s ^= (_s << 13) & 0xFFFFFFFF;
    _s ^= _s >> 17;
    _s ^= (_s << 5) & 0xFFFFFFFF;
    _s &= 0xFFFFFFFF;
    return _s;
  }

  /// Four distinct indices in [0, n).
  List<int> four(int n) {
    final out = <int>[];
    while (out.length < 4) {
      final i = next() % n;
      if (!out.contains(i)) out.add(i);
    }
    return out;
  }
}

// ----------------------------------------------------------------------------- planarity
/// Is the motion between [a] and [b] explained by ONE plane? Statistic: the fraction of tile
/// motions the best single homography explains (RANSAC inlier fraction), as in the reference.
/// [work] is the length of the LONGER side after downscaling (the reference's `work`).
PlanarityResult planarity(RgbImage a, RgbImage b, {int work = 512, int tile = 64, double minBaselinePx = 1.5, int seed = 0x9E3779B9}) {
  if (a.width != b.width || a.height != b.height) {
    return const PlanarityResult('inconclusive', reasons: ['the two frames differ in size, so their motion cannot be compared']);
  }
  final sc = work / math.max(a.height, a.width);
  var (ga, h, w) = zoomLinear(sqrtLinearLuminance(a), a.height, a.width, sc);
  var (gb, _, _) = zoomLinear(sqrtLinearLuminance(b), b.height, b.width, sc);
  final ba = gaussianFilter(ga, h, w, 6), bb = gaussianFilter(gb, h, w, 6);
  for (var i = 0; i < ga.length; i++) {
    ga[i] -= ba[i]; // texture only
    gb[i] -= bb[i];
  }
  final pts = <(double, double)>[], qts = <(double, double)>[];
  final ta = Float64List(tile * tile), tb = Float64List(tile * tile);
  final step = math.max(1, tile ~/ 2);
  for (var y = 0; y <= h - tile; y += step) {
    for (var x = 0; x <= w - tile; x += step) {
      for (var j = 0; j < tile; j++) {
        for (var i = 0; i < tile; i++) {
          ta[j * tile + i] = ga[(y + j) * w + x + i];
          tb[j * tile + i] = gb[(y + j) * w + x + i];
        }
      }
      if (_std(ta) < 0.004 || _std(tb) < 0.004) continue;
      final (dx, dy, pk) = phaseShift(ta, tb, tile, tile);
      if (pk < 0.18 || dx.abs() > tile / 3 || dy.abs() > tile / 3) continue;
      final cx = x + tile / 2, cy = y + tile / 2;
      pts.add((cx, cy));
      qts.add((cx + dx, cy + dy));
    }
  }
  if (pts.length < 12) {
    return PlanarityResult('inconclusive', tiles: pts.length, reasons: const ['too few textured, matchable tiles']);
  }
  final mot = [for (var i = 0; i < pts.length; i++) math.sqrt(math.pow(qts[i].$1 - pts[i].$1, 2) + math.pow(qts[i].$2 - pts[i].$2, 2)).toDouble()];
  final base = median(mot);
  final baseR = _round(base, 2);
  if (base < minBaselinePx) {
    return PlanarityResult('inconclusive',
        tiles: pts.length, baselinePx: baseR, reasons: ['camera moved only ${base.toStringAsFixed(2)}px at ${work}px width: not enough baseline']);
  }
  final rng = _Rng(seed == 0 ? 1 : seed);
  final tau = math.max(0.35, 0.06 * base); // px tolerance at the working resolution
  var bestIn = -1;
  List<double>? bestErr;
  for (var it = 0; it < 300; it++) {
    final idx = rng.four(pts.length);
    final h3 = dlt([for (final i in idx) pts[i]], [for (final i in idx) qts[i]]);
    if (h3 == null) continue;
    final err = List<double>.filled(pts.length, 0);
    var nIn = 0;
    for (var i = 0; i < pts.length; i++) {
      final (px, py) = _apply(h3, pts[i].$1, pts[i].$2);
      final e = math.sqrt(math.pow(px - qts[i].$1, 2) + math.pow(py - qts[i].$2, 2)).toDouble();
      err[i] = e;
      if (e <= tau) nIn++; // NaN compares false, like np.nansum(err <= tau)
    }
    if (nIn > bestIn) {
      bestIn = nIn;
      bestErr = err;
    }
  }
  if (bestErr == null) return PlanarityResult('inconclusive', tiles: pts.length, baselinePx: baseR);
  final frac = bestIn / pts.length;
  final residual = median([for (final e in bestErr) if (!e.isNaN) e]);
  final res = (residual: _round(residual, 3), frac: _round(frac, 3));
  if (frac >= 0.85) {
    return PlanarityResult('planar',
        residualPx: res.residual,
        baselinePx: baseR,
        tiles: pts.length,
        inlierFrac: res.frac,
        reasons: [
          '${(frac * 100).round()}% of tiles move as ONE plane (motion ${base.toStringAsFixed(1)}px). Consistent with a screen/print - '
              'but also with a flat, shallow scene or a pure camera rotation.'
        ]);
  }
  if (frac < 0.8) {
    return PlanarityResult('non_planar', residualPx: res.residual, baselinePx: baseR, tiles: pts.length, inlierFrac: res.frac);
  }
  return PlanarityResult('inconclusive',
      residualPx: res.residual, baselinePx: baseR, tiles: pts.length, inlierFrac: res.frac, reasons: const ['ambiguous planarity']);
}

double _std(Float64List v) {
  var s = 0.0;
  for (final x in v) {
    s += x;
  }
  final m = s / v.length;
  var q = 0.0;
  for (final x in v) {
    q += (x - m) * (x - m);
  }
  return math.sqrt(q / v.length);
}
