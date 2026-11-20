import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/planarity.dart';

// ---------------------------------------------------------------- helpers shared by the tests
Float64List unpackGray(String b64) => Float64List.fromList(zlib.decode(base64Decode(b64)).map((e) => e.toDouble()).toList());

RgbImage grayToRgb(Float64List g, int w, int h) {
  final out = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    final v = g[i].round().clamp(0, 255);
    out[3 * i] = v;
    out[3 * i + 1] = v;
    out[3 * i + 2] = v;
  }
  return RgbImage(w, h, out);
}

/// Seeded N(0,1) field (Box-Muller over an LCG): deterministic, platform independent.
Float64List randn(int n, int seed) {
  var s = seed * 2654435761 + 12345;
  double u() {
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF;
    return (s + 1) / 2147483649.0;
  }

  final out = Float64List(n);
  for (var i = 0; i < n; i += 2) {
    final r = math.sqrt(-2 * math.log(u())), t = 2 * math.pi * u();
    out[i] = r * math.cos(t);
    if (i + 1 < n) out[i + 1] = r * math.sin(t);
  }
  return out;
}

/// The toy scene of ml/tests/test_active.py `textured` (smooth multi-scale texture), as 0..1 linear
/// luminance, then gamma-encoded to u8 like `to_u8`.
Float64List texturedU8(int h, int w, int seed) {
  final g = Float64List(h * w);
  var k = 0;
  for (final (sigma, amp) in const [(1.5, 5.0), (5.0, 10.0), (16.0, 14.0), (40.0, 18.0)]) {
    final f = gaussianFilter(randn(h * w, seed * 10 + k++), h, w, sigma);
    for (var i = 0; i < g.length; i++) {
      g[i] += f[i] * amp;
    }
  }
  var lo = double.infinity, hi = -double.infinity;
  for (final v in g) {
    lo = math.min(lo, v);
    hi = math.max(hi, v);
  }
  return Float64List.fromList([for (final v in g) 255 * math.pow(((0.08 + 0.3 * (v - lo) / (hi - lo)).clamp(0.0, 1.0)), 1 / 2.2).toDouble()]);
}

/// Affine warp with PIL's inverse-mapping convention (ml/tests/test_active.py `warp`), bilinear.
Float64List warp(Float64List g, int h, int w, double sx, double sy, [double rotDeg = 0]) {
  final a = rotDeg * math.pi / 180, c = math.cos(a), s = math.sin(a);
  final c2 = -sx - c * w / 2 - s * h / 2 + w / 2, c5 = -sy + s * w / 2 - c * h / 2 + h / 2;
  final out = Float64List(h * w);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final px = c * (x + 0.5) + s * (y + 0.5) + c2 - 0.5, py = -s * (x + 0.5) + c * (y + 0.5) + c5 - 0.5;
      if (px < 0 || py < 0 || px > w - 1 || py > h - 1) {
        out[y * w + x] = 0;
        continue;
      }
      final x0 = px.floor(), y0 = py.floor(), x1 = math.min(w - 1, x0 + 1), y1 = math.min(h - 1, y0 + 1);
      final fx = px - x0, fy = py - y0;
      out[y * w + x] = (g[y0 * w + x0] * (1 - fx) + g[y0 * w + x1] * fx) * (1 - fy) + (g[y1 * w + x0] * (1 - fx) + g[y1 * w + x1] * fx) * fy;
    }
  }
  return out;
}

/// Circularly shifts [g] by a (possibly fractional) displacement via the Fourier shift theorem:
/// the content moves by (+dx, +dy). Exact for periodic data, so the true shift is known precisely.
Float64List fourierShift(Float64List g, int h, int w, double dx, double dy) {
  final re = Float64List.fromList(g), im = Float64List(h * w);
  fft2d(re, im, h, w);
  for (var ky = 0; ky < h; ky++) {
    final fy = (ky <= h ~/ 2 ? ky : ky - h) / h;
    for (var kx = 0; kx < w; kx++) {
      final fx = (kx <= w ~/ 2 ? kx : kx - w) / w;
      final a = -2 * math.pi * (fx * dx + fy * dy), c = math.cos(a), s = math.sin(a);
      final i = ky * w + kx, r = re[i], m = im[i];
      re[i] = r * c - m * s;
      im[i] = r * s + m * c;
    }
  }
  fft2d(re, im, h, w, inverse: true);
  for (var i = 0; i < h * w; i++) {
    re[i] /= h * w;
  }
  return re;
}

void main() {
  group('FFT', () {
    test('matches a naive DFT for power-of-two and arbitrary (Bluestein) lengths, and round-trips', () {
      for (final n in [8, 16, 12, 7, 30, 64]) {
        final rng = randn(2 * n, n);
        final re = Float64List.fromList(rng.sublist(0, n)), im = Float64List.fromList(rng.sublist(n));
        final nr = Float64List(n), ni = Float64List(n);
        for (var k = 0; k < n; k++) {
          for (var t = 0; t < n; t++) {
            final a = -2 * math.pi * k * t / n;
            nr[k] += re[t] * math.cos(a) - im[t] * math.sin(a);
            ni[k] += re[t] * math.sin(a) + im[t] * math.cos(a);
          }
        }
        final r2 = Float64List.fromList(re), i2 = Float64List.fromList(im);
        fft1d(r2, i2, n);
        for (var k = 0; k < n; k++) {
          expect(r2[k], closeTo(nr[k], 1e-8), reason: 'n=$n re[$k]');
          expect(i2[k], closeTo(ni[k], 1e-8), reason: 'n=$n im[$k]');
        }
        fft1d(r2, i2, n, inverse: true);
        for (var k = 0; k < n; k++) {
          expect(r2[k] / n, closeTo(re[k], 1e-9));
          expect(i2[k] / n, closeTo(im[k], 1e-9));
        }
      }
    });

    test('2-D transform round-trips on a non-square, non-power-of-two array', () {
      const h = 12, w = 20;
      final re = Float64List.fromList(randn(h * w, 3)), im = Float64List(h * w);
      final r0 = Float64List.fromList(re);
      fft2d(re, im, h, w);
      fft2d(re, im, h, w, inverse: true);
      for (var i = 0; i < h * w; i++) {
        expect(re[i] / (h * w), closeTo(r0[i], 1e-9));
        expect(im[i] / (h * w), closeTo(0, 1e-9));
      }
    });
  });

  group('phase correlation', () {
    test('integer shifts: right sign (content moved by +d reads as +d), small Hann-window bias toward zero', () {
      const n = 64;
      final base = gaussianFilter(randn(n * n, 4), n, n, 2.0);
      for (final (sx, sy) in const [(3, 5), (-4, 2), (6, -7)]) {
        final b = fourierShift(base, n, n, sx.toDouble(), sy.toDouble());
        final (dx, dy, pk) = phaseShift(base, b, n, n);
        expect(dx, closeTo(sx, 0.7), reason: 'dx for shift ($sx,$sy)');
        expect(dy, closeTo(sy, 0.7), reason: 'dy for shift ($sx,$sy)');
        expect(dx.sign, sx.sign.toDouble());
        expect(dy.sign, sy.sign.toDouble());
        expect(pk, greaterThan(0.1));
      }
    });

    test('sub-pixel shift on a 128x128 texture (the Python test_phase_shift_is_subpixel_accurate case)', () {
      const n = 128;
      final a = gaussianFilter(randn(n * n, 7), n, n, 2.0);
      final b = fourierShift(a, n, n, -3.7, 2.4); // ndi.shift(a, (2.4, -3.7)): content moves (dx=-3.7, dy=+2.4)
      final (dx, dy, pk) = phaseShift(a, b, n, n);
      expect(dx, closeTo(-3.7, 0.4));
      expect(dy, closeTo(2.4, 0.4));
      expect(pk, greaterThan(0.15));
    });

    test('arbitrary (non power-of-two) size takes the Bluestein path and agrees', () {
      const h = 96, w = 120;
      final a = gaussianFilter(randn(h * w, 9), h, w, 2.0);
      final b = fourierShift(a, h, w, 5.0, -3.0);
      final (dx, dy, _) = phaseShift(a, b, h, w);
      expect(dx, closeTo(5.0, 0.7));
      expect(dy, closeTo(-3.0, 0.7));
    });

    test('alignTranslation returns the roll that lines the moving frame up with the reference', () {
      const h = 120, w = 160;
      final base = texturedU8(h, w, 5);
      final ref = grayToRgb(base, w, h);
      // mov = ref rolled by (+3 rows, -4 cols)
      final rolled = Float64List(h * w);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          rolled[((y + 3) % h) * w + (x - 4 + w) % w] = base[y * w + x];
        }
      }
      final (dx, dy) = alignTranslation(ref, grayToRgb(rolled, w, h));
      expect((dx, dy), (4, -3)); // rolling mov by (dy=-3, dx=+4) restores ref
    });
  });

  group('resampling and blur match scipy', () {
    test('zoom(order=1) maps first/last samples onto first/last samples', () {
      const w = 100, h = 4;
      final g = Float64List.fromList([for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) x.toDouble()]);
      final (z, oh, ow) = zoomLinear(g, h, w, 0.5);
      expect((oh, ow), (2, 50));
      expect(z[0], 0);
      expect(z[49], closeTo(99, 1e-9));
      expect(z[10], closeTo(10 * 99 / 49, 1e-9));
    });

    test('gaussian: constant stays constant at the borders, an impulse keeps unit mass and is symmetric', () {
      final c = gaussianFilter(Float64List.fromList(List.filled(40 * 50, 7.0)), 40, 50, 6);
      for (final v in c) {
        expect(v, closeTo(7.0, 1e-9));
      }
      final imp = Float64List(61 * 61)..[30 * 61 + 30] = 1.0;
      final f = gaussianFilter(imp, 61, 61, 3);
      expect(f.fold(0.0, (a, b) => a + b), closeTo(1.0, 1e-9));
      expect(f[30 * 61 + 25], closeTo(f[30 * 61 + 35], 1e-12));
    });

    test('pyRound is banker\'s rounding like Python', () {
      expect([pyRound(0.5), pyRound(1.5), pyRound(2.5), pyRound(-0.5), pyRound(2.4), pyRound(2.6)], [0, 2, 2, 0, 2, 3]);
    });
  });

  group('homography', () {
    test('DLT recovers a known homography from 4 points', () {
      const h = [1.02, 0.03, 4.0, -0.02, 0.98, -2.5, 1e-5, -2e-5, 1.0];
      final p = [(10.0, 12.0), (200.0, 15.0), (190.0, 170.0), (15.0, 160.0)];
      final q = [
        for (final (x, y) in p) ((h[0] * x + h[1] * y + h[2]) / (h[6] * x + h[7] * y + h[8]), (h[3] * x + h[4] * y + h[5]) / (h[6] * x + h[7] * y + h[8]))
      ];
      final est = dlt(p, q)!;
      for (var i = 0; i < 9; i++) {
        expect(est[i], closeTo(h[i], 1e-6), reason: 'h[$i]');
      }
    });

    test('degenerate (collinear) points never throw', () {
      final p = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
      expect(() => dlt(p, p), returnsNormally);
    });
  });

  group('planarity on toy scenes (ml/tests/test_active.py constructions)', () {
    const h = 480, w = 640;
    late Float64List big;
    setUpAll(() => big = texturedU8(h, w, 11));

    test('a flat replay (everything moves as ONE plane) is planar', () {
      final b = warp(big, h, w, 7, 3, 1.0);
      final r = planarity(grayToRgb(big, w, h), grayToRgb(b, w, h));
      expect(r.verdict, 'planar', reason: '$r ${r.toJson()}');
      expect(r.inlierFrac, greaterThanOrEqualTo(0.85));
      expect(r.baselinePx, greaterThan(minBaselineForTests));
    });

    test('a layered scene (nearer block moves 3x as far) is non_planar', () {
      final bg = warp(big, h, w, 3, 1, 0.3), fg = warp(big, h, w, 11, 4, 0.3);
      final lay = Float64List(h * w);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final inside = y >= 100 && y < 380 && x >= 140 && x < 520;
          lay[y * w + x] = inside ? fg[y * w + x] : bg[y * w + x];
        }
      }
      final r = planarity(grayToRgb(big, w, h), grayToRgb(lay, w, h));
      expect(r.verdict, 'non_planar', reason: '${r.toJson()}');
      expect(r.inlierFrac, lessThan(0.8));
    });

    test('too little camera motion is inconclusive, and says why', () {
      final b = warp(big, h, w, 0.2, 0.1);
      final r = planarity(grayToRgb(big, w, h), grayToRgb(b, w, h));
      expect(r.verdict, 'inconclusive');
      expect(r.reasons.join(), contains('not enough baseline'));
    });

    test('a featureless scene is inconclusive (too few matchable tiles)', () {
      final flat = grayToRgb(Float64List.fromList(List.filled(h * w, 128.0)), w, h);
      final r = planarity(flat, flat);
      expect(r.verdict, 'inconclusive');
      expect(r.reasons.join(), contains('too few'));
    });

    test('frames of different sizes are not compared', () {
      final r = planarity(grayToRgb(big, w, h), grayToRgb(Float64List(100 * 100), 100, 100));
      expect(r.verdict, 'inconclusive');
    });

    test('deterministic: the same inputs give the same result', () {
      final b = warp(big, h, w, 7, 3, 1.0);
      final a1 = planarity(grayToRgb(big, w, h), grayToRgb(b, w, h)), a2 = planarity(grayToRgb(big, w, h), grayToRgb(b, w, h));
      expect(a1.toJson(), a2.toJson());
    });
  });

  group('golden: Python reference on stored frames', () {
    final data = (jsonDecode(File('test/fixtures/golden/liveness_planarity.json').readAsStringSync()) as Map).cast<String, dynamic>();
    final w = data['w'] as int, h = data['h'] as int;
    final a = grayToRgb(unpackGray(data['a'] as String), w, h);

    for (final name in ['flat', 'layered', 'tiny']) {
      test('pair "$name" agrees with ml.vinplate.authenticity.active.planarity', () {
        final p = (data['pairs'] as Map)[name] as Map;
        final exp = (p['expect'] as Map).cast<String, dynamic>();
        final sw = Stopwatch()..start();
        final r = planarity(a, grayToRgb(unpackGray(p['b'] as String), w, h));
        sw.stop();
        // ignore: avoid_print
        print('planarity "$name" 640x480: ${sw.elapsedMilliseconds} ms (VM, JIT)  ->  ${r.verdict}, inliers ${r.inlierFrac}, baseline ${r.baselinePx}px, ${r.tiles} tiles');
        expect(r.verdict, exp['verdict']);
        expect(r.tiles, exp['tiles'], reason: 'the tile selection is deterministic: it must match exactly');
        expect(r.baselinePx, closeTo(exp['baseline_px'] as double, 0.05));
        if (exp['inlier_frac'] != null) {
          expect(r.inlierFrac, closeTo(exp['inlier_frac'] as double, 0.08), reason: 'RANSAC draws differ (different PRNG), so allow a small band');
        }
        if (exp['residual_px'] != null) expect(r.residualPx, closeTo(exp['residual_px'] as double, 0.15));
      });
    }

    test('a blank pair is inconclusive with no tiles, like Python', () {
      final blank = grayToRgb(Float64List.fromList(List.filled(w * h, 128.0)), w, h);
      final r = planarity(blank, blank);
      final exp = ((data['blank'] as Map)['expect'] as Map);
      expect(r.verdict, exp['verdict']);
      expect(r.tiles, exp['tiles']);
    });
  });
}

const minBaselineForTests = 1.5;
