// The flash test's uniformity / hot-spot rule (ml/vinplate/authenticity/active.py `gain_uniformity` and the
// verdict ladder of `retroreflection`), on a deterministic toy-physics scene. Ported from the cases in
// ml/tests/test_active.py ("flash uniformity / hot-spot"): the Python builds its scenes with numpy's random
// generator, which has no Dart twin, so these frames are generated here; the arithmetic under test is the same
// and the cross-check on PYTHON-recorded frames is in liveness_test.dart (liveness_retro.json).
//
// These prove the code computes what the physics says on a toy model. They do NOT prove anything about real
// phones, real flashes or real HSRP plates.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/checks.dart';
import 'package:vinplate_tester/reader/liveness.dart';

const H = 240, W = 400;
const plate = Box(100, 90, 300, 150); // PLATE[90:150, 100:300]
bool inPlate(int x, int y) => plate.contains(x, y);

/// deterministic pseudo-noise in [0, 1) from integer coordinates
double hash(int x, int y, int seed) {
  var h = (x * 374761393 + y * 668265263 + seed * 1274126177) & 0x7fffffff;
  h = ((h ^ (h >> 13)) * 1274126177) & 0x7fffffff;
  return ((h ^ (h >> 16)) & 0xffff) / 65536.0;
}

/// smooth, feature-rich background in 0.08 .. 0.38 (linear light)
double body(int x, int y) {
  final v = math.sin(x / 9.0) * 0.5 + math.sin(y / 7.0 + x / 23.0) * 0.5 + math.sin((x + 2 * y) / 31.0) * 0.8 + math.cos(x / 53.0 - y / 17.0);
  return 0.08 + 0.3 * ((v + 3.0) / 6.0).clamp(0.0, 1.0);
}

int toU8(double lin) => (255 * math.pow(lin.clamp(0.0, 1.0), 1 / 2.2)).floor();
double toLin(int v) => math.pow(v / 255.0, 2.2).toDouble();

Rgb gray(List<List<int>> rows) {
  final h = rows.length, w = rows[0].length;
  final d = Uint8List(w * h * 3);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      d[(y * w + x) * 3] = d[(y * w + x) * 3 + 1] = d[(y * w + x) * 3 + 2] = rows[y][x];
    }
  }
  return Rgb(w, h, d);
}

List<List<double>> offLinear() => [
      for (var y = 0; y < H; y++)
        [for (var x = 0; x < W; x++) inPlate(x, y) ? 0.18 + 0.01 * (hash(x, y, 0) - 0.5) * 3.4 : body(x, y)] // white plate background, a little noise
    ];

Rgb frameOf(List<List<double>> lin) => gray([for (final r in lin) [for (final v in r) toU8(v)]]);

/// `scenes(kind)` of test_active.py: (off, on) as 8-bit frames
(Rgb, Rgb) scenes(String kind) {
  final off = offLinear();
  List<List<double>> on;
  double lit(int x, int y) => 0.9 + 0.1 * hash(x, y, 1);
  switch (kind) {
    case 'real_hsrp': // diffuse paint brightens a little, retroreflective sheeting a lot
      on = [
        for (var y = 0; y < H; y++)
          [for (var x = 0; x < W; x++) inPlate(x, y) ? off[y][x] * 1.25 + 1.4 : off[y][x] * (1 + 0.25 * lit(x, y))]
      ];
    case 'screen': // emissive: flash adds only a faint veiling glare on the glass
      on = [
        for (var y = 0; y < H; y++) [for (var x = 0; x < W; x++) off[y][x] + 0.015 * lit(x, y)]
      ];
    default:
      throw ArgumentError(kind);
  }
  return (frameOf(off), frameOf(on));
}

/// A screen replay whose flash reflection washes out the right [frac] of the plate (glare on the glass).
(Rgb, Rgb) withGlarePatch(double frac, {double gain = 0.9}) {
  final (off, on) = scenes('screen');
  final x0 = 100 + (1 - frac) * 200;
  final rows = [
    for (var y = 0; y < H; y++)
      [
        for (var x = 0; x < W; x++)
          () {
            final edge = 1 / (1 + math.exp(-(x - x0) / 6.0));
            final r = 0.7 + 0.3 * math.exp(-math.pow((y - 120) / 30.0, 2));
            return toU8(toLin(on.at(x, y, 0)) + gain * edge * r);
          }()
      ]
  ];
  return (off, gray(rows));
}

void main() {
  group('flash uniformity and hot-spot', () {
    test('a uniform HSRP gain stays retroreflective and reports its (small) spread', () {
      final (off, on) = scenes('real_hsrp');
      final r = judgeFlash(off, on, plate, hsrpExpected: true);
      expect(r.verdict, 'retroreflective', reason: '${r.reasons}');
      expect(r.hotspot, isFalse);
      expect(r.gainSpread, isNotNull);
      expect(r.gainSpread!, lessThanOrEqualTo(1.5), reason: 'spread ${r.gainSpread}');
    });

    for (final frac in [0.15, 0.25, 0.35]) {
      test('a glare patch on a screen ($frac of the plate) is a specular hot-spot', () {
        final (off, on) = withGlarePatch(frac);
        final r = judgeFlash(off, on, plate, hsrpExpected: true);
        expect(r.verdict, 'specular_hotspot', reason: '${r.verdict} spread=${r.gainSpread} ratio=${r.ratio} ${r.reasons}');
        expect(r.hotspot, isTrue);
        expect(r.gainSpread!, greaterThan(2.5));
        expect(r.reasons.any((x) => x.contains('glare on a glossy surface')), isTrue);
      });
    }

    test('a plain screen replay (no glare) is still not_retroreflective', () {
      final (off, on) = scenes('screen');
      final r = judgeFlash(off, on, plate, hsrpExpected: true);
      expect(r.verdict, 'not_retroreflective', reason: '${r.reasons}');
      expect(r.hotspot, isFalse);
    });

    test('the hot-spot rule needs hsrp_expected and a flashed frame', () {
      final (off, on) = withGlarePatch(0.25);
      expect(judgeFlash(off, on, plate, hsrpExpected: false).verdict, 'inconclusive');
      expect(judgeFlash(off, off, plate, hsrpExpected: true).verdict, 'inconclusive'); // the flash did not fire
    });

    test('a strong but non-uniform gain is not accepted as retroreflective: inconclusive, with the reason', () {
      final (off, on) = scenes('real_hsrp');
      // keep the plate bright overall (most of it gains ~5x) but let its left 30% gain almost nothing
      final rows = [
        for (var y = 0; y < H; y++)
          [
            for (var x = 0; x < W; x++)
              () {
                if (!inPlate(x, y)) return on.at(x, y, 0);
                final lo = toLin(off.at(x, y, 0)), ln = toLin(on.at(x, y, 0));
                final boost = x < 160 ? 0.05 : 1.0;
                return toU8(lo * 1.25 + (ln - lo * 1.25) * boost);
              }()
          ]
      ];
      final r = judgeFlash(off, gray(rows), plate, hsrpExpected: true);
      expect(r.ratio!, greaterThanOrEqualTo(2.5), reason: 'the plate-level gain alone WOULD pass');
      expect(r.hotspot, isFalse);
      expect(r.verdict, 'inconclusive', reason: '${r.verdict} ${r.reasons}');
      expect(r.reasons.any((x) => x.contains('not uniform')), isTrue);
      expect(r.gainSpread!, greaterThan(2.5));
    });

    test('gain_uniformity ignores dark characters and needs enough pixels', () {
      const w = 200, h = 60;
      final lo = toU8(0.2), hi = toU8(0.6), dark = toU8(0.01);
      final off = gray([for (var y = 0; y < h; y++) [for (var x = 0; x < w; x++) x >= 90 && x < 110 ? dark : lo]]);
      final on = gray([for (var y = 0; y < h; y++) [for (var x = 0; x < w; x++) x >= 90 && x < 110 ? dark : hi]]);
      final (spread, hot) = flashGainUniformity(off, on, const Box(0, 0, w, h));
      expect(hot, isFalse);
      expect(spread, closeTo(1.0, 0.05)); // the dark pixels are excluded from the gain map
      final small = gray([for (var y = 0; y < 5; y++) [for (var x = 0; x < 5; x++) lo]]);
      expect(flashGainUniformity(small, small, const Box(0, 0, 5, 5)), (null, false));
    });

    test('alignForGain finds no motion where there is none, even with the flash changing the frame (checks.alignShift slips on it)', () {
      for (final (off, on) in [scenes('real_hsrp'), withGlarePatch(0.25), withGlarePatch(0.15)]) {
        expect(alignForGain(off, on, plate), (0, 0));
      }
    });

    test('alignForGain recovers a hand-shake and the verdict survives it', () {
      final (off, on) = scenes('real_hsrp');
      Rgb shifted(int sx, int sy) {
        final d = Uint8List(W * H * 3);
        for (var y = 0; y < H; y++) {
          for (var x = 0; x < W; x++) {
            final v = on.at(((x - sx) % W + W) % W, ((y - sy) % H + H) % H, 0);
            d[(y * W + x) * 3] = d[(y * W + x) * 3 + 1] = d[(y * W + x) * 3 + 2] = v;
          }
        }
        return Rgb(W, H, d);
      }

      final moved = shifted(8, -4); // content of `off` at p sits at p + (8, -4) in `moved`
      expect(alignForGain(off, moved, plate), (8, -4));
      final r = judgeFlash(off, moved, plate, hsrpExpected: true);
      expect(r.verdict, 'retroreflective', reason: '${r.reasons}');
      expect(r.gainSpread!, lessThanOrEqualTo(1.5));
    });

    test('alignForGain refuses to invent motion in a window too small to search', () {
      final tiny = gray([for (var y = 0; y < 20; y++) [for (var x = 0; x < 30; x++) 100 + x]]);
      expect(alignForGain(tiny, tiny, const Box(5, 5, 25, 15)), (0, 0));
    });

    test('guards come first: a saturated plate is inconclusive whatever the uniformity', () {
      final sat = gray([for (var y = 0; y < H; y++) [for (var x = 0; x < W; x++) 250]]);
      final r = judgeFlash(sat, sat, plate, hsrpExpected: true);
      expect(r.verdict, 'inconclusive');
      expect(r.reasons, isNotEmpty);
      expect(r.gainSpread, isNull);
    });
  });
}
