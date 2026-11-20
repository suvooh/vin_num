import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/geometry.dart';

Map<String, dynamic> golden() => (jsonDecode(File('test/fixtures/golden/image_geometry.json').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage img(Map<String, dynamic> o) => RgbImage(o['w'] as int, o['h'] as int, base64Decode(o['rgb'] as String));

List<double> nums(dynamic l) => [for (final v in l as List) (v as num).toDouble()];

void main() {
  final g = golden();

  group('minAreaRect matches the Python reference', () {
    final rects = (g['rects'] as List).cast<Map<String, dynamic>>();
    for (var i = 0; i < rects.length; i++) {
      test('case $i (${(rects[i]['points'] as List).length ~/ 2} points)', () {
        final r = minAreaRect(nums(rects[i]['points']));
        final exp = nums(rects[i]['corners']);
        for (var k = 0; k < 8; k++) {
          expect(r.corners[k], closeTo(exp[k], 1e-6), reason: 'corner coordinate $k');
        }
        expect(r.long, closeTo(rects[i]['long'] as double, 1e-6));
        expect(r.short, closeTo(rects[i]['short'] as double, 1e-6));
        expect(r.long, greaterThanOrEqualTo(r.short));
      });
    }

    test('corners are TL, TR, BR, BL: text runs left to right, rows top to bottom', () {
      final pts = <double>[];
      for (var y = 0; y < 20; y++) {
        for (var x = 0; x < 100; x++) {
          pts..add(x.toDouble())..add(y.toDouble());
        }
      }
      final r = minAreaRect(pts);
      expect(r.corners[2], greaterThan(r.corners[0])); // TR right of TL
      expect(r.corners[7], greaterThan(r.corners[1])); // BL below TL
      expect(r.long, closeTo(99, 1e-6));
      expect(r.short, closeTo(19, 1e-6));
    });

    test('degenerate inputs do not throw', () {
      expect(minAreaRect([5, 5]).long, 0);
      expect(minAreaRect([]).corners, hasLength(8));
      expect(minAreaRect([1, 1, 1, 1, 1, 1]).short, 0);
    });
  });

  group('homographyCoeffs', () {
    for (final h in (g['homographies'] as List).cast<Map<String, dynamic>>()) {
      test('matches numpy.linalg.solve for ${h['src']}', () {
        final c = homographyCoeffs(nums(h['dst']), nums(h['src']));
        final exp = nums(h['coeffs']);
        for (var k = 0; k < 8; k++) {
          expect(c[k], closeTo(exp[k], 1e-9 * math.max(1, exp[k].abs())));
        }
      });
    }

    test('maps the destination corners onto the source corners', () {
      final dst = <double>[0, 0, 100, 0, 100, 40, 0, 40], src = <double>[10, 12, 112, 22, 105, 70, 3, 58];
      final c = homographyCoeffs(dst, src);
      for (var k = 0; k < 4; k++) {
        final x = dst[2 * k], y = dst[2 * k + 1];
        final den = c[6] * x + c[7] * y + 1;
        expect((c[0] * x + c[1] * y + c[2]) / den, closeTo(src[2 * k], 1e-6));
        expect((c[3] * x + c[4] * y + c[5]) / den, closeTo(src[2 * k + 1], 1e-6));
      }
    });

    test('a degenerate quad is refused, not turned into NaN', () {
      expect(() => homographyCoeffs(<double>[0, 0, 1, 0, 1, 1, 0, 1], <double>[5, 5, 5, 5, 5, 5, 5, 5]), throwsFormatException);
    });
  });

  group('warpQuad matches Pillow PERSPECTIVE/BICUBIC', () {
    final src = img(g['warp_source'] as Map<String, dynamic>);
    for (final w in (g['warps'] as List).cast<Map<String, dynamic>>()) {
      test(w['name'] as String, () {
        final exp = img(w['out'] as Map<String, dynamic>);
        final got = warpQuad(src, nums(w['quad']), pad: (w['pad'] as num).toDouble());
        expect([got.width, got.height], [exp.width, exp.height]);
        var maxDiff = 0, sum = 0;
        for (var i = 0; i < exp.rgb.length; i++) {
          final d = (got.rgb[i] - exp.rgb[i]).abs();
          sum += d;
          if (d > maxDiff) maxDiff = d;
        }
        // Same kernel, same sampling grid, same truncation: only float rounding can differ.
        expect(maxDiff, lessThanOrEqualTo(1), reason: 'mean abs diff ${sum / exp.rgb.length}');
        expect(sum / exp.rgb.length, lessThan(0.02));
      });
    }

    test('vertical text comes out upright (wider than tall)', () {
      final w = (g['warps'] as List).cast<Map<String, dynamic>>().firstWhere((e) => e['name'] == 'vertical_text');
      final got = warpQuad(src, nums(w['quad']));
      expect(got.width, greaterThan(got.height));
    });
  });

  group('TextBox', () {
    const b = TextBox([10, 20, 110, 20, 110, 50, 10, 50], 0.9);
    test('size, centre, angle, bbox', () {
      expect(b.width, closeTo(100, 1e-9));
      expect(b.height, closeTo(30, 1e-9));
      expect(b.center, (60.0, 35.0));
      expect(b.angle, closeTo(0, 1e-9));
      expect(b.bbox, (10, 20, 110, 50));
    });

    test('a tilted box reports its direction and a floor/ceil bbox', () {
      final c = math.cos(math.pi / 12), s = math.sin(math.pi / 12);
      final t = TextBox([0, 0, 100 * c, 100 * s, 100 * c - 30 * s, 100 * s + 30 * c, -30 * s, 30 * c], 1);
      expect(t.angle, closeTo(15, 1e-9));
      expect(t.width, closeTo(100, 1e-9));
      final (x0, y0, x1, y1) = t.bbox;
      expect(x0, lessThan(0));
      expect(y0, 0);
      expect(x1, greaterThanOrEqualTo(97));
      expect(y1, greaterThanOrEqualTo(55));
    });
  });

  test('warpQuad never reads outside the image and keeps pixel layout RGB', () {
    final src = RgbImage(4, 4, Uint8List.fromList([for (var i = 0; i < 16; i++) ...[200, 100, 50]]));
    final out = warpQuad(src, [-10, -10, 20, -10, 20, 20, -10, 20]);
    expect(out.width, greaterThanOrEqualTo(2));
    expect(out.rgb[0], 200);
    expect(out.rgb[1], 100);
    expect(out.rgb[2], 50);
  });
}
