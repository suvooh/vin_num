import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/enhance.dart';

Map<String, dynamic> golden() => (jsonDecode(File('test/fixtures/golden/image_enhance.json').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage img(Map<String, dynamic> o) => RgbImage(o['w'] as int, o['h'] as int, base64Decode(o['rgb'] as String));

/// Compares a Dart view (3 replicated channels) with the Python grey plane.
({int max, double mean}) diffGray(RgbImage got, Map<String, dynamic> exp) {
  final e = base64Decode(exp['gray'] as String);
  expect([got.width, got.height], [exp['w'], exp['h']]);
  var mx = 0, sum = 0;
  for (var i = 0; i < e.length; i++) {
    expect(got.rgb[3 * i] == got.rgb[3 * i + 1] && got.rgb[3 * i + 1] == got.rgb[3 * i + 2], isTrue, reason: 'views are replicated greys');
    final d = (got.rgb[3 * i] - e[i]).abs();
    sum += d;
    if (d > mx) mx = d;
  }
  return (max: mx, mean: sum / e.length);
}

// Tolerances (grey levels, 0..255). The Python runs its blurs and percentiles in float32, Dart in
// float64, and both truncate to 8 bits, so a value that lands on an integer boundary can differ by
// one level; sharpen / deglare / retinex chain a blur and an amplifying stretch, hence one more.
const tolStretch = 2, tolBlurred = 3;

void main() {
  final cases = (golden()['cases'] as List).cast<Map<String, dynamic>>();

  for (final c in cases) {
    final name = c['name'] as String;
    group('golden: $name', () {
      final crop = img(c['crop'] as Map<String, dynamic>);

      test('stretch', () {
        final d = diffGray(stretch(crop), c['stretch'] as Map<String, dynamic>);
        expect(d.max, lessThanOrEqualTo(tolStretch), reason: 'mean ${d.mean}');
      });

      test('bestChannel', () {
        final d = diffGray(bestChannel(crop), c['channel'] as Map<String, dynamic>);
        expect(d.max, lessThanOrEqualTo(tolStretch), reason: 'mean ${d.mean}');
      });

      test('sharpen', () {
        final d = diffGray(sharpen(crop), c['sharpen'] as Map<String, dynamic>);
        expect(d.max, lessThanOrEqualTo(tolBlurred), reason: 'mean ${d.mean}');
      });

      test('retinex', () {
        final d = diffGray(retinex(crop), c['retinex'] as Map<String, dynamic>);
        expect(d.max, lessThanOrEqualTo(tolBlurred), reason: 'mean ${d.mean}');
      });

      test('deglare applies exactly when Python says it does', () {
        final dg = deglare(crop);
        final exp = c['deglare'] as Map<String, dynamic>?;
        expect(dg == null, exp == null);
        if (dg != null) {
          final d = diffGray(dg, exp!);
          expect(d.max, lessThanOrEqualTo(tolBlurred), reason: 'mean ${d.mean}');
        }
      });

      test('clipMask is bit-exact and clipStats agree', () {
        final m = clipMask(crop);
        expect(m, orderedEquals(base64Decode(c['clip_mask'] as String)));
        final s = clipStats(crop), es = (c['clip_stats'] as Map).cast<String, num>();
        expect(s['clipped_frac'], closeTo(es['clipped_frac']!.toDouble(), 1e-9));
        expect(s['largest_blob'], closeTo(es['largest_blob']!.toDouble(), 1e-9));
      });

      test('variants: orig first, same size, names as expected', () {
        final v = variants(crop);
        expect(v.map((e) => e.$1).toList(), (c['variants_default'] as List).cast<String>());
        expect(v.first.$2, same(crop));
        for (final e in v) {
          expect([e.$2.width, e.$2.height], [crop.width, crop.height]);
        }
      });
    });
  }

  group('behaviour', () {
    RgbImage strokes({int base = 225, int ink = 20}) {
      const w = 120, h = 40;
      final px = Uint8List(w * h * 3)..fillRange(0, w * h * 3, base);
      for (var x = 10; x < w - 10; x += 14) {
        for (var y = 6; y < h - 6; y++) {
          for (var k = 0; k < 4; k++) {
            final i = (y * w + x + k) * 3;
            px[i] = px[i + 1] = px[i + 2] = ink;
          }
        }
      }
      return RgbImage(w, h, px);
    }

    int spread(RgbImage im) {
      var lo = 255, hi = 0;
      for (var i = 0; i < im.rgb.length; i += 3) {
        if (im.rgb[i] < lo) lo = im.rgb[i];
        if (im.rgb[i] > hi) hi = im.rgb[i];
      }
      return hi - lo;
    }

    test('stretch restores the contrast of a crushed crop', () {
      final crushed = strokes(base: 170, ink: 130);
      expect(spread(crushed), 40);
      expect(spread(stretch(crushed)), greaterThan(200));
    });

    test('a flat crop is left alone (no noise amplification)', () {
      final flat = RgbImage(20, 10, Uint8List(600)..fillRange(0, 600, 130));
      expect(spread(stretch(flat)), 0);
    });

    test('deglare is null without clipping and for a fully white crop', () {
      expect(deglare(strokes()), isNull);
      expect(deglare(RgbImage(40, 20, Uint8List(2400)..fillRange(0, 2400, 255))), isNull);
    });

    test('isolated clipped pixels are noise, not glare', () {
      final px = Uint8List(40 * 20 * 3)..fillRange(0, 2400, 200);
      px.setRange(3 * (5 * 40 + 5), 3 * (5 * 40 + 5) + 3, [255, 255, 255]);
      expect(clipMask(RgbImage(40, 20, px)).any((v) => v == 1), isFalse);
    });

    test('viewing set is configurable and orig is always first', () {
      final c = strokes();
      expect(variants(c, views: {'orig'}).map((e) => e.$1), ['orig']);
      expect(variants(c, views: {'orig', 'stretch', 'sharpen', 'channel', 'retinex'}).map((e) => e.$1), ['orig', 'stretch', 'retinex', 'sharpen', 'channel']);
    });

    test('the helpers agree with their numpy / scipy meaning', () {
      expect(percentiles([1, 2, 3, 4, 5], [0, 50, 100, 25]), [1, 3, 5, 2]);
      expect([for (var i = -3; i < 8; i++) reflectIndex(i, 4)], [2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]);
      expect(reflectIndex(5, 1), 0);
    });
  });
}
