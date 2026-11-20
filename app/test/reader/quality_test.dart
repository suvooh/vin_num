import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/quality.dart';

Map<String, dynamic> golden() => (jsonDecode(File('test/fixtures/golden/image_enhance.json').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage img(Map<String, dynamic> o) => RgbImage(o['w'] as int, o['h'] as int, base64Decode(o['rgb'] as String));

void main() {
  final cases = (golden()['cases'] as List).cast<Map<String, dynamic>>();

  group('assessCrop matches the Python gate on golden crops', () {
    for (final c in cases) {
      test(c['name'] as String, () {
        final exp = (c['quality'] as Map).cast<String, dynamic>();
        final r = assessCrop(img(c['crop'] as Map<String, dynamic>));
        final em = (exp['metrics'] as Map).cast<String, num>();
        for (final k in em.keys) {
          expect(r.metrics[k], isNotNull, reason: k);
          // 1e-2 as specified: metrics are rounded to 1-4 decimals, sharpness to 2
          expect(r.metrics[k]!, closeTo(em[k]!.toDouble(), 1e-2), reason: 'metric $k');
        }
        expect(r.metrics.keys.toSet(), em.keys.toSet());
        expect(r.issues, (exp['issues'] as List).cast<String>());
        expect(r.guidance, (exp['guidance'] as List).cast<String>());
        expect(r.ok, exp['ok']);
      });
    }
  });

  group('verdicts', () {
    RgbImage solid(int w, int h, int v) => RgbImage(w, h, Uint8List(w * h * 3)..fillRange(0, w * h * 3, v));

    test('a tiny crop is flagged as too small with advice', () {
      final r = assessCrop(solid(60, 12, 128));
      expect(r.issues, contains('too small'));
      expect(r.ok, isFalse);
      expect(r.guidance.join(' '), contains('Move closer'));
    });

    test('toJson mirrors the Python as_dict', () {
      final j = assessCrop(solid(80, 30, 128)).toJson();
      expect(j.keys.toSet(), {'ok', 'metrics', 'issues', 'guidance', 'calibrated'});
      expect(j['calibrated'], isFalse);
    });

    test('thresholds can be overridden', () {
      final strict = assessCrop(solid(80, 30, 128), thresholds: {'min_height_px': 40});
      expect(strict.issues, contains('too small'));
    });

    test('a very bright crop is overexposed and a very dark one is too dark', () {
      expect(assessCrop(solid(100, 40, 240)).issues, contains('overexposed'));
      expect(assessCrop(solid(100, 40, 10)).issues, contains('too dark'));
    });
  });
}
