// Tests of the on-device orchestration (scan_pipeline.dart, padnet.dart, photo_decode.dart).
//
// What is REAL here: role routing, best-of selection, the JSON the app parses, the flash test
// (checks.dart), parallax (planarity.dart), the liveness decision (liveness.dart), provenance
// scanning, and JPEG/PNG decoding. What is FAKED: the scene reader (needs ONNX Runtime, which has no
// native library under `flutter test`) and the PadNet model (same reason) - their callers are
// tested against hand-written stand-ins. The real PP-OCR / PadNet calls can only run on a device.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:vinplate_tester/data/models.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/offline/engine_host.dart' show EnginePhoto;
import 'package:vinplate_tester/offline/padnet.dart';
import 'package:vinplate_tester/offline/photo_decode.dart';
import 'package:vinplate_tester/offline/scan_pipeline.dart';
import 'package:vinplate_tester/reader/grammar.dart' show parsePlate;
import 'package:vinplate_tester/reader/plate_category.dart';

// ------------------------------------------------------------------------------ fixtures
Map<String, dynamic> _fixture() => (jsonDecode(File('test/fixtures/scan_plate_accept.json').readAsStringSync()) as Map).cast<String, dynamic>();
Map<String, dynamic> _plateJson() => ((_fixture()['fields'] as Map)['plate'] as Map).cast<String, dynamic>();

const Box4 roi = (100, 90, 220, 150);

ReadField plateField({String status = 'accept', int image = 0, Box4 box = roi, String text = 'MH12AB1234', String? masked, Rank? rank}) {
  final j = _plateJson()
    ..['status'] = status
    ..['image'] = image
    ..['box'] = [box.$1, box.$2, box.$3, box.$4]
    ..['value'] = text
    ..['masked'] = masked ?? text;
  return ReadField(
    domain: 'plate',
    status: status,
    rank: rank ?? (status == 'accept' ? (2, 10, 0.99) : (1, 8, 0.6)),
    image: image,
    box: box,
    text: text,
    masked: masked ?? text,
    json: j,
  );
}

ReadField vinField({String status = 'accept', int image = 0, String text = 'MAL907KE1E6602791', bool? checkDigit = true}) {
  final j = <String, dynamic>{
    'domain': 'vin',
    'status': status,
    'value': text,
    'masked': text,
    'confidence': 0.9,
    'parsed': {'layout': 'vin17', 'vin': text, 'check_digit_ok': checkDigit, 'known_maker': null},
    'box': [10, 10, 200, 40],
    'image': image,
  };
  return ReadField(domain: 'vin', status: status, rank: status == 'accept' ? (2, 17, 0.9) : (1, 9, 0.5), image: image, box: (10, 10, 200, 40), text: text, masked: text, json: j);
}

class Call {
  Call(this.n, this.domains, this.legacy);
  final int n;
  final Set<String> domains;
  final bool legacy;
}

class FakeReader implements SceneReaderPort {
  FakeReader([this.script]);
  final calls = <Call>[];
  ReadScene Function(Call c)? script;

  @override
  ReadScene readFrames(List<RgbImage> rgbs, {required Set<String> domains, bool allowLegacyVin = false}) {
    final c = Call(rgbs.length, domains, allowLegacyVin);
    calls.add(c);
    if (script != null) return script!(c);
    // default: every photo shows the plate in the ROI
    final one = ReadScene(plate: domains.contains('plate') ? plateField() : null, lines: const [
      {'text': 'MH12AB1234', 'box': [100, 90, 220, 150], 'detector_score': 0.9}
    ]);
    return rgbs.length == 1
        ? one
        : ReadScene(plate: one.plate, lines: one.lines, perImage: [for (var i = 0; i < rgbs.length; i++) one], timings: const {'detect': 5.0, 'recognise': 3.0, 'decode': 2.0});
  }
}

class FakePad implements PadScorer {
  FakePad(this.value, {this.throws = false, this.note});
  final double value;
  final bool throws;
  final String? note;
  final seen = <(int, int, Box4?)>[];
  bool disposed = false;

  @override
  (double?, String?) score(RgbImage frame, Box4? roi) {
    seen.add((frame.width, frame.height, roi));
    if (throws) throw StateError('boom');
    return (value, note);
  }

  @override
  void dispose() => disposed = true;
}

// ------------------------------------------------------------ synthetic frames (toy physics)
int _noise(int x, int y, int seed) => (((x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)) & 0xFF) % 17 - 8;

/// kind 1: a plain frame (plate region 80, surround 70); 2: the SAME scene with flash on a retroreflective
/// plate (plate x2.5 brighter, surround barely); 3: a replayed screen (everything +10%); 4: another aspect ratio.
RgbImage synth(int kind, int seed, {int w = 320, int h = 240}) {
  if (kind == 4) {
    w = 200;
    h = 300;
  }
  final out = Uint8List(w * h * 3);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final inPlate = x >= 100 && x < 220 && y >= 90 && y < 150;
      final base = switch (kind) {
        2 => inPlate ? 200 : 80,
        3 => inPlate ? 88 : 77,
        _ => inPlate ? 80 : 70,
      };
      final v = (base + _noise(x, y, seed)).clamp(0, 255);
      final o = (y * w + x) * 3;
      out[o] = v;
      out[o + 1] = v;
      out[o + 2] = v;
    }
  }
  return RgbImage(w, h, out);
}

/// Bytes that the fake decoder understands: [kind, seed].
EnginePhoto photo(String role, int kind, int seed, [String? name]) => EnginePhoto(role, name ?? '$role.jpg', Uint8List.fromList([kind, seed]));

DecodedPhoto fakeDecode(Uint8List b) => DecodedPhoto(synth(b[0], b[1]));

ScanPipeline pipe(FakeReader r, {PadScorer? pad, String? padWhy}) => ScanPipeline(reader: r, pad: pad, decode: fakeDecode, padUnavailableReason: padWhy);

Map<String, dynamic> auth(Map<String, dynamic> out) => (out['authenticity'] as Map).cast<String, dynamic>();
Map<String, dynamic> evidence(Map<String, dynamic> out) => (auth(out)['evidence'] as Map).cast<String, dynamic>();
Map<String, dynamic> notRunWhy(Map<String, dynamic> out) => (evidence(out)['not_run_reasons'] as Map).cast<String, dynamic>();

void main() {
  group('role routing', () {
    test('plate + angle photos are read for a plate, vin photos for a VIN (legacy allowed), auto photos for both', () {
      final r = FakeReader();
      final out = pipe(r).run([
        photo('plate', 1, 1),
        photo('plate_flash', 2, 1),
        photo('angle', 1, 2),
        photo('vin', 1, 3),
        photo('auto', 1, 4),
      ], hsrp: true, requireLiveness: false);
      expect(r.calls, hasLength(3));
      final plateCall = r.calls.firstWhere((c) => c.domains.length == 1 && c.domains.contains('plate'));
      expect(plateCall.n, 2); // plate + angle; the flash frame is NOT read
      expect(plateCall.legacy, isFalse);
      final vinCall = r.calls.firstWhere((c) => c.domains.contains('vin') && c.domains.length == 1);
      expect(vinCall.n, 1);
      expect(vinCall.legacy, isTrue);
      final autoCall = r.calls.firstWhere((c) => c.domains.length == 2);
      expect(autoCall.n, 1);
      expect(autoCall.legacy, isFalse);
      expect((out['images'] as List).map((i) => (i as Map)['role']), ['plate', 'plate_flash', 'angle', 'vin', 'auto']);
    });

    test('an unknown role is treated as auto', () {
      final r = FakeReader();
      pipe(r).run([photo('selfie', 1, 1)], hsrp: false, requireLiveness: false);
      expect(r.calls.single.domains, {'plate', 'vin'});
    });

    test('nothing is read from a group that has no photos', () {
      final r = FakeReader();
      pipe(r).run([photo('vin', 1, 1)], hsrp: false, requireLiveness: false);
      expect(r.calls.single.domains, {'vin'});
    });

    test('no photos at all is a programming error, not a result', () {
      expect(() => pipe(FakeReader()).run([], hsrp: false, requireLiveness: false), throwsArgumentError);
    });
  });

  group('best-of selection', () {
    test('the stronger reading wins across groups and its photo index is GLOBAL', () {
      // photos: 0 vin, 1 plate, 2 auto, 3 angle. plate group = [1, 3]; auto group = [2]
      final r = FakeReader((c) {
        if (c.domains.length == 2) return ReadScene(plate: plateField(status: 'accept', image: 0), vin: vinField(status: 'review'));
        if (c.domains.contains('vin')) return ReadScene(vin: vinField(status: 'accept', image: 0));
        return ReadScene(plate: plateField(status: 'review', image: 1, masked: 'MH12?B1234'), perImage: [ReadScene(plate: plateField(status: 'review')), ReadScene(plate: plateField(status: 'review'))]);
      });
      final out = pipe(r).run([photo('vin', 1, 1), photo('plate', 1, 2), photo('auto', 1, 3), photo('angle', 1, 4)], hsrp: false, requireLiveness: false);
      final plate = (out['fields'] as Map)['plate'] as Map;
      final vin = (out['fields'] as Map)['vin'] as Map;
      expect(plate['status'], 'accept'); // the auto photo's accepted read beat the plate group's review
      expect(plate['image'], 2); // auto photo = global index 2
      expect(vin['status'], 'accept');
      expect(vin['image'], 0); // vin photo = global index 0 (beats the auto photo's review)
    });

    test('on a tie the first candidate (the dedicated plate group) wins, like Python max()', () {
      final r = FakeReader((c) => c.domains.length == 2
          ? ReadScene(plate: plateField(image: 0, text: 'AUTO0000'))
          : ReadScene(plate: plateField(image: 0, text: 'PLATEGRP')));
      final out = pipe(r).run([photo('plate', 1, 1), photo('auto', 1, 2)], hsrp: false, requireLiveness: false);
      expect(((out['fields'] as Map)['plate'] as Map)['value'], 'PLATEGRP');
    });

    test('rank comparison is lexicographic (status, evidence, confidence)', () {
      expect(compareRank((2, 5, 0.1), (1, 17, 0.99)), greaterThan(0));
      expect(compareRank((1, 9, 0.5), (1, 8, 0.99)), greaterThan(0));
      expect(compareRank((1, 9, 0.5), (1, 9, 0.6)), lessThan(0));
      expect(compareRank((1, 9, 0.5), (1, 9, 0.5)), 0);
    });
  });

  group('response shape', () {
    test('field_out additions: value only when actionable, display, needs_review, VIN fields', () {
      final acc = fieldOut(plateField(), 3);
      expect(acc['image'], 3);
      expect(acc['value'], 'MH12AB1234');
      expect(acc['display'], 'MH 12 AB 1234'); // from parsed.display
      expect(acc['needs_review'], isFalse);
      final rej = fieldOut(plateField(status: 'reject', text: 'XX'), 0);
      expect(rej['value'], isNull); // nothing to act on
      expect(rej['needs_review'], isTrue);
      final v = fieldOut(vinField(checkDigit: false), 1);
      expect(v['check_digit_ok'], isFalse);
      expect(v['read_length'], 17);
      expect(fieldOut(plateField(), 0).containsKey('check_digit_ok'), isFalse);
    });

    test('the output parses with ScanResult.fromJson, serialises to JSON, and has the documented keys', () {
      final r = FakeReader((c) => ReadScene(plate: plateField(), vin: c.domains.contains('vin') ? vinField() : null, lines: const [
            {'text': 'MH12AB1234', 'box': [1, 2, 3, 4], 'detector_score': 0.9}
          ], timings: const {'detect': 11.0, 'recognise': 7.0, 'decode': 3.0}));
      final out = pipe(r, pad: FakePad(0.05)).run([photo('plate', 1, 1), photo('vin', 1, 2)], hsrp: true, requireLiveness: false);
      expect(out.keys, containsAll(['fields', 'authenticity', 'conflicts', 'images', 'lines', 'timings_ms']));
      expect(out.containsKey('capture_id'), isFalse); // the scanner adds it
      jsonEncode(out); // must be pure JSON data (it crosses an isolate boundary)
      final res = ScanResult.fromJson(out);
      expect(res.plate!.value, 'MH12AB1234');
      expect(res.vin!.domain, 'vin');
      expect(res.photos.map((p) => p.role), ['plate', 'vin']);
      expect(res.photos.first.width, 320);
      expect(res.auth.liveness, 'not_established'); // no flash / angle photo
      final t = (out['timings_ms'] as Map).cast<String, double>();
      expect(t['detect'], 11.0 + 11.0); // summed over the groups that were read
      expect(t.keys, containsAll(['recognise', 'decode', 'padnet', 'authenticity', 'total']));
      final lines = (out['lines'] as List).cast<Map>();
      expect(lines.map((l) => l['image']).toSet(), {0, 1});
      expect(lines.first.keys, containsAll(['image', 'text', 'box', 'detector_score']));
    });

    test('the vehicle category worked out by the reader reaches ScanResult.plate.category (and a VIN has none)', () {
      RgbImage crop((int, int, int) bg, (int, int, int) ink) {
        final d = Uint8List(48 * 240 * 3);
        for (var i = 0; i < 48 * 240; i++) {
          d[i * 3] = bg.$1;
          d[i * 3 + 1] = bg.$2;
          d[i * 3 + 2] = bg.$3;
        }
        for (var x = 0; x < 240; x += 16) {
          for (var y = 9; y < 38; y++) {
            for (var xx = x; xx < x + 4; xx++) {
              final i = (y * 240 + xx) * 3;
              d[i] = ink.$1;
              d[i + 1] = ink.$2;
              d[i + 2] = ink.$3;
            }
          }
        }
        return RgbImage(240, 48, d);
      }

      for (final (bg, ink, key, ev, colour) in <((int, int, int), (int, int, int), String, bool, String)>[
        ((255, 206, 22), (18, 18, 18), 'commercial', false, 'yellow'),
        ((22, 150, 72), (248, 248, 246), 'ev_private', true, 'green'),
      ]) {
        final category = classifyPlate(crop(bg, ink), parsePlate('MH12AB1234'));
        final plate = plateField();
        plate.json['category'] = category; // what FieldReading.toJson() puts there
        final r = FakeReader((c) => ReadScene(plate: plate, vin: c.domains.contains('vin') ? vinField() : null));
        final out = pipe(r, pad: FakePad(0.05)).run([photo('plate', 1, 1), photo('vin', 1, 2)], hsrp: true, requireLiveness: false);
        final fields = (out['fields'] as Map).cast<String, dynamic>();
        expect(jsonEncode((fields['plate'] as Map)['category']), jsonEncode(category), reason: 'the dict crosses the pipeline untouched');
        expect((fields['vin'] as Map).containsKey('category'), isFalse);
        jsonEncode(out); // still pure JSON (it crosses an isolate boundary)
        final res = ScanResult.fromJson(out);
        expect(res.plate!.category, isNotNull);
        expect(res.plate!.category!.key, key);
        expect(res.plate!.category!.ev, ev);
        expect(res.plate!.category!.bh, isFalse);
        expect(res.plate!.category!.colourName, colour);
        expect(res.plate!.category!.label, categoryLabels[key]);
        expect(res.plate!.category!.reasons.first, 'plate background looks $colour');
        expect(res.vin!.category, isNull);
      }
    });

    test('a plate without a category (a rejected or pixel-less read) still parses: category is null', () {
      final plate = plateField();
      plate.json['category'] = null;
      final out = pipe(FakeReader((c) => ReadScene(plate: plate)), pad: FakePad(0.05)).run([photo('plate', 1, 1)], hsrp: true, requireLiveness: false);
      expect(ScanResult.fromJson(out).plate!.category, isNull);
    });

    test('nothing found is two nulls, not an error', () {
      final out = pipe(FakeReader((c) => const ReadScene())).run([photo('auto', 1, 1)], hsrp: false, requireLiveness: false);
      expect(out['fields'], {'plate': null, 'vin': null});
      expect(ScanResult.fromJson(out).foundAnything, isFalse);
    });

    test('conflicts between photos are mapped to global photo indexes', () {
      // photos: 0 vin, 1 auto, 2 auto -> the auto group is [1, 2]; a conflict on its local image 1 is global photo 2
      final r = FakeReader((c) => c.domains.length == 2
          ? const ReadScene(conflicts: [
              {'domain': 'plate', 'image': 1, 'masked': 'KA01XY9876', 'agrees_with': [0], 'main': 'MH12AB1234'}
            ])
          : const ReadScene());
      final out = pipe(r).run([photo('vin', 1, 1), photo('auto', 1, 2), photo('auto', 1, 3)], hsrp: false, requireLiveness: false);
      final c = (out['conflicts'] as List).single as Map;
      expect(c['image'], 2);
      expect(c['agrees_with'], [1]);
      expect(ScanResult.fromJson(out).conflicts.single, contains('photo 3'));
    });

    test('a real JPEG goes through the real decoder inside the pipeline', () {
      final bytes = File('test/fixtures/plate.jpg').readAsBytesSync();
      final out = ScanPipeline(reader: FakeReader()).run([EnginePhoto('plate', 'plate.jpg', bytes)], hsrp: false, requireLiveness: false);
      expect(((out['images'] as List).single as Map)['size'], [900, 560]);
    });
  });

  group('authenticity inputs (REAL flash test, parallax, liveness)', () {
    List<EnginePhoto> bundle({int flashKind = 2}) => [photo('plate', 1, 1), photo('plate_flash', flashKind, 1), photo('angle', 1, 2)];

    test('a genuine flash + angle bundle: the flash test passes and liveness is confirmed', () {
      final pad = FakePad(0.05);
      final out = pipe(FakeReader(), pad: pad).run(bundle(), hsrp: true, requireLiveness: false);
      final a = auth(out);
      expect(a['liveness'], 'confirmed');
      expect(a['positive'], contains('retroreflection_passed'));
      expect(evidence(out)['liveness_basis'], contains('retroreflection_passed'));
      // the primary comes first; then the others in photo order
      expect((a['per_frame'] as List).map((f) => (f as Map)['role']), ['primary', 'flash_on', 'angle']);
      expect(((evidence(out)['planarity_by_frame']) as Map).keys, ['2:angle.jpg']); // parallax was attempted for the angle frame
      // PadNet saw every frame; the flash frame (no box of its own) used the PRIMARY's plate region
      expect(pad.seen, hasLength(3));
      expect(pad.seen.map((s) => s.$3), everyElement(roi));
    });

    test('a replayed screen (flash does not change the plate) is a suspected spoof, not a pass', () {
      final out = pipe(FakeReader(), pad: FakePad(0.05)).run(bundle(flashKind: 3), hsrp: true, requireLiveness: false);
      expect(auth(out)['liveness'], 'suspected_spoof');
      expect(auth(out)['positive'], isNot(contains('retroreflection_passed')));
    });

    test('hsrp=false: the flash test does not apply and liveness is not confirmed by it', () {
      final out = pipe(FakeReader(), pad: FakePad(0.05)).run(bundle(), hsrp: false, requireLiveness: false);
      expect(auth(out)['liveness'], isNot('confirmed'));
      expect(auth(out)['not_run'], contains('retroreflection'));
    });

    test('a flash frame of another aspect ratio cannot be compared: not run, with the reason', () {
      final out = pipe(FakeReader()).run([photo('plate', 1, 1), photo('plate_flash', 4, 1)], hsrp: true, requireLiveness: false);
      expect(auth(out)['not_run'], contains('retroreflection'));
      expect(notRunWhy(out)['retroreflection'], contains('different aspect ratio'));
      expect(auth(out)['liveness'], 'not_established');
    });

    test('no plate region found: the flash test is not run (never a pass)', () {
      final out = pipe(FakeReader((c) => const ReadScene())).run([photo('plate', 1, 1), photo('plate_flash', 2, 1)], hsrp: true, requireLiveness: false);
      expect(notRunWhy(out)['retroreflection'], contains('no plate region'));
      expect(auth(out)['liveness'], 'not_established');
    });

    test('an angle frame of another aspect ratio is skipped with the reason', () {
      final out = pipe(FakeReader()).run([photo('plate', 1, 1), photo('angle', 4, 2)], hsrp: false, requireLiveness: false);
      expect(auth(out)['not_run'], contains('planarity'));
      expect(notRunWhy(out)['planarity'], contains('differ in aspect ratio'));
    });

    test('flash / angle photos without a plate photo are handled gracefully (not run, with a reason)', () {
      final out = pipe(FakeReader()).run([photo('plate_flash', 2, 1), photo('angle', 1, 2)], hsrp: true, requireLiveness: false);
      expect(notRunWhy(out)['retroreflection'], contains('normal plate photo'));
      expect(notRunWhy(out)['planarity'], contains('normal plate photo'));
      expect(auth(out)['liveness'], 'not_established');
    });

    test('require_liveness turns "not established" into a review', () {
      final out = pipe(FakeReader()).run([photo('plate', 1, 1)], hsrp: true, requireLiveness: true);
      expect(auth(out)['status'], isNot('no_flags'));
      expect((auth(out)['reasons'] as List).join(' '), contains('liveness required'));
    });

    test('two plate photos that read differently are reported and send the capture to review', () {
      final r = FakeReader((c) => ReadScene(
            plate: plateField(),
            perImage: [ReadScene(plate: plateField(masked: 'MH12AB1234')), ReadScene(plate: plateField(masked: 'KA01XY9876'))],
          ));
      final out = pipe(r).run([photo('plate', 1, 1), photo('angle', 1, 2)], hsrp: false, requireLiveness: false);
      expect((auth(out)['reasons'] as List).join(' '), contains('plate text differs between frames'));
      expect(auth(out)['status'], isNot('no_flags'));
    });

    test('pixel forensics is always listed as not available on the device (never a silent pass)', () {
      final out = pipe(FakeReader(), pad: FakePad(0.05)).run([photo('plate', 1, 1)], hsrp: false, requireLiveness: false);
      expect(auth(out)['not_run'], contains('forensics'));
      expect(notRunWhy(out)['forensics'], contains('not available on device'));
    });
  });

  group('graceful degradation (a failing check never aborts the scan)', () {
    test('PadNet missing: the reading is returned, the recapture check is reported as not run with the reason', () {
      final out = pipe(FakeReader(), padWhy: 'the screen / print model is not bundled with this build').run([photo('plate', 1, 1)], hsrp: false, requireLiveness: false);
      expect(((out['fields'] as Map)['plate'] as Map)['value'], 'MH12AB1234');
      expect(auth(out)['not_run'], contains('recapture'));
      expect(notRunWhy(out)['recapture'], contains('not bundled'));
      expect(evidence(out)['recapture_by_frame'], isEmpty);
    });

    test('PadNet throws: recorded as not run, the scan still completes with its reading', () {
      final out = pipe(FakeReader(), pad: FakePad(0, throws: true)).run([photo('plate', 1, 1)], hsrp: false, requireLiveness: false);
      expect(((out['fields'] as Map)['plate'] as Map)['status'], 'accept');
      expect(auth(out)['not_run'], contains('recapture'));
      expect(notRunWhy(out)['recapture'], contains('the screen / print model failed: boom'));
      expect((auth(out)['reasons'] as List).join(' '), contains('boom'));
    });

    test('PadNet notes (small plate judged by its surroundings) are surfaced', () {
      final out = pipe(FakeReader(), pad: FakePad(0.1, note: 'plate smaller than one native patch: recapture model judged the surrounding area'))
          .run([photo('plate', 1, 1)], hsrp: false, requireLiveness: false);
      expect((auth(out)['reasons'] as List).join(' '), contains('judged the surrounding area'));
      expect(auth(out)['not_run'], isNot(contains('recapture'))); // it DID run
      expect((evidence(out)['recapture_by_frame'] as Map).values.single, 0.1);
    });

    test('a bad photo is a FormatException (the worker reports it as a bad photo)', () {
      expect(() => ScanPipeline(reader: FakeReader()).run([EnginePhoto('plate', 'x.jpg', Uint8List.fromList([1, 2, 3, 4]))], hsrp: false, requireLiveness: false),
          throwsFormatException);
    });

    test('a reader that cannot find anything still yields a complete, parseable response', () {
      final out = pipe(FakeReader((c) => const ReadScene()), pad: FakePad(0.05)).run([photo('plate', 1, 1), photo('plate_flash', 2, 1), photo('angle', 1, 2)], hsrp: true, requireLiveness: false);
      expect(ScanResult.fromJson(out).foundAnything, isFalse);
      expect(auth(out)['liveness'], isNot('confirmed')); // no plate region, no flash verdict
    });
  });

  group('PadNet patch selection (recaptureScore)', () {
    RgbImage noisy(int w, int h) {
      final b = Uint8List(w * h * 3);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final v = (128 + _noise(x, y, 7) * 6).clamp(0, 255);
          final o = (y * w + x) * 3;
          b[o] = v;
          b[o + 1] = v;
          b[o + 2] = v;
        }
      }
      return RgbImage(w, h, b);
    }

    test('a big plate region: 4 native patches, no note', () {
      Float32List? got;
      final (s, note) = recaptureScore(noisy(640, 480), (100, 100, 400, 300), (p) {
        got = p;
        return 0.3;
      });
      expect(s, 0.3);
      expect(note, isNull);
      expect(got!.length, 4 * 3 * 96 * 96);
    });

    test('a plate smaller than one patch: the surrounding window is judged, and the note says so', () {
      var ran = false;
      final (s, note) = recaptureScore(noisy(640, 480), (300, 220, 360, 250), (p) {
        ran = true;
        return 0.2;
      });
      expect(ran, isTrue);
      expect(s, 0.2);
      expect(note, contains('judged the surrounding area'));
    });

    test('no region: the whole frame is used', () {
      final (s, note) = recaptureScore(noisy(320, 240), null, (p) => 0.4);
      expect(s, 0.4);
      expect(note, isNull);
    });

    test('a frame smaller than one patch: not run, with the reason', () {
      var ran = false;
      final (s, note) = recaptureScore(noisy(80, 60), (10, 10, 50, 30), (p) {
        ran = true;
        return 0.9;
      });
      expect(ran, isFalse);
      expect(s, isNull);
      expect(note, contains('region smaller than one native patch'));
    });

    test('PadNetScorer.tryLoad never throws: no model, or a model that cannot be loaded, is a null plus a reason', () {
      String? why;
      expect(PadNetScorer.tryLoad(null, error: (r) => why = r), isNull);
      expect(why, contains('not bundled'));
      why = null;
      expect(PadNetScorer.tryLoad(Uint8List.fromList([1, 2, 3]), error: (r) => why = r), isNull); // no native ONNX Runtime here, and garbage bytes
      expect(why, contains('could not be loaded'));
    });
  });

  group('decodePhoto (real decoder)', () {
    Uint8List jpeg(img.Image im, {int quality = 90}) => Uint8List.fromList(img.encodeJpg(im, quality: quality));

    test('returns upright 3-channel pixels at the photo size when it is small enough', () {
      final im = img.Image(width: 400, height: 200);
      img.fill(im, color: img.ColorRgb8(10, 120, 240));
      final d = decodePhoto(jpeg(im));
      expect((d.rgb.width, d.rgb.height), (400, 200));
      expect(d.rgb.rgb.length, 400 * 200 * 3);
      expect(d.rgb.rgb[0], closeTo(10, 6)); // red channel first (RGB order)
      expect(d.rgb.rgb[2], closeTo(240, 6));
      expect(d.takenAt, isNull);
    });

    test('the longest side is capped at 1600 and the aspect ratio is kept', () {
      final im = img.Image(width: 2400, height: 1200);
      img.fill(im, color: img.ColorRgb8(200, 200, 200));
      final d = decodePhoto(jpeg(im, quality: 60));
      expect((d.rgb.width, d.rgb.height), (1600, 800));
    });

    test('EXIF orientation is applied (a phone photo held upright)', () {
      final im = img.Image(width: 300, height: 200);
      img.fill(im, color: img.ColorRgb8(90, 90, 90));
      im.exif.imageIfd.orientation = 6; // rotate 90 degrees clockwise to display
      final d = decodePhoto(jpeg(im));
      expect((d.rgb.width, d.rgb.height), (200, 300));
    });

    test('orientation is applied to a downscaled photo too', () {
      final im = img.Image(width: 2400, height: 1200);
      img.fill(im, color: img.ColorRgb8(90, 90, 90));
      im.exif.imageIfd.orientation = 6;
      final d = decodePhoto(jpeg(im, quality: 50));
      expect((d.rgb.width, d.rgb.height), (800, 1600));
    });

    test('the EXIF capture time is returned as ISO text', () {
      final im = img.Image(width: 64, height: 64);
      img.fill(im, color: img.ColorRgb8(1, 2, 3));
      im.exif.exifIfd['DateTimeOriginal'] = '2026:11:11 15:30:05';
      expect(decodePhoto(jpeg(im)).takenAt, '2026-11-11T15:30:05');
    });

    test('PNG works too', () {
      final im = img.Image(width: 50, height: 40);
      img.fill(im, color: img.ColorRgb8(5, 6, 7));
      final d = decodePhoto(Uint8List.fromList(img.encodePng(im)));
      expect((d.rgb.width, d.rgb.height), (50, 40));
    });

    test('garbage and empty input are FormatExceptions', () {
      expect(() => decodePhoto(Uint8List.fromList([0, 1, 2, 3, 4, 5])), throwsFormatException);
      expect(() => decodePhoto(Uint8List(0)), throwsFormatException);
    });
  });
}
