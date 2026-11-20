// OfflineEngine wired to the REAL scene reader, decoder, geometry, flash test and liveness decision,
// with only the ONNX Runtime layer replaced by a stand-in recogniser (flutter test has no native ONNX
// Runtime). Proves the pieces fit and the JSON the app parses comes out. It cannot prove that the real
// PP-OCR / PadNet models read real photos: that needs the phone.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:vinplate_tester/data/models.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/offline/engine_host.dart';
import 'package:vinplate_tester/offline/offline_engine.dart';
import 'package:vinplate_tester/offline/padnet.dart';
import 'package:vinplate_tester/reader/geometry.dart';
import 'package:vinplate_tester/reader/pp_engine.dart';
import 'package:vinplate_tester/reader/scene_reader.dart';
import 'package:vinplate_tester/reader/types.dart';

/// The log-probabilities a trained CTC recogniser gives for [text]: one spike per character
/// (3 frames each, as in ml/vinplate/testing.py `make_logp`), blanks in between.
Float64List peaky(String text, {double peak = 0.93}) {
  const eps = 0.002, fpc = 3;
  final rows = <List<double>>[];
  List<double> row(int hot, double p) {
    final r = List<double>.filled(kVocab, eps);
    r[hot] = p;
    final z = r.reduce((a, b) => a + b);
    return [for (final v in r) v / z];
  }

  for (final ch in text.split('')) {
    final k = kChars.indexOf(ch) + 1;
    for (var f = 0; f < fpc; f++) {
      rows.add(f == 0 ? row(k, peak) : row(0, 0.95));
    }
    rows.add(row(0, 0.95));
  }
  final out = Float64List(rows.length * kVocab);
  for (var t = 0; t < rows.length; t++) {
    for (var v = 0; v < kVocab; v++) {
      out[t * kVocab + v] = math.log(rows[t][v]);
    }
  }
  return out;
}

/// [peaky] with the spike row of character [i] replaced by an explicit class distribution (the glare-hit slot of
/// HR88B8888: mostly "unknown", a little B).
Float64List peakyWithSlot(String text, int i, Map<String, double> masses) {
  final out = peaky(text);
  final p = List<double>.filled(kVocab, 0.002);
  masses.forEach((ch, v) => p[kChars.indexOf(ch) + 1] = v);
  final z = p.reduce((a, b) => a + b);
  for (var v = 0; v < kVocab; v++) {
    out[4 * i * kVocab + v] = math.log(p[v] / z); // 3 frames + a blank row per character
  }
  return out;
}

/// A recogniser stand-in that "finds" one text line in a fixed place and "reads" a fixed string from
/// every crop it is given.
class FakeRec implements RecEngine {
  FakeRec(this.text, {this.quad = plateQuad, this.logp});
  final String text;
  final List<double> quad;

  /// the posterior to hand out instead of [peaky]`(text)`
  final Float64List? logp;
  int detects = 0;

  @override
  List<TextBox> detect(RgbImage rgb) {
    detects++;
    return [TextBox(quad, 0.95)];
  }

  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) => [for (final _ in crops) RecOut(Float32List(0), (logp ?? peaky(text)).length ~/ kVocab, 6625)]; // t must match toLogp's frames

  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => logp ?? peaky(text);

  @override
  String greedyText(RecOut r) => text;
}

const plateQuad = [100.0, 90.0, 220.0, 90.0, 220.0, 150.0, 100.0, 150.0];

/// A photo of a MONITOR as the recogniser sees it: the plate line plus two small interface lines (a web address,
/// browser words) at the top-left. Crops are told apart by their size; anything not listed reads as the plate.
class ScreenRec implements RecEngine {
  ScreenRec({this.plateText = 'MH12AB1234'});
  final String plateText;

  /// (crop height, crop width) -> what the recogniser would say about it
  static const url = (12, 80), words = (14, 90);
  final List<(int, int)> _shapes = [];

  String _textOf((int, int) shape) => shape == url ? 'IMAGESBING' : (shape == words ? 'SEARCHSETTINGS' : plateText);

  String _rawOf((int, int) shape) =>
      shape == url ? 'https://www.bing.com/images/search' : (shape == words ? 'Search Settings Download Related' : plateText);

  @override
  List<TextBox> detect(RgbImage rgb) => [
        TextBox(plateQuad, 0.95),
        TextBox(const [10.0, 10.0, 90.0, 10.0, 90.0, 22.0, 10.0, 22.0], 0.9), // 80 x 12
        TextBox(const [10.0, 30.0, 100.0, 30.0, 100.0, 44.0, 10.0, 44.0], 0.9), // 90 x 14
      ];

  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) {
    final out = <RecOut>[];
    for (final c in crops) {
      _shapes.add((c.height, c.width));
      out.add(RecOut(Float32List(0), peaky(_textOf((c.height, c.width))).length ~/ kVocab, _shapes.length - 1));
    }
    return out;
  }

  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => peaky(_textOf(_shapes[r.classes]));

  @override
  String greedyText(RecOut r) => _rawOf(_shapes[r.classes]);
}

/// A 320x240 JPEG: grey surround, a plate-shaped patch at (100,90)-(220,150).
Uint8List frame({int surround = 70, int plate = 80, int w = 320, int h = 240}) {
  final im = img.Image(width: w, height: h);
  img.fill(im, color: img.ColorRgb8(surround, surround, surround));
  img.fillRect(im, x1: 100, y1: 90, x2: 219, y2: 149, color: img.ColorRgb8(plate, plate, plate));
  return Uint8List.fromList(img.encodeJpg(im, quality: 95));
}

class FakePad implements PadScorer {
  FakePad(this.value);
  final double value;
  int calls = 0;
  @override
  (double?, String?) score(RgbImage frame, (int, int, int, int)? roi) {
    calls++;
    return (value, null);
  }

  @override
  void dispose() {}
}

Map<String, dynamic> auth(Map<String, dynamic> out) => (out['authenticity'] as Map).cast<String, dynamic>();

void main() {
  group('OfflineEngine.create', () {
    test('models that cannot be loaded give a not-ready engine with the real reason, and never throw', () {
      // flutter test has no native ONNX Runtime, so loading ANY model bytes fails here - exactly the
      // failure the app must survive (missing / corrupt models, no runtime)
      final e = OfflineEngine.create(EngineAssets(det: Uint8List.fromList([1, 2, 3]), rec: Uint8List.fromList([4, 5]), charset: 'a\nb'));
      expect(e.status.ready, isFalse);
      expect(e.status.error, contains('could not be loaded'));
      expect(() => e.scan(const [], hsrp: false, requireLiveness: false), throwsA(isA<StateError>()));
      e.dispose(); // no-op, must not throw
    });
  });

  group('the whole pipeline above ONNX Runtime (real scene reader, decoder, geometry, authenticity)', () {
    test('a plate photo becomes an accepted reading with its region, parsed by the app model', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MH12AB1234'));
      final out = eng.scan([EnginePhoto('plate', 'plate.jpg', frame())], hsrp: false, requireLiveness: false);
      final r = ScanResult.fromJson(out);
      expect(r.plate, isNotNull);
      expect(r.plate!.status, 'accept');
      expect(r.plate!.value, 'MH12AB1234');
      expect(r.plate!.display, 'MH 12 AB 1234');
      expect(r.plate!.stateName, 'Maharashtra');
      expect(r.plate!.image, 0);
      expect(r.plate!.box, isNotNull);
      expect(r.plate!.box![0], closeTo(100, 6));
      expect(r.plate!.box![3], closeTo(150, 6));
      expect(r.vin, isNull);
      expect(r.photos.single.width, 320);
      expect(r.auth.liveness, 'not_established'); // one photo can never prove a real car
      expect(r.auth.notRun, containsAll(['retroreflection', 'planarity', 'forensics']));
    });

    test('an unreadable letter slot comes out as "?" plus ranked letter candidates, and the whole thing reaches the app model', () {
      // the real failure: HR88B8888 with the B lost to glare. The slot after the RTO digits must be a letter.
      final lp = peakyWithSlot('HR88B8888', 4, {'?': 0.55, 'B': 0.30, '8': 0.10, 'R': 0.03, 'D': 0.02});
      final eng = OfflineEngine.withRecEngine(FakeRec('HR88B8888', logp: lp));
      final out = eng.scan([EnginePhoto('plate', 'plate.jpg', frame())], hsrp: false, requireLiveness: false);
      final plate = ((out['fields'] as Map)['plate'] as Map).cast<String, dynamic>();
      expect(plate['masked'], 'HR88?8888');
      expect(plate['needs_review'], isTrue);
      final cands = (plate['unknown_candidates'] as List).cast<Map<String, dynamic>>();
      expect(cands.single['position'], 4);
      final opts = (cands.single['options'] as List).cast<Map<String, dynamic>>();
      expect(opts.first['char'], 'B');
      expect(opts.any((o) => RegExp(r'[0-9]').hasMatch(o['char'] as String)), isFalse, reason: 'a letter slot never offers a digit');
      expect(cands.single['suggested'], 'B');
      expect(ScanResult.fromJson(out).plate, isNotNull); // the app model still parses the response
    });

    test('a VIN photo is read for a VIN only', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MAL907KE1E6602791', quad: const [20.0, 100.0, 300.0, 100.0, 300.0, 130.0, 20.0, 130.0]));
      final r = ScanResult.fromJson(eng.scan([EnginePhoto('vin', 'vin.jpg', frame())], hsrp: false, requireLiveness: false));
      expect(r.vin, isNotNull);
      expect(r.vin!.value, 'MAL907KE1E6602791');
      expect(r.plate, isNull);
    });

    test('a sticker label that looks like a VIN is not reported as one', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('TYREPRESSURE32PSI', quad: const [20.0, 100.0, 300.0, 100.0, 300.0, 130.0, 20.0, 130.0]));
      final r = ScanResult.fromJson(eng.scan([EnginePhoto('vin', 'vin.jpg', frame())], hsrp: false, requireLiveness: false));
      expect(r.vin, isNull);
    });

    test('text that is not a plate gives no plate (nothing is invented)', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MITSUBISHIMOTORS'));
      final r = ScanResult.fromJson(eng.scan([EnginePhoto('plate', 'p.jpg', frame())], hsrp: false, requireLiveness: false));
      expect(r.plate, isNull);
    });

    test('a genuine flash bundle (reflective plate) confirms liveness through the REAL flash test', () {
      final pad = FakePad(0.05);
      final eng = OfflineEngine.withRecEngine(FakeRec('MH12AB1234'), pad: pad);
      final out = eng.scan([
        EnginePhoto('plate', 'plate.jpg', frame()),
        EnginePhoto('plate_flash', 'flash.jpg', frame(surround: 80, plate: 200)), // plate far brighter, surround barely
      ], hsrp: true, requireLiveness: false);
      final r = ScanResult.fromJson(out);
      expect(r.plate!.value, 'MH12AB1234');
      expect(r.auth.liveness, 'confirmed', reason: auth(out)['reasons'].toString());
      expect(r.auth.positive, contains('retroreflection_passed'));
      expect(pad.calls, 2); // the screen / print model ran on both frames
      expect((out['screen_context'] as Map)['score'], 0.0); // a plate and nothing else: no interface text
    });

    test('a photo of a MONITOR (web address + browser words around the plate) is a suspected spoof, however clean everything else looks', () {
      final eng = OfflineEngine.withRecEngine(ScreenRec(), pad: FakePad(0.003)); // PadNet scored the real monitor photo 0.003
      final out = eng.scan([EnginePhoto('plate', 'monitor.jpg', frame())], hsrp: false, requireLiveness: false);
      final r = ScanResult.fromJson(out);
      expect(r.plate!.value, 'MH12AB1234', reason: 'the reading is still returned: the plate is real text on the screen');
      final ctx = (out['screen_context'] as Map).cast<String, dynamic>();
      expect(ctx['url'], isTrue, reason: '$ctx');
      expect(ctx['ui_words'], containsAll(['SEARCH', 'SETTINGS', 'DOWNLOAD', 'RELATED']));
      expect(ctx['score'], 2.5); // URL 1.0 + interface words 1.5
      final a = auth(out);
      expect(a['liveness'], 'suspected_spoof', reason: a['reasons'].toString());
      expect(a['status'], 'review');
      expect(r.auth.isSpoofSuspected, isTrue);
      expect(r.auth.isConfirmed, isFalse);
      expect((a['evidence'] as Map)['screen_context'], ctx); // the evidence travels with the decision
      expect((a['reasons'] as List).join(' '), contains('screen / interface context in the scene (+2.5)'));
      expect((a['reasons'] as List).join(' '), contains('a clean score is never used as proof'));
      expect(a['positive'], isEmpty);
    });

    test('...even when the flash test passes on the screen photo (the plate patch brightens): the screen text still wins', () {
      final eng = OfflineEngine.withRecEngine(ScreenRec(), pad: FakePad(0.003));
      final out = eng.scan([
        EnginePhoto('plate', 'monitor.jpg', frame()),
        EnginePhoto('plate_flash', 'monitor_flash.jpg', frame(surround: 80, plate: 200)),
      ], hsrp: true, requireLiveness: false);
      final a = auth(out);
      expect(a['liveness'], 'suspected_spoof', reason: a['reasons'].toString());
      expect(a['liveness'], isNot('confirmed'));
      expect((out['screen_context'] as Map)['score'], greaterThanOrEqualTo(1.0));
    });

    test('a replayed screen (flash changes everything alike) is flagged', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MH12AB1234'), pad: FakePad(0.05));
      final out = eng.scan([
        EnginePhoto('plate', 'plate.jpg', frame()),
        EnginePhoto('plate_flash', 'flash.jpg', frame(surround: 77, plate: 88)),
      ], hsrp: true, requireLiveness: false);
      expect(ScanResult.fromJson(out).auth.liveness, 'suspected_spoof');
    });

    test('the screen / print model failing to exist does not stop the reading', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MH12AB1234'), padUnavailableReason: 'not bundled');
      final out = eng.scan([EnginePhoto('plate', 'plate.jpg', frame())], hsrp: false, requireLiveness: false);
      expect(ScanResult.fromJson(out).plate!.value, 'MH12AB1234');
      expect(auth(out)['not_run'], contains('recapture'));
    });

    test('a corrupt photo is a FormatException the worker can map to a bad-photo error', () {
      final eng = OfflineEngine.withRecEngine(FakeRec('MH12AB1234'));
      expect(() => eng.scan([EnginePhoto('plate', 'x.jpg', Uint8List.fromList([9, 9, 9]))], hsrp: false, requireLiveness: false), throwsFormatException);
    });
  });
}
