import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/data/scanner.dart';
import 'package:vinplate_tester/data/history.dart';
import 'package:vinplate_tester/data/models.dart';
import 'package:vinplate_tester/data/settings.dart';
import 'package:vinplate_tester/ui/capture_plan.dart';
import 'package:vinplate_tester/offline/engine_host.dart';
import 'package:vinplate_tester/ui/explain.dart';
import 'package:vinplate_tester/ui/result_page.dart' show SlotHints;

Map<String, dynamic> fixture() => (jsonDecode(File('test/fixtures/scan_plate_accept.json').readAsStringSync()) as Map).cast<String, dynamic>();

void main() {
  group('models (parsed from a REAL backend response)', () {
    test('accepted plate', () {
      final r = ScanResult.fromJson(fixture());
      final p = r.plate!;
      expect(p.status, 'accept');
      expect(p.isAccepted, isTrue);
      expect(p.value, 'MH12AB1234');
      expect(p.display, 'MH 12 AB 1234');
      expect(p.shown, 'MH 12 AB 1234');
      expect(p.confidence, greaterThan(0.9));
      expect(p.stateName, 'Maharashtra');
      expect(p.box, hasLength(4));
      expect(r.vin, isNull);
      expect(r.photos.single.role, 'plate');
      expect(r.auth.liveness, 'not_established'); // one photo: never "confirmed"
      expect(r.auth.isConfirmed, isFalse);
      expect(r.raw, isNotNull);
    });

    test('a review reading shows the masked text, never the clean value', () {
      final j = fixture();
      final p = j['fields']['plate'] as Map<String, dynamic>;
      p['status'] = 'review';
      p['masked'] = 'MH12?B1234';
      final f = ScanResult.fromJson(j).plate!;
      expect(f.needsReview, isTrue);
      expect(f.shown, 'MH12?B1234');
      expect(f.unknownCount, 1);
    });

    test('unknown_candidates and screen_context are parsed; their absence is fine', () {
      final j = fixture();
      expect(ScanResult.fromJson(j).plate!.unknownSlots, isEmpty);
      expect(ScanResult.fromJson(j).screenContext.score, 0);
      (j['fields']['plate'] as Map<String, dynamic>)['unknown_candidates'] = [
        {'position': 4, 'options': [{'char': 'B', 'p': 0.71}, {'char': 'R', 'p': 0.12}], 'suggested': 'B'},
        {'position': 6, 'options': [{'char': 'X', 'p': 0.5}, {'char': 'Y', 'p': 0.5}], 'suggested': null},
      ];
      j['screen_context'] = {'score': 2.5, 'reasons': ['web address', 'search words']};
      final r = ScanResult.fromJson(j);
      expect(r.plate!.unknownSlots, hasLength(2));
      expect(r.plate!.unknownSlots.first.suggested, 'B');
      expect(r.plate!.unknownSlots.first.options.map((o) => o.char), ['B', 'R']);
      expect(r.plate!.unknownSlots[1].suggested, isNull);
      expect(r.screenContext.score, 2.5);
      expect(r.screenContext.reasons, hasLength(2));
    });

    test('slot hints say "probably" only for a dominant candidate', () {
      const dominant = UnknownSlot(4, [SlotOption('B', 0.71), SlotOption('R', 0.12), SlotOption('D', 0.08)], 'B');
      const open = UnknownSlot(6, [SlotOption('X', 0.5), SlotOption('Y', 0.5)], null);
      expect(SlotHints.describe(dominant), 'Character 5: probably B (71%), or R 12%, D 8%');
      expect(SlotHints.describe(open), 'Character 7: could be X 50%, Y 50%');
    });

    test('missing optional keys never throw; a missing "fields" does', () {
      final r = ScanResult.fromJson({'fields': {'plate': null, 'vin': null}});
      expect(r.foundAnything, isFalse);
      expect(r.auth.liveness, 'not_established');
      expect(() => ScanResult.fromJson({}), throwsFormatException);
    });

    test('authenticity flags', () {
      expect(Authenticity.fromJson({'status': 'reject', 'liveness': 'suspected_spoof'}).isSpoofSuspected, isTrue);
      expect(Authenticity.fromJson({'status': 'no_flags', 'liveness': 'confirmed'}).isConfirmed, isTrue);
      expect(Authenticity.fromJson({'status': 'reject', 'liveness': 'confirmed'}).isConfirmed, isFalse);
    });
  });

  group('OnDeviceScanner (no network anywhere)', () {
    Future<EngineAssets> assets() async => EngineAssets(det: Uint8List(2), rec: Uint8List(2), charset: 'x');
    final photos = [CapturedPhoto(role: 'plate', bytes: Uint8List.fromList([1, 2, 3]), name: 'plate.jpg'), CapturedPhoto(role: 'angle', bytes: Uint8List.fromList([9]), name: 'side.jpg')];

    test('prepare loads the engine once and reports what loaded', () async {
      var made = 0;
      final sc = OnDeviceScanner(runner: InlineRunner((a) {
        made++;
        return _FakeEngine(const EngineStatus(ready: true, recaptureModel: true, detail: 'PP-OCR + decoder'));
      }), assets: assets);
      final a = await sc.prepare();
      final b = await sc.prepare();
      expect(a.ready && a.recaptureModel, isTrue);
      expect(identical(a, b), isTrue);
      expect(made, 1);
    });

    test('scan hands every photo and the flags to the engine, and parses what comes back', () async {
      final eng = _FakeEngine(const EngineStatus(ready: true));
      final sc = OnDeviceScanner(runner: InlineRunner((_) => eng), assets: assets);
      final r = await sc.scan(photos, hsrp: true, requireLiveness: true);
      expect(eng.seen.map((p) => p.role), ['plate', 'angle']);
      expect(eng.seen.map((p) => p.name), ['plate.jpg', 'side.jpg']);
      expect(eng.hsrp && eng.live, isTrue);
      expect(r.plate!.value, 'MH12AB1234');
      expect(r.captureId, isNotEmpty); // assigned on the device
      expect(r.processedAt, isNotEmpty);
    });

    test('an engine that cannot start is an engine error, never an empty result', () async {
      final sc = OnDeviceScanner(runner: InlineRunner((_) => _FakeEngine(const EngineStatus(ready: false, error: 'out of memory'))), assets: assets);
      await expectLater(sc.scan(photos, hsrp: true, requireLiveness: false),
          throwsA(isA<ScanException>().having((e) => e.kind, 'kind', ScanErrorKind.engine).having((e) => e.detail, 'detail', 'out of memory')));
    });

    test('a failing factory (models missing) is reported, not thrown to the UI', () async {
      final sc = OnDeviceScanner(runner: InlineRunner((_) => throw StateError('no models')), assets: assets);
      final st = await sc.prepare();
      expect(st.ready, isFalse);
      expect(st.error, contains('no models'));
    });

    test('an unexpected engine reply is a scan error', () async {
      final sc = OnDeviceScanner(runner: InlineRunner((_) => _FakeEngine(const EngineStatus(ready: true), reply: {'nonsense': true})), assets: assets);
      await expectLater(sc.scan(photos, hsrp: true, requireLiveness: false), throwsA(isA<ScanException>().having((e) => e.kind, 'kind', ScanErrorKind.failed)));
    });

    test('IsolateRunner: models load once in a worker, photos cross the boundary, bad images map to badImage', () async {
      final sc = OnDeviceScanner(runner: IsolateRunner(_isolateFactory), assets: assets);
      final st = await sc.prepare();
      expect(st.ready, isTrue);
      expect(st.detail, 'worker engine (det=2 rec=2 B)');
      final r = await sc.scan(photos, hsrp: false, requireLiveness: false);
      expect(r.plate!.value, 'MH12AB1234');
      expect(r.photos, hasLength(2)); // the worker saw both photos
      await expectLater(sc.scan([CapturedPhoto(role: 'plate', bytes: Uint8List.fromList([0]), name: 'bad.jpg')], hsrp: false, requireLiveness: false),
          throwsA(isA<ScanException>().having((e) => e.kind, 'kind', ScanErrorKind.badImage)));
      sc.dispose();
    });

    test('IsolateRunner: cancel kills the worker, then the next scan starts a fresh one', () async {
      final sc = OnDeviceScanner(runner: IsolateRunner(_isolateFactory), assets: assets);
      await sc.prepare();
      final slow = sc.scan([CapturedPhoto(role: 'plate', bytes: Uint8List.fromList([7]), name: 'slow.jpg')], hsrp: false, requireLiveness: false);
      await Future<void>.delayed(const Duration(milliseconds: 150));
      sc.cancel();
      await expectLater(slow, throwsA(isA<ScanException>().having((e) => e.kind, 'kind', ScanErrorKind.cancelled)));
      final again = await sc.scan(photos, hsrp: false, requireLiveness: false);
      expect(again.plate!.value, 'MH12AB1234');
      sc.dispose();
    });
  });

  group('persistence', () {
    late Directory tmp;
    setUp(() => tmp = Directory.systemTemp.createTempSync('vinplate_test'));
    tearDown(() => tmp.deleteSync(recursive: true));

    test('settings round-trip and survive a corrupt file', () async {
      final f = File('${tmp.path}/s.json');
      final a = AppSettings(file: f);
      await a.update(hsrp: false, requireLiveness: true, onboarded: true);
      final b = AppSettings(file: f);
      await b.load();
      expect(b.hsrp, isFalse);
      expect(b.requireLiveness && b.onboarded, isTrue);
      f.writeAsStringSync('{ not json');
      final c = AppSettings(file: f);
      await c.load();
      expect(c.hsrp, isTrue); // defaults
      expect(c.onboarded, isFalse);
    });

    test('history: add, list newest first, delete, clear, trim', () async {
      final store = HistoryStore(root: Directory('${tmp.path}/h'));
      final r = ScanResult.fromJson(fixture());
      final e1 = await store.add(r, [Uint8List.fromList([1, 2, 3])], confirmedPlate: 'MH12AB1234');
      await Future<void>.delayed(const Duration(milliseconds: 5));
      await store.add(r, [Uint8List.fromList([9])]);
      var l = await store.list();
      expect(l, hasLength(2));
      expect(l.first.savedAt.isAfter(l.last.savedAt), isTrue);
      final first = l.firstWhere((e) => e.id == e1.id);
      expect(first.confirmedPlate, 'MH12AB1234');
      expect(first.title, 'MH12AB1234');
      expect(first.photoFile(0)!.readAsBytesSync(), [1, 2, 3]);
      expect(first.result.plate!.value, 'MH12AB1234');
      await store.delete(first);
      expect(await store.list(), hasLength(1));
      await store.clear();
      expect(await store.list(), isEmpty);
    });
  });

  group('capture plan', () {
    Map<String, Uint8List> shots(Iterable<String> keys) => {for (final i in keys) i: Uint8List.fromList([1])};

    test('a plate scan is ALWAYS front + left + right; the full check adds an optional VIN; a VIN scan is one photo', () {
      expect(CapturePlan.forMode(ScanMode.plate).steps.map((s) => s.id), ['front', 'left', 'right']);
      expect(CapturePlan.forMode(ScanMode.full).steps.map((s) => s.id), ['front', 'left', 'right', 'vin']);
      expect(CapturePlan.forMode(ScanMode.vin).steps.map((s) => s.id), ['vin']);
    });

    test('front, left and right are all required in any plate scan; only the VIN of a full check is optional', () {
      for (final m in [ScanMode.plate, ScanMode.full]) {
        final p = CapturePlan.forMode(m);
        expect(p.steps.where((s) => !s.optional).map((s) => s.id), ['front', 'left', 'right'], reason: '$m');
        expect(p.canSubmit(shots([])), isFalse);
        expect(p.canSubmit(shots(['front'])), isFalse);
        expect(p.canSubmit(shots(['front', 'left'])), isFalse);
        expect(p.canSubmit(shots(['front', 'right'])), isFalse);
        expect(p.canSubmit(shots(['left', 'right'])), isFalse); // no front
        expect(p.canSubmit(shots(['front', 'left', 'right'])), isTrue);
      }
      final full = CapturePlan.forMode(ScanMode.full);
      expect(full.steps.firstWhere((s) => s.id == 'vin').skipWarning, isNotEmpty);
      expect(CapturePlan.forMode(ScanMode.vin).canSubmit(shots([])), isFalse);
      expect(CapturePlan.forMode(ScanMode.vin).canSubmit(shots(['vin'])), isTrue);
    });

    test('the flash frame is part of the FRONT step, taken automatically (the person never repeats the pose)', () {
      final p = CapturePlan.forMode(ScanMode.plate);
      expect(p.steps.where((s) => s.withFlash).map((s) => s.id), ['front']);
      expect(p.steps.first.flashKey, 'front_flash');
      expect(p.checksLiveness, isTrue);
      expect(CapturePlan.forMode(ScanMode.vin).checksLiveness, isFalse);
    });

    test('photos carry the roles the reader expects: plate, plate_flash, angle, angle (and vin)', () {
      final p = CapturePlan.forMode(ScanMode.full);
      final ph = p.photos(shots(['vin', 'right', 'front_flash', 'front', 'left']));
      expect(ph.map((x) => x.role), ['plate', 'plate_flash', 'angle', 'angle', 'vin']);
      expect(ph.map((x) => x.name), ['front.jpg', 'front_flash.jpg', 'left.jpg', 'right.jpg', 'vin.jpg']);
      // no flash frame (torch unavailable): the rest still goes, nothing pretends a flash exists
      final noFlash = p.photos(shots(['front', 'left', 'right']));
      expect(noFlash.map((x) => x.role), ['plate', 'angle', 'angle']);
      expect(p.hasFlash(shots(['front', 'left', 'right'])), isFalse);
      expect(p.hasFlash(shots(['front', 'front_flash'])), isTrue);
    });

    test('missing steps are reported for the review sheet', () {
      final p = CapturePlan.forMode(ScanMode.full);
      expect(p.missing(shots(['front', 'left', 'right'])).map((s) => s.id), ['vin']);
    });

    test('there is no way to add a photo from outside the camera: the plan has no gallery notion at all', () {
      // photos can only come from the capture steps' own shots; nothing in CapturePlan accepts external files
      final p = CapturePlan.forMode(ScanMode.plate);
      expect(p.photos(shots(['gallery', 'upload', 'file'])), isEmpty);
    });
  });

  group('wording and validation', () {
    test('technical reasons become plain language; unknown ones pass through', () {
      expect(humanReason('ambiguous look-alike character(s) at position(s) [4]: shown as unknown'), contains('look like others'));
      expect(humanReason('series letter I/O/Q: not issued on HSRP plates, so it is probably a misread look-alike'), contains('not used on modern plates'));
      expect(humanReason('something new the server says'), 'something new the server says');
      expect(humanReasons(["grammar/priors changed the network's raw reading"]), isEmpty);
    });

    test('plate validation', () {
      expect(validateIdentifier('MH12AB1234', plate: true), isNull);
      expect(validateIdentifier('mh12ab1234', plate: true), isNull);
      expect(validateIdentifier('MH12', plate: true), isNotNull);
      expect(validateIdentifier('MH12?B1234', plate: true), isNotNull);
      expect(validateIdentifier('', plate: true), isNotNull);
    });

    test('VIN validation', () {
      expect(validateIdentifier('MAL907KE1E6602791', plate: false), isNull);
      expect(validateIdentifier('MAL907KE1E660279I', plate: false), contains('never contains I, O or Q'));
      expect(validateIdentifier('81GZ16D60VVY9', plate: false), isNull); // older, shorter chassis number
      expect(validateIdentifier('ABC', plate: false), isNotNull);
    });

    test('what to do next follows from what was not checked', () {
      expect(improvements(['retroreflection', 'planarity']), hasLength(2));
      expect(improvements([]), isEmpty);
    });
  });
}

// ----------------------------------------------------------------------------- test doubles
class _FakeEngine implements ScanEngine {
  _FakeEngine(this.status, {this.reply});
  @override
  final EngineStatus status;
  final Map<String, dynamic>? reply;
  List<EnginePhoto> seen = [];
  bool hsrp = false, live = false;

  @override
  Map<String, dynamic> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) {
    seen = photos;
    this.hsrp = hsrp;
    live = requireLiveness;
    return reply ?? fixture();
  }

  @override
  void dispose() {}
}

/// Top level (an isolate entry point may not be a closure). Behaves like the real engine in the ways
/// the runner cares about: it sees the model bytes, answers per photo, fails on a bad image, and a
/// photo whose first byte is 7 takes a long time (so cancel can be tested).
ScanEngine _isolateFactory(EngineAssets a) => _WorkerEngine(a.det.length, a.rec.length);

class _WorkerEngine implements ScanEngine {
  _WorkerEngine(this.det, this.rec);
  final int det, rec;

  @override
  EngineStatus get status => EngineStatus(ready: true, recaptureModel: true, detail: 'worker engine (det=$det rec=$rec B)');

  @override
  Map<String, dynamic> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) {
    if (photos.first.bytes.first == 0) throw const FormatException('not an image');
    if (photos.first.bytes.first == 7) {
      final end = DateTime.now().add(const Duration(seconds: 30));
      while (DateTime.now().isBefore(end)) {} // a long native inference cannot be interrupted: only killing the isolate stops it
    }
    final j = fixture();
    j['images'] = [for (var i = 0; i < photos.length; i++) {'index': i, 'name': photos[i].name, 'role': photos[i].role, 'size': [10, 10]}];
    return j;
  }

  @override
  void dispose() {}
}
