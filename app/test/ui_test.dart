import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/app_services.dart';
import 'package:vinplate_tester/data/scanner.dart';
import 'package:vinplate_tester/data/history.dart';
import 'package:vinplate_tester/data/models.dart';
import 'package:vinplate_tester/data/settings.dart';
import 'package:vinplate_tester/ui/capture_page.dart';
import 'package:vinplate_tester/ui/capture_plan.dart';
import 'package:vinplate_tester/ui/home_page.dart';
import 'package:vinplate_tester/ui/plate_types.dart';
import 'package:vinplate_tester/ui/result_page.dart';
import 'package:vinplate_tester/ui/theme.dart';
import 'package:vinplate_tester/ui/widgets.dart';

Map<String, dynamic> fixture() => (jsonDecode(File('test/fixtures/scan_plate_accept.json').readAsStringSync()) as Map).cast<String, dynamic>();
Uint8List photoBytes() => File('test/fixtures/plate.jpg').readAsBytesSync();

ScanResult acceptResult() => ScanResult.fromJson(fixture());

ScanResult reviewResult() {
  final j = fixture();
  final p = j['fields']['plate'] as Map<String, dynamic>;
  p['status'] = 'review';
  p['masked'] = 'MH12?B1234';
  p['confidence'] = 0.62;
  p['inferred'] = [4];
  p['reasons'] = ['ambiguous look-alike character(s) at position(s) [4]: shown as unknown'];
  p['alternatives'] = [
    {'text': 'MH12AB1234', 'p': 0.5},
    {'text': 'MH12AD1234', 'p': 0.3},
  ];
  return ScanResult.fromJson(j);
}

/// The HR88B8888 case from a real test: the 5th character is unreadable, the plate format says it must
/// be a letter, and the recogniser's own evidence ranks the LEGAL letters.
ScanResult unknownSlotResult({bool screen = false}) {
  final j = fixture();
  final p = j['fields']['plate'] as Map<String, dynamic>;
  p['status'] = 'review';
  p['value'] = 'HR88?8888';
  p['masked'] = 'HR88?8888';
  p['display'] = null;
  p['confidence'] = 0.65;
  p['inferred'] = [4];
  p['reasons'] = [];
  p['alternatives'] = [];
  p['parsed'] = {'layout': 'standard', 'state_name': 'Haryana'};
  p['unknown_candidates'] = [
    {
      'position': 4,
      'options': [
        {'char': 'B', 'p': 0.71},
        {'char': 'R', 'p': 0.12},
        {'char': 'D', 'p': 0.08},
      ],
      'suggested': 'B',
    }
  ];
  if (screen) {
    j['screen_context'] = {
      'score': 3.0,
      'reasons': ['The photo is full of web-page text (a web address, search and menu words, 30 lines of text): it looks like a picture of a screen.']
    };
    (j['authenticity'] as Map<String, dynamic>)['liveness'] = 'suspected_spoof';
    (j['authenticity'] as Map<String, dynamic>)['status'] = 'reject';
  }
  return ScanResult.fromJson(j);
}

class FakeScanner implements Scanner {
  FakeScanner({this.status = const EngineStatus(ready: true, recaptureModel: true, detail: 'PP-OCR v3 + grammar decoder, CPU', loadMs: 800)});
  EngineStatus status;
  ScanException? scanError;
  int prepares = 0;

  @override
  Future<EngineStatus> prepare() async {
    prepares++;
    return status;
  }

  @override
  Future<ScanResult> scan(List<CapturedPhoto> photos, {required bool hsrp, required bool requireLiveness}) async {
    if (scanError != null) throw scanError!;
    return acceptResult();
  }

  @override
  void cancel() {}
  @override
  void dispose() {}
}

class MemHistory extends HistoryStore {
  final items = <HistoryEntry>[];
  @override
  Future<HistoryEntry> add(ScanResult r, List<Uint8List> photos, {String? confirmedPlate, String? confirmedVin, String? confirmedCategory}) async {
    final e = HistoryEntry(id: 'e${items.length}', savedAt: DateTime.now(), result: r, confirmedPlate: confirmedPlate, confirmedVin: confirmedVin, confirmedCategory: confirmedCategory);
    items.insert(0, e);
    return e;
  }

  @override
  Future<List<HistoryEntry>> list() async => List.of(items);
  @override
  Future<void> delete(HistoryEntry e) async => items.remove(e);
  @override
  Future<void> clear() async => items.clear();
}

({AppServices services, FakeScanner scanner, MemHistory history, AppSettings settings}) rig({FakeScanner? scanner, bool onboarded = true}) {
  final sc = scanner ?? FakeScanner();
  final h = MemHistory();
  final st = AppSettings(persist: false)..onboarded = onboarded;
  return (services: AppServices(settings: st, history: h, scanner: sc), scanner: sc, history: h, settings: st);
}

Widget app(AppServices s, Widget home) => AppScope(services: s, child: MaterialApp(theme: buildTheme(Brightness.light), home: home));

List<String> maskedTexts(WidgetTester t) => [for (final w in t.widgetList<MaskedText>(find.byType(MaskedText))) w.text];

Future<void> setPhone(WidgetTester t) async {
  t.view.physicalSize = const Size(1080, 2340);
  t.view.devicePixelRatio = 2.75;
  addTearDown(t.view.reset);
}

Future<void> dispose(WidgetTester t) => t.pumpWidget(const SizedBox());

/// Opens ResultPage on top of a stub root, as the capture flow does, so Done can pop back to it.
Future<void> openResult(WidgetTester t, AppServices s, ScanResult r, {Map<int, Uint8List>? photos}) async {
  await t.pumpWidget(app(s, Builder(builder: (ctx) => Scaffold(body: Center(child: TextButton(onPressed: () => Navigator.of(ctx).push(MaterialPageRoute<void>(builder: (_) => ResultPage(result: r, photos: photos ?? {0: photoBytes()}))), child: const Text('open')))))));
  await t.tap(find.text('open'));
  await t.pumpAndSettle();
}

void main() {
  group('home', () {
    testWidgets('shows the offline reader as ready and the three scan actions', (t) async {
      await setPhone(t);
      final r = rig();
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      expect(find.text('Offline reader ready'), findsOneWidget);
      expect(find.textContaining('Runs entirely on this phone'), findsOneWidget);
      expect(find.textContaining('screen / print detector on'), findsOneWidget);
      expect(find.byKey(const ValueKey('scan-full')), findsOneWidget);
      expect(find.byKey(const ValueKey('scan-plate')), findsOneWidget);
      expect(find.byKey(const ValueKey('scan-vin')), findsOneWidget);
      expect(find.text('Front, left, right'), findsOneWidget); // a plate scan is three photos, never one
      expect(find.textContaining('appear here'), findsOneWidget); // empty history
      expect(find.textContaining('server', findRichText: true), findsNothing); // no server concept anywhere
      await dispose(t);
    });

    testWidgets('an engine that cannot start is explained and can be retried', (t) async {
      await setPhone(t);
      final r = rig(scanner: FakeScanner(status: const EngineStatus(ready: false, error: 'The reading engine could not start: out of memory')));
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      expect(find.text('The reading engine could not start'), findsOneWidget);
      expect(find.textContaining('out of memory'), findsOneWidget);
      r.scanner.status = const EngineStatus(ready: true, recaptureModel: false);
      await t.tap(find.byKey(const ValueKey('retry-engine')));
      await t.pumpAndSettle();
      expect(find.text('Offline reader ready'), findsOneWidget);
      expect(find.textContaining('screen / print detector off'), findsOneWidget);
      expect(r.scanner.prepares, greaterThanOrEqualTo(2));
      await dispose(t);
    });

    testWidgets('settings: the engine can be checked, and there is no server address field', (t) async {
      await setPhone(t);
      final r = rig();
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      await t.tap(find.byKey(const ValueKey('open-settings')));
      await t.pumpAndSettle();
      expect(find.text('Reading engine'), findsOneWidget);
      expect(find.byKey(const ValueKey('server-url')), findsNothing);
      expect(find.textContaining('never uses the network'), findsOneWidget);
      expect(find.text('Lab tools'), findsNothing); // the research harness (with a gallery upload) is not part of the product
      await t.tap(find.byKey(const ValueKey('self-test')));
      await t.pumpAndSettle();
      expect(find.byKey(const ValueKey('test-result')), findsOneWidget);
      expect(find.textContaining('Ready. PP-OCR v3 + grammar decoder, CPU'), findsOneWidget);
      expect(find.textContaining('Loaded in 0.8 s'), findsOneWidget);
      await t.ensureVisible(find.byKey(const ValueKey('save-settings')));
      await t.tap(find.byKey(const ValueKey('save-settings')));
      await t.pumpAndSettle();
      await dispose(t);
    });

    testWidgets('first run shows the walkthrough once, then remembers it', (t) async {
      await setPhone(t);
      final r = rig(onboarded: false);
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      expect(find.text('Read plates and VINs'), findsOneWidget);
      for (var i = 0; i < 3; i++) {
        await t.tap(find.byKey(const ValueKey('intro-next')));
        await t.pumpAndSettle();
      }
      expect(find.text('Your photos stay with you'), findsOneWidget);
      expect(find.textContaining('no photo ever leaves the device'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('intro-next'))); // "Get started"
      await t.pumpAndSettle();
      expect(find.text('Offline reader ready'), findsOneWidget);
      expect(r.settings.onboarded, isTrue);
      await dispose(t);
    });

    testWidgets('starting a scan while the engine failed explains why and offers a retry', (t) async {
      await setPhone(t);
      final r = rig(scanner: FakeScanner(status: const EngineStatus(ready: false, error: 'models missing')));
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      await t.tap(find.byKey(const ValueKey('scan-plate')));
      await t.pumpAndSettle();
      expect(find.text('The reading engine is not ready'), findsOneWidget);
      expect(find.text('models missing'), findsWidgets);
      r.scanner.status = const EngineStatus(ready: true);
      await t.tap(find.text('Try again'));
      await t.pumpAndSettle();
      expect(find.text('Offline reader ready'), findsOneWidget);
      await dispose(t);
    });

    testWidgets('recent scans list, reopen and delete with confirmation', (t) async {
      await setPhone(t);
      final r = rig();
      await r.history.add(acceptResult(), [photoBytes()], confirmedPlate: 'MH12AB1234');
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      expect(find.text('1 saved'), findsOneWidget);
      expect(maskedTexts(t), contains('MH12AB1234'));
      await t.drag(find.byType(Dismissible), const Offset(-600, 0));
      await t.pumpAndSettle();
      expect(find.text('Delete this scan?'), findsOneWidget);
      await t.tap(find.text('Delete'));
      await t.pumpAndSettle();
      expect(r.history.items, isEmpty);
      await dispose(t);
    });
  });

  group('result', () {
    testWidgets('an accepted plate is shown plainly and Done saves it to history', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, acceptResult());
      expect(find.text('Read successfully'), findsOneWidget);
      expect(maskedTexts(t), contains('MH 12 AB 1234'));
      expect(find.byKey(const ValueKey('unknown-char')), findsNothing);
      expect(find.text('Liveness not established'), findsOneWidget); // one photo: not proof of a real car
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(r.history.items, hasLength(1));
      expect(r.history.items.single.confirmedPlate, 'MH12AB1234');
      expect(find.text('open'), findsOneWidget); // back at the root
      await dispose(t);
    });

    testWidgets('a reading with an unknown character must be confirmed by the person', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, reviewResult());
      expect(find.textContaining('1 reading needs your check'), findsOneWidget);
      expect(find.byKey(const ValueKey('unknown-char')), findsOneWidget); // the red ?, not a guess
      expect(find.text('Done (1 to review)'), findsOneWidget);

      await t.tap(find.byKey(const ValueKey('field-plate')));
      await t.pumpAndSettle();
      expect(find.text('Why this needs review'), findsOneWidget);
      expect(find.textContaining('look like others'), findsOneWidget); // plain language, not the raw reason
      expect(find.textContaining('MH12AD1234'), findsOneWidget); // an alternative to choose
      // the field is NOT pre-filled with the decoder's guess
      expect(t.widget<TextField>(find.byKey(const ValueKey('confirm-field'))).controller!.text, isEmpty);
      // invalid entry is refused with a reason
      await t.enterText(find.byKey(const ValueKey('confirm-field')), 'MH12');
      await t.ensureVisible(find.byKey(const ValueKey('confirm-button')));
      await t.tap(find.byKey(const ValueKey('confirm-button')));
      await t.pumpAndSettle();
      expect(find.text('A plate has 6 to 11 characters'), findsOneWidget);
      // choose an alternative, confirm
      await t.tap(find.textContaining('MH12AD1234'));
      await t.pumpAndSettle();
      await t.ensureVisible(find.byKey(const ValueKey('confirm-button')));
      await t.tap(find.byKey(const ValueKey('confirm-button')));
      await t.pumpAndSettle();
      expect(find.text('Confirmed by you'), findsOneWidget);
      expect(maskedTexts(t), contains('MH12AD1234'));
      expect(find.byKey(const ValueKey('unknown-char')), findsNothing);
      expect(find.text('Done'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(r.history.items.single.confirmedPlate, 'MH12AD1234');
      await dispose(t);
    });

    testWidgets('finishing with an unconfirmed reading asks first, and saves it as not confirmed', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, reviewResult());
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(find.text('Finish with 1 unconfirmed?'), findsOneWidget);
      await t.tap(find.text('Keep reviewing'));
      await t.pumpAndSettle();
      expect(r.history.items, isEmpty);
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      await t.tap(find.text('Finish anyway'));
      await t.pumpAndSettle();
      expect(r.history.items.single.confirmedPlate, isNull);
      await dispose(t);
    });

    testWidgets('history saving can be turned off', (t) async {
      await setPhone(t);
      final r = rig();
      r.settings.saveHistory = false;
      await openResult(t, r.services, acceptResult());
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(r.history.items, isEmpty);
      await dispose(t);
    });

    testWidgets('the authenticity card opens a plain explanation, including what was NOT checked', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, acceptResult());
      await t.tap(find.byKey(const ValueKey('auth-card')));
      await t.pumpAndSettle();
      expect(find.text('Is this a real car?'), findsWidgets);
      expect(find.text('Not checked'), findsOneWidget);
      expect(find.text('Flash test (reflective plate)'), findsOneWidget);
      expect(find.text('Side-step test (depth)'), findsOneWidget);
      await t.scrollUntilVisible(find.textContaining('not on real devices'), 300, scrollable: find.byType(Scrollable).last);
      expect(find.textContaining('not on real devices'), findsOneWidget);
      await t.scrollUntilVisible(find.text('Got it'), 300, scrollable: find.byType(Scrollable).last);
      await t.tap(find.text('Got it'));
      await t.pumpAndSettle();
      await dispose(t);
    });

    testWidgets('nothing found: says so and how to retake', (t) async {
      await setPhone(t);
      final r = rig();
      final none = ScanResult.fromJson({
        'fields': {'plate': null, 'vin': null},
        'images': [{'index': 0, 'name': 'p.jpg', 'role': 'plate', 'size': [100, 100]}]
      });
      await openResult(t, r.services, none, photos: const {});
      expect(find.text('Nothing could be read'), findsOneWidget);
      expect(find.text('Number plate not found'), findsOneWidget);
      await dispose(t);
    });

    testWidgets('an unknown character shows the ranked LEGAL candidates, never auto-fills, and one tap adopts one', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, unknownSlotResult());
      // the dominant legal letter sits IN the slot as an amber, unconfirmed suggestion - not a bare red '?'
      expect(find.byKey(const ValueKey('suggested-char')), findsOneWidget);
      expect(find.byKey(const ValueKey('unknown-char')), findsNothing);
      expect(find.text('Done (1 to review)'), findsOneWidget); // still needs the person's confirmation
      // the hint is on the card itself: the '?' is not a dead end
      expect(find.byKey(const ValueKey('slot-hint-4')), findsOneWidget);
      expect(find.textContaining('Character 5: probably B (71%), or R 12%, D 8%'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('field-plate')));
      await t.pumpAndSettle();
      expect(find.text('Character 5 could be'), findsOneWidget);
      expect(find.textContaining('Only characters a plate can have in this position'), findsOneWidget);
      // never pre-filled with the decoder's guess
      expect(t.widget<TextField>(find.byKey(const ValueKey('confirm-field'))).controller!.text, isEmpty);
      expect(find.byKey(const ValueKey('slot-4-B')), findsOneWidget);
      expect(find.byKey(const ValueKey('slot-4-R')), findsOneWidget);
      await t.ensureVisible(find.byKey(const ValueKey('slot-4-B')));
      await t.tap(find.byKey(const ValueKey('slot-4-B')));
      await t.pumpAndSettle();
      expect(t.widget<TextField>(find.byKey(const ValueKey('confirm-field'))).controller!.text, 'HR88B8888');
      await t.ensureVisible(find.byKey(const ValueKey('confirm-button')));
      await t.tap(find.byKey(const ValueKey('confirm-button')));
      await t.pumpAndSettle();
      expect(maskedTexts(t), contains('HR88B8888'));
      expect(find.byKey(const ValueKey('slot-hint-4')), findsNothing); // confirmed: the hint goes away
      await dispose(t);
    });

    testWidgets('a screen photo says why: the authenticity sheet lists what the scene shows', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, unknownSlotResult(screen: true));
      expect(find.text('Do not trust this capture'), findsOneWidget);
      await t.tap(find.byKey(const ValueKey('auth-card')));
      await t.pumpAndSettle();
      expect(find.text('What the photo shows'), findsOneWidget);
      expect(find.textContaining('web-page text'), findsOneWidget);
      expect(find.text('Suspected screen / print'), findsWidgets);
      await dispose(t);
    });

    testWidgets('a suspected screen / print is called out loudly', (t) async {
      await setPhone(t);
      final r = rig();
      final j = fixture();
      (j['authenticity'] as Map<String, dynamic>)['liveness'] = 'suspected_spoof';
      (j['authenticity'] as Map<String, dynamic>)['status'] = 'reject';
      await openResult(t, r.services, ScanResult.fromJson(j));
      expect(find.text('Do not trust this capture'), findsOneWidget);
      await dispose(t);
    });
  });

  group('plate type (vehicle category)', () {
    Map<String, dynamic> withCategory(String key, {double conf = 0.9, bool bh = false, List<Map<String, dynamic>> alts = const []}) => {
          'key': key,
          'label': key,
          'confidence': conf,
          'bg': {'name': 'x', 'rgb': [255, 255, 255]},
          'ink': 'black',
          'reasons': <String>[],
          'alternatives': alts,
          'ev': key.startsWith('ev_'),
          'bh': bh,
        };

    ScanResult plateWith(Map<String, dynamic>? category, {bool review = false}) {
      final j = fixture();
      final p = j['fields']['plate'] as Map<String, dynamic>;
      if (category != null) p['category'] = category;
      if (review) {
        p['status'] = 'review';
        p['masked'] = 'MH12?B1234';
      }
      return ScanResult.fromJson(j);
    }

    // The picker lists every type in one scrolling sheet; a tall test screen shows them all without dragging.
    Future<void> setTallPhone(WidgetTester t) async {
      t.view.physicalSize = const Size(1080, 9000);
      t.view.devicePixelRatio = 2.75;
      addTearDown(t.view.reset);
    }

    Color? boardColour(WidgetTester t) => (t.widget<Container>(find.byKey(const ValueKey('plate-board'))).decoration as BoxDecoration).color;

    test('models: category parses, tolerates alternative shapes, and is absent for old scans', () {
      final c = PlateCategory.fromJson(withCategory('ev_commercial', conf: 0.7, bh: true, alts: [
        {'key': 'commercial', 'p': 0.2},
        {'key': 'ev_private', 'confidence': 0.1},
      ]));
      expect(c.key, 'ev_commercial');
      expect(c.ev, isTrue);
      expect(c.bh, isTrue);
      expect(c.alternatives.map((a) => a.key), ['commercial', 'ev_private']);
      expect(c.alternatives[1].p, 0.1);
      expect(PlateCategory.fromJson(const {}).key, 'unknown');
      expect(plateWith(null).plate!.category, isNull);
      expect(plateWith(withCategory('private')).plate!.category!.key, 'private');
    });

    testWidgets('a private plate is drawn on a white board with a Private chip', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('private')));
      expect(find.byKey(const ValueKey('plate-type-chip')), findsOneWidget);
      expect(find.text('Private vehicle'), findsWidgets);
      expect(boardColour(t), const Color(0xFFFFFFFF));
      await dispose(t);
    });

    testWidgets('a commercial plate is yellow, an EV plate green, a rental plate black', (t) async {
      await setPhone(t);
      final r = rig();
      final expected = {'commercial': const Color(0xFFFFD21F), 'ev_private': const Color(0xFF1E8E3E), 'rental': const Color(0xFF151515)};
      for (final e in expected.entries) {
        await openResult(t, r.services, plateWith(withCategory(e.key)));
        expect(boardColour(t), e.value, reason: e.key);
        await dispose(t);
      }
    });

    testWidgets('an unsure type says "Probably" and asks the person to check; unknown has no board', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('commercial', conf: 0.4)));
      expect(find.textContaining('Probably commercial'), findsOneWidget);
      expect(find.textContaining('Not sure of the type'), findsOneWidget);
      await dispose(t);
      await openResult(t, r.services, plateWith(withCategory('unknown', conf: 0)));
      expect(find.text('Type not determined'), findsOneWidget);
      expect(find.byKey(const ValueKey('plate-board')), findsNothing);
      expect(find.textContaining('Not sure of the type'), findsOneWidget);
      await dispose(t);
    });

    testWidgets('a type that is only a guess is never painted as the plate colour (a real green EV was once shown on a blue board)', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('diplomatic', conf: 0.42)));
      expect(find.textContaining('Probably diplomatic'), findsOneWidget);
      expect(find.byKey(const ValueKey('plate-board')), findsNothing);
      await dispose(t);
      // a confident type is drawn
      await openResult(t, r.services, plateWith(withCategory('ev_private', conf: 0.9)));
      expect(boardColour(t), const Color(0xFF1E8E3E));
      await dispose(t);
    });

    ScanResult screenPlate(Map<String, dynamic> category) {
      final j = fixture();
      (j['fields']['plate'] as Map<String, dynamic>)['category'] = category;
      (j['authenticity'] as Map<String, dynamic>)['liveness'] = 'suspected_spoof';
      (j['authenticity'] as Map<String, dynamic>)['status'] = 'reject';
      return ScanResult.fromJson(j);
    }

    testWidgets('colours seen in a photo of a screen are only a guess: hedged, no coloured board, and the person can still choose', (t) async {
      await setTallPhone(t);
      final r = rig();
      await openResult(t, r.services, screenPlate(withCategory('ev_private', conf: 0.95)));
      expect(find.text('Do not trust this capture'), findsOneWidget);
      expect(find.textContaining('Probably electric vehicle (private)'), findsOneWidget); // even at 95 % confidence
      expect(find.byKey(const ValueKey('plate-board')), findsNothing);
      expect(find.byKey(const ValueKey('type-unsure-note')), findsOneWidget);
      expect(find.textContaining('looks like a screen or print'), findsOneWidget);
      // choosing a type is the person's own statement: it is shown as theirs, in the plate colours
      await t.tap(find.byKey(const ValueKey('plate-type-chip')));
      await t.pumpAndSettle();
      await t.tap(find.byKey(const ValueKey('type-commercial')));
      await t.pumpAndSettle();
      expect(find.textContaining('Probably'), findsNothing);
      expect(boardColour(t), const Color(0xFFFFD21F));
      await dispose(t);
    });

    testWidgets('a plate with unknown characters keeps the plain card, so the ? boxes stay legible', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('commercial'), review: true));
      expect(find.byKey(const ValueKey('plate-board')), findsNothing);
      expect(find.byKey(const ValueKey('unknown-char')), findsOneWidget);
      expect(find.text('Commercial / transport vehicle'), findsOneWidget); // the chip still says it
      await dispose(t);
    });

    testWidgets('a scan without a category (older scan, or none worked out) shows no chip', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(null));
      expect(find.byKey(const ValueKey('plate-type-chip')), findsNothing);
      expect(find.byKey(const ValueKey('plate-board')), findsNothing);
      await dispose(t);
    });

    testWidgets('the person can pick another type; it is marked as theirs, copied, and saved', (t) async {
      await setTallPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('private', conf: 0.5, alts: [{'key': 'ev_private', 'p': 0.3}])));
      await t.tap(find.byKey(const ValueKey('plate-type-chip')));
      await t.pumpAndSettle();
      expect(find.text('What type of vehicle is it?'), findsOneWidget);
      expect(find.text('detected'), findsOneWidget);
      expect(find.text('possible'), findsOneWidget); // the evidence also allowed EV
      await t.tap(find.byKey(const ValueKey('type-ev_private')));
      await t.pumpAndSettle();
      expect(find.text('Electric vehicle (private)'), findsWidgets);
      expect(find.textContaining('Probably'), findsNothing);
      expect(find.textContaining('Not sure of the type'), findsNothing);
      expect(find.byIcon(Icons.person), findsOneWidget);
      expect(boardColour(t), const Color(0xFF1E8E3E));
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(r.history.items.single.confirmedCategory, 'ev_private');
      expect(r.history.items.single.categoryKey, 'ev_private');
      await dispose(t);
    });

    testWidgets('picking nothing keeps what the reader said and saves no override', (t) async {
      await setTallPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('commercial')));
      await t.tap(find.byKey(const ValueKey('plate-type-chip')));
      await t.pumpAndSettle();
      await t.tap(find.text('Keep as is'));
      await t.pumpAndSettle();
      await t.tap(find.byKey(const ValueKey('done')));
      await t.pumpAndSettle();
      expect(r.history.items.single.confirmedCategory, isNull);
      expect(r.history.items.single.categoryKey, 'commercial');
      await dispose(t);
    });

    testWidgets('history shows the type badge and name, and a reopened scan cannot change it', (t) async {
      await setPhone(t);
      final r = rig();
      await r.history.add(plateWith(withCategory('commercial')), [photoBytes()], confirmedPlate: 'MH12AB1234');
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      expect(find.textContaining('Commercial / transport vehicle'), findsOneWidget);
      expect(find.byType(PlateBadge), findsOneWidget);
      await t.pumpWidget(app(r.services, ResultPage(result: r.history.items.single.result, photos: const {}, entry: r.history.items.single)));
      await t.pumpAndSettle();
      expect(find.byKey(const ValueKey('plate-type-chip')), findsOneWidget);
      expect(t.widget<InkWell>(find.byKey(const ValueKey('plate-type-chip'))).onTap, isNull);
      await dispose(t);
    });

    testWidgets('a Bharat series plate says so', (t) async {
      await setPhone(t);
      final r = rig();
      await openResult(t, r.services, plateWith(withCategory('private', bh: true)));
      expect(find.text('Bharat (BH) series'), findsOneWidget);
      await dispose(t);
    });

    testWidgets('Settings has a plate types guide listing every type and the caveat', (t) async {
      await setPhone(t);
      final r = rig();
      await t.pumpWidget(app(r.services, const HomePage()));
      await t.pumpAndSettle();
      await t.tap(find.byKey(const ValueKey('open-settings')));
      await t.pumpAndSettle();
      await t.ensureVisible(find.byKey(const ValueKey('plate-types-guide')));
      await t.tap(find.byKey(const ValueKey('plate-types-guide')));
      await t.pumpAndSettle();
      expect(find.text('Plate types'), findsOneWidget);
      for (final key in ['private', 'commercial', 'rental', 'ev_private', 'ev_commercial', 'temporary', 'trade', 'diplomatic', 'military']) {
        await t.scrollUntilVisible(find.byKey(ValueKey('type-$key')), 200, scrollable: find.byType(Scrollable).last);
        expect(find.byKey(ValueKey('type-$key')), findsOneWidget, reason: key);
      }
      await t.scrollUntilVisible(find.textContaining('have not been checked against the official notification'), 200, scrollable: find.byType(Scrollable).last);
      await dispose(t);
    });
  });

  group('capture screen', () {
    Future<void> openCapture(WidgetTester t, AppServices s, Future<List<CameraDescription>> Function() loader) async {
      await t.pumpWidget(app(s, CapturePage(mode: ScanMode.full, cameraLoader: loader)));
      await t.pump();
      await t.pump(const Duration(milliseconds: 700)); // "Starting camera" spins behind the dialog: step time, do not settle
    }

    testWidgets('says why the camera is needed before the system asks, once', (t) async {
      await setPhone(t);
      final r = rig();
      await openCapture(t, r.services, () async => throw CameraException('CameraAccessDenied', 'denied'));
      expect(find.text('Camera access'), findsOneWidget);
      expect(find.textContaining('only to photograph the number plate and VIN'), findsOneWidget);
      expect(find.textContaining('nothing is uploaded'), findsOneWidget);
      await t.tap(find.text('Continue'));
      await t.pumpAndSettle();
      expect(r.settings.cameraRationaleShown, isTrue);
      await dispose(t);
    });

    testWidgets('permission denied: explains how to enable it, and there is no gallery way around it', (t) async {
      await setPhone(t);
      final r = rig();
      r.settings.cameraRationaleShown = true;
      await openCapture(t, r.services, () async => throw CameraException('CameraAccessDenied', 'denied'));
      await t.pumpAndSettle();
      expect(find.text('Camera permission is turned off.'), findsOneWidget);
      expect(find.textContaining('Permissions and allow the Camera'), findsOneWidget);
      expect(find.text('Try again'), findsOneWidget);
      expect(find.textContaining('Gallery'), findsNothing); // live camera only: no upload path to fall back to
      expect(find.textContaining('live camera'), findsWidgets);
      await dispose(t);
    });

    testWidgets('no camera at all is reported plainly', (t) async {
      await setPhone(t);
      final r = rig();
      r.settings.cameraRationaleShown = true;
      await openCapture(t, r.services, () async => <CameraDescription>[]);
      await t.pumpAndSettle();
      expect(find.text('No camera was found on this device.'), findsOneWidget);
      await dispose(t);
    });

    testWidgets('closing with no photos taken leaves without asking', (t) async {
      await setPhone(t);
      final r = rig();
      r.settings.cameraRationaleShown = true;
      await t.pumpWidget(app(r.services, Builder(builder: (ctx) => Scaffold(body: Center(child: TextButton(onPressed: () => Navigator.of(ctx).push(MaterialPageRoute<void>(builder: (_) => CapturePage(mode: ScanMode.plate, cameraLoader: () async => throw CameraException('x', 'no camera')))), child: const Text('open')))))));
      await t.tap(find.text('open'));
      await t.pumpAndSettle();
      await t.tap(find.byTooltip('Close'));
      await t.pumpAndSettle();
      expect(find.text('open'), findsOneWidget);
      expect(find.text('Discard this scan?'), findsNothing);
      await dispose(t);
    });
  });

  group('widgets', () {
    testWidgets('MaskedText draws every ? as a boxed unknown', (t) async {
      await t.pumpWidget(MaterialApp(home: const Scaffold(body: MaskedText('MH12?B12?4'))));
      expect(find.byKey(const ValueKey('unknown-char')), findsNWidgets(2));
    });

    testWidgets('scan error dialog explains an engine failure and offers a retry', (t) async {
      late ErrorAction got;
      await t.pumpWidget(MaterialApp(home: Builder(builder: (ctx) => TextButton(onPressed: () async => got = await showScanErrorDialog(ctx, const ScanException(ScanErrorKind.engine, 'x', detail: 'OOM')), child: const Text('go')))));
      await t.tap(find.text('go'));
      await t.pumpAndSettle();
      expect(find.text("The reading engine can't start"), findsOneWidget);
      expect(find.textContaining('Closing other apps frees memory'), findsOneWidget);
      expect(find.textContaining('network'), findsNothing);
      await t.tap(find.text('Try again'));
      await t.pumpAndSettle();
      expect(got, ErrorAction.retry);
    });

    testWidgets('confirm dialog returns the choice', (t) async {
      bool? got;
      await t.pumpWidget(MaterialApp(home: Builder(builder: (ctx) => TextButton(onPressed: () async => got = await showConfirmDialog(ctx, title: 'Sure?', message: 'm', confirm: 'Yes', cancel: 'No'), child: const Text('go')))));
      await t.tap(find.text('go'));
      await t.pumpAndSettle();
      await t.tap(find.text('No'));
      await t.pumpAndSettle();
      expect(got, isFalse);
    });
  });
}
