// Scene reader: (1) the model-free decision logic, ported test-for-test from ml/tests/test_scene.py,
// and (2) a GOLDEN replay: the Python SceneReader's recorded detector / recogniser output is replayed
// into the Dart SceneReader over the same pixels, and the readings must come out the same.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/ctc.dart' show greedy;
import 'package:vinplate_tester/reader/decoder.dart';
import 'package:vinplate_tester/reader/geometry.dart';
import 'package:vinplate_tester/reader/grammar.dart';
import 'package:vinplate_tester/reader/pp_engine.dart' show RecOut;
import 'package:vinplate_tester/reader/scene_reader.dart';
import 'package:vinplate_tester/reader/types.dart';

// ======================================================================= test helpers
/// `vinplate.testing.make_logp`: the log-probabilities a CTC recogniser would emit for a known string,
/// with controllable failure modes.
Logp makeLogp(
  String text, {
  int framesPerChar = 3,
  double pPeak = 0.93,
  Map<int, (String, double)> confuse = const {},
  Set<int> flat = const {},
  Set<int> unreadable = const {},
  double eps = 0.002,
  bool smeared = false,
}) {
  final rows = <List<double>>[];
  List<double> norm(List<double> p) {
    final s = p.fold(0.0, (a, b) => a + b);
    return [for (final v in p) math.log(v / s)];
  }

  for (var i = 0; i < text.length; i++) {
    for (var f = 0; f < framesPerChar; f++) {
      final spike = smeared || f == 0;
      final p = List<double>.filled(kVocab, eps);
      if (!spike) {
        p[0] = 0.95;
      } else if (unreadable.contains(i)) {
        p[kChars.indexOf('?') + 1] = pPeak;
      } else if (flat.contains(i)) {
        p[0] = 0.6;
        for (var v = 1; v < kVocab; v++) {
          p[v] = 0.4 / (kVocab - 1);
        }
      } else {
        final k = kChars.indexOf(text[i]) + 1;
        if (confuse.containsKey(i)) {
          final (wrong, pw) = confuse[i]!;
          p[kChars.indexOf(wrong) + 1] = pw;
          p[k] = pPeak - pw > 0 ? pPeak - pw : 0.3;
        } else {
          p[k] = pPeak;
        }
      }
      rows.add(norm(p));
    }
    final gap = List<double>.filled(kVocab, eps);
    gap[0] = 0.95;
    rows.add(norm(gap));
  }
  return Logp.fromRows(rows);
}

class _NoEngine implements RecEngine {
  @override
  List<TextBox> detect(RgbImage rgb) => const [];
  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) => throw StateError('no recogniser in this test');
  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => throw StateError('no recogniser in this test');
  @override
  String greedyText(RecOut r) => throw StateError('no recogniser in this test');
}

TextBox box(double x, double y, double w, double h, [double angle = 0]) {
  final c = math.cos(angle * math.pi / 180), s = math.sin(angle * math.pi / 180);
  final ux = c * w, uy = s * w, vx = -s * h, vy = c * h;
  return TextBox([x, y, x + ux, y + uy, x + ux + vx, y + uy + vy, x + vx, y + vy], 0.9);
}

FieldReading field(String domain, String text, (int, int, int, int) b, {String status = 'accept', int? evidence, double confidence = 0.9}) {
  final r = Reading(
    text: text, valid: true, status: status, confidence: confidence, seqPosterior: 0.9, charConf: List.filled(text.length, 0.9),
    inferred: [], suspectedGaps: [], grammarCost: 0, greedyText: text, nBest: const [], reasons: [],
  );
  return FieldReading(domain: domain, reading: r, parsed: null, box: b, quads: const [], evidence: evidence ?? text.length);
}

final reader = SceneReader(_NoEngine());

/// Detector + recogniser stand-in (ml/tests/test_scene.py `FakeEngine`). A crop's "posterior" is just its
/// (h, w), so the log-posterior and the raw greedy text can be chosen per crop SIZE, and the sizes the
/// reader asks the recogniser about are recorded in [seen].
class FakeEngine implements RecEngine {
  FakeEngine(this.boxes, {this.lp, this.raw = '', this.lps = const {}, this.raws = const {}});
  final List<TextBox> boxes;
  final Logp? lp;
  final String raw;
  final Map<(int, int), Logp> lps;
  final Map<(int, int), String> raws;
  final List<(int, int)> seen = [];
  final List<(int, int)> _shapes = [];

  @override
  List<TextBox> detect(RgbImage rgb) => boxes;

  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) {
    final out = <RecOut>[];
    for (final c in crops) {
      seen.add((c.height, c.width));
      _shapes.add((c.height, c.width));
      out.add(RecOut(Float32List(0), (lps[(c.height, c.width)] ?? lp)?.t ?? 4, _shapes.length - 1));
    }
    return out;
  }

  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => (lps[_shapes[r.classes]] ?? lp ?? (throw StateError('no log-posterior for this crop'))).data;

  @override
  String greedyText(RecOut r) => raws[_shapes[r.classes]] ?? raw;
}

/// A deterministic textured scene (the Python tests use random noise).
RgbImage sceneImg({int w = 500, int h = 400}) {
  final d = Uint8List(w * h * 3);
  var s = 12345;
  for (var i = 0; i < d.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    d[i] = (s >> 16) & 0xff;
  }
  return RgbImage(w, h, d);
}

/// `make_logp(text)` with the spike frame of character [i] replaced by an explicit class distribution.
Logp lpWithSlot(String text, int i, Map<String, double> masses, {int fpc = 3}) {
  final lp = makeLogp(text, framesPerChar: fpc);
  final p = List<double>.filled(kVocab, 0.002);
  masses.forEach((ch, v) => p[kChars.indexOf(ch) + 1] = v);
  final z = p.fold(0.0, (a, b) => a + b);
  final d = Float64List.fromList(lp.data);
  for (var v = 0; v < kVocab; v++) {
    d[(fpc + 1) * i * kVocab + v] = math.log(p[v] / z);
  }
  return Logp(lp.t, d);
}

bool _isDigitChar(String c) => c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57;

RgbImage zeros(int h, int w) => RgbImage(w, h, Uint8List(w * h * 3));

const noScreen = {'score': 0.0, 'url': false, 'ui_words': <String>[], 'text_lines': 0, 'reasons': <String>[]};

// ================================================================================ tests
void main() {
  group('VIN folding', () {
    test('I, O and Q mass moves onto the digits', () {
      final rows = [List<double>.filled(kVocab, math.log(1e-6))];
      for (final (ch, p) in [('I', 0.6), ('O', 0.2), ('Q', 0.1)]) {
        rows[0][kChars.indexOf(ch) + 1] = math.log(p);
      }
      final out = foldVin(Logp.fromRows(rows));
      double pr(String c) => math.exp(out.at(0, kChars.indexOf(c) + 1));
      expect(pr('1'), greaterThan(0.55));
      expect(pr('0'), greaterThan(0.28));
      for (final c in ['I', 'O', 'Q']) {
        expect(pr(c), lessThan(1e-6));
      }
      expect([for (var v = 0; v < kVocab; v++) math.exp(out.at(0, v))].fold(0.0, (a, b) => a + b), closeTo(1.0, 1e-9));
    });

    test('a VIN read with an I where a 1 belongs is recovered', () {
      final lp = makeLogp('MAL907KE1E6602791', confuse: {13: ('I', 0.9)});
      final r = reader.decode('vin', [lp], allowLegacy: false)!.$1;
      expect(r.text == 'MAL907KE1E6602791' || r.text[13] == '1', isTrue, reason: r.text);
    });
  });

  group('VIN false positives', () {
    test('label text that looks like a VIN is rejected', () {
      final lp = makeLogp('TYREPRESSURE32PSI'); // exactly 17 legal VIN characters once I is folded to 1
      expect(reader.decode('vin', [lp], allowLegacy: false), isNull);
      expect(reader.decode('vin', [lp], allowLegacy: true), isNull);
    });

    test('the clutter filter never fires when the check digit passes', () {
      expect(SceneReader.isClutter('TYREPRESSURE32PS1'), isTrue);
      expect(SceneReader.isClutter('1M8GDM9AXKP042788'), isFalse);
    });

    test('legacy chassis needs permission and is never auto-accepted', () {
      final lp = makeLogp('81GZ16D60VVY9'); // 13 chars: not a 17-char VIN
      expect(reader.decode('vin', [lp], allowLegacy: false), isNull);
      final got = reader.decode('vin', [lp], allowLegacy: true);
      expect(got, isNotNull);
      expect(got!.$1.status, isNot('accept'));
      expect(got.$1.reasons.any((x) => x.contains('legacy')), isTrue);
    });

    test('a word is not a chassis number', () {
      expect(reader.decode('vin', [makeLogp('ENGINEFRONTWEIGHT')], allowLegacy: true), isNull);
    });

    test('a strict VIN read is accepted', () {
      final got = reader.decode('vin', [makeLogp('MAL907KE1E6602791')], allowLegacy: false);
      expect(got, isNotNull);
      expect(got!.$1.text, 'MAL907KE1E6602791');
      expect(['accept', 'review'], contains(got.$1.status));
    });
  });

  group('plate windows', () {
    for (final (noisy, plate) in [
      ('INDMH12AB1234', 'MH12AB1234'), // hologram text before the plate
      ('UP16C0059TC20', 'UP16C0059'), // temporary-registration mark after it
      ('IND22BH1234AAX', '22BH1234AA'),
    ]) {
      test('a plate inside surrounding text is found: $noisy', () {
        final got = reader.decode('plate', [makeLogp(noisy)], allowLegacy: false);
        expect(got, isNotNull);
        final (r, note) = got!;
        expect(r.text, plate, reason: '$noisy -> ${r.text} (${r.status})');
        expect(note, contains('sub-window'));
        expect(parsePlate(r.text), isNotNull);
      });
    }

    test('a clean plate is decoded whole', () {
      final (r, note) = reader.decode('plate', [makeLogp('TN71JE5250')], allowLegacy: false)!;
      expect(r.text, 'TN71JE5250');
      expect(r.status, 'accept');
      expect(note, '');
    });

    test('an unreadable character is flagged, not invented', () {
      final (r, _) = reader.decode('plate', [makeLogp('KA01AB1234', unreadable: {4})], allowLegacy: false)!;
      expect(r.status, 'review');
      expect(r.masked, contains('?'));
    });

    test('a hallucinated short read is not a plate', () {
      final r = reader.decode('plate', [makeLogp('MH12')], allowLegacy: false);
      expect(r == null || r.$1.status != 'accept', isTrue);
    });
  });

  group('line ownership', () {
    test('a text line is never both the plate and the VIN', () {
      final plate = field('plate', 'MH12AB1234', (100, 100, 300, 140));
      final vin = field('vin', 'MAL907KE1E6602791', (95, 98, 310, 142), status: 'review', evidence: 12);
      final (p, v) = assignFields([plate, vin]);
      expect(identical(p, plate) && identical(v, vin), isFalse);
      expect(p == null || v == null || p.box != v.box, isTrue);
    });

    test('separate lines give both fields', () {
      final plate = field('plate', 'MH12AB1234', (100, 100, 300, 140));
      final vin = field('vin', 'MAL907KE1E6602791', (100, 400, 500, 440));
      final (p, v) = assignFields([plate, vin]);
      expect(identical(p, plate), isTrue);
      expect(identical(v, vin), isTrue);
    });

    test('weak or rejected candidates are not reported', () {
      final weak = field('plate', 'MH12AB1234', (0, 0, 10, 10), evidence: 3);
      final rejected = field('plate', 'MH12AB1234', (0, 0, 10, 10), status: 'reject');
      final low = field('plate', 'MH12AB1234', (0, 0, 10, 10), confidence: 0.05);
      final (p, v) = assignFields([weak, rejected, low]);
      expect(p, isNull);
      expect(v, isNull);
    });

    test('ranking prefers accept over review, then more evidence, then confidence; ties keep the first', () {
      final a = field('plate', 'MH12AB1234', (0, 0, 10, 10), status: 'review');
      final b = field('plate', 'KA05MN4821', (500, 0, 510, 10));
      expect(identical(assignFields([a, b]).$1, b), isTrue);
      final c = field('plate', 'TN71JE5250', (900, 0, 910, 10));
      expect(identical(assignFields([b, c]).$1, b), isTrue); // equal rank: the first stays
    });
  });

  group('geometry predicates', () {
    test('two-row plate rows are stacked, not side by side', () {
      final top = box(100, 100, 90, 40), bottom = box(100, 150, 200, 44);
      expect(isStacked(top, bottom), isTrue);
      expect(isSideBySide(top, bottom), isFalse);
      expect(isStacked(bottom, top), isFalse);
    });

    test('split plate pieces are side by side', () {
      final a = box(100, 100, 90, 40), b = box(215, 102, 200, 40);
      expect(isSideBySide(a, b), isTrue);
      expect(isStacked(a, b), isFalse);
    });

    test('tilted rows still stack', () {
      expect(isStacked(box(100, 100, 90, 40, 12), box(100 - 9, 100 + 50, 200, 42, 12)), isTrue);
    });

    test('far apart boxes are not one plate', () {
      expect(isNear(box(0, 0, 100, 30), box(0, 400, 100, 30)), isFalse);
      expect(isNear(box(0, 0, 100, 30), box(0, 40, 100, 30)), isTrue);
    });

    test('masked readings are compatible across photos', () {
      expect(readsCompatible('MH12AB1234', 'MH12AB1234'), isTrue);
      expect(readsCompatible('MH12??1234', 'MH12AB1234'), isTrue);
      expect(readsCompatible('MH12AB1234', 'MH12AB1284'), isTrue); // one misread digit tolerated
      expect(readsCompatible('MH12AB1234', 'KA01XY9876'), isFalse);
      expect(readsCompatible('MH12AB1234', 'MH12AB123'), isFalse);
    });

    test('state-code look-alikes (one common substitution away from a state code)', () {
      expect(nearState('MH'), isTrue); // already a state code
      expect(nearState('K4'), isTrue); // 4 -> A: KA
      expect(nearState('0D'), isTrue); // 0 -> O: OD
      expect(nearState('M8'), isFalse); // M8, MB: no such state
      expect(nearState('XQ'), isFalse);
      expect(nearState('M'), isFalse);
    });
  });

  group('decode-time safety policy', () {
    test('look-alike ambiguity is shown as unknown, not read', () {
      // series letter Q vs O: the net says Q (68%) but O has 25% and the grammar allows both
      final lp = makeLogp('WB97MQ3156', confuse: {5: ('O', 0.25)});
      final (r, _) = reader.decode('plate', [lp], allowLegacy: false)!;
      expect(r.status, 'review');
      expect(r.inferred, contains(5));
      expect(r.masked[5], '?');
      expect(r.reasons.any((x) => x.contains('look-alike')), isTrue);
    });

    test('a look-alike the grammar forbids is not flagged', () {
      // a digit slot cannot be the letter O, so an O alternative there is not an ambiguity
      final lp = makeLogp('TN71JE5250', confuse: {9: ('O', 0.25)});
      final (r, _) = reader.decode('plate', [lp], allowLegacy: false)!;
      expect(r.text, 'TN71JE5250');
      expect(r.status, 'accept');
    });

    test('a series-less plate is never auto-accepted', () {
      final (r, _) = reader.decode('plate', [makeLogp('CG656426')], allowLegacy: false)!;
      expect(r.status, 'review');
      expect(r.reasons.any((x) => x.contains('series')), isTrue);
    });

    test('a short number is never auto-accepted', () {
      final (r, _) = reader.decode('plate', [makeLogp('MH12AB123')], allowLegacy: false)!;
      expect(r.status, isNot('accept'));
    });

    test('series letter I/O/Q is never auto-accepted', () {
      for (final plate in ['WB97MO3156', 'AR87QC5837']) {
        final (r, _) = reader.decode('plate', [makeLogp(plate)], allowLegacy: false)!;
        expect(r.status, 'review', reason: '$plate ${r.status} ${r.reasons}');
        expect(r.reasons.any((x) => x.contains('I/O/Q')), isTrue);
      }
      final (ok, _) = reader.decode('plate', [makeLogp('WB97MD3156')], allowLegacy: false)!;
      expect(ok.status, 'accept');
    });
  });

  group('reader plumbing', () {
    test('an image with no text gives no findings and does not throw', () {
      final r = reader.read(RgbImage(64, 48, Uint8List(64 * 48 * 3)));
      expect(r.plate, isNull);
      expect(r.vin, isNull);
      expect(r.lines, isEmpty);
      expect(r.size, (64, 48));
      expect(r.timings.keys, containsAll(['detect', 'recognise', 'decode', 'total']));
    });

    test('python-compatible rounding and quad rounding', () {
      expect(pyRound(0.0625, 3), 0.062); // exact tie -> even
      expect(pyRound(0.2345, 3), 0.234); // binary value just below the tie
      expect(pyRound(0.5, 0), 0.0);
      expect(pyRound(1.5, 0), 2.0);
      expect(pyRound(0.9996, 3), 1.0);
    });
  });

  // ================================================================ vehicle category (plate colours + layout)
  group('vehicle category', () {
    /// A plate-like crop: [bg] everywhere with ink bars over the middle 60 % of the rows (ml/tests/test_category.py).
    RgbImage plateCrop((int, int, int) bg, (int, int, int) ink, {int h = 48, int w = 240}) {
      final d = Uint8List(h * w * 3);
      for (var i = 0; i < h * w; i++) {
        d[i * 3] = bg.$1;
        d[i * 3 + 1] = bg.$2;
        d[i * 3 + 2] = bg.$3;
      }
      for (var x = 0; x < w; x += 16) {
        for (var y = (0.2 * h).toInt(); y < (0.8 * h).toInt(); y++) {
          for (var xx = x; xx < x + 4 && xx < w; xx++) {
            final i = (y * w + xx) * 3;
            d[i] = ink.$1;
            d[i + 1] = ink.$2;
            d[i + 2] = ink.$3;
          }
        }
      }
      return RgbImage(w, h, d);
    }

    /// A grey scene with [crop] pasted at (x, y); the detector stand-in returns exactly that box.
    (RgbImage, TextBox) sceneWith(RgbImage crop, {int x = 60, int y = 80, int w = 760, int h = 420}) {
      final d = Uint8List(w * h * 3);
      for (var i = 0; i < w * h; i++) {
        d[i * 3] = 112;
        d[i * 3 + 1] = 112;
        d[i * 3 + 2] = 118;
      }
      for (var r = 0; r < crop.height; r++) {
        d.setRange(((y + r) * w + x) * 3, ((y + r) * w + x + crop.width) * 3, crop.rgb, r * crop.width * 3);
      }
      return (RgbImage(w, h, d), box(x.toDouble(), y.toDouble(), crop.width.toDouble(), crop.height.toDouble()));
    }

    const yellow = (255, 206, 22), black = (18, 18, 18), green = (22, 150, 72), white = (248, 248, 246);
    const resultKeys = {'key', 'label', 'confidence', 'bg', 'ink', 'reasons', 'alternatives', 'ev', 'bh'};

    for (final (name, bg, ink, key, ev) in <(String, (int, int, int), (int, int, int), String, bool)>[
      ('commercial', yellow, black, 'commercial', false),
      ('private', white, black, 'private', false),
      ('EV (private)', green, white, 'ev_private', true),
    ]) {
      test('the reader attaches the category of the plate crop: $name', () {
        final (img0, b) = sceneWith(plateCrop(bg, ink));
        final res = SceneReader(FakeEngine([b], lp: makeLogp('MH12AB1234'), raw: 'MH12AB1234'), multiView: false).read(img0, domains: {'plate'});
        expect(res.plate, isNotNull);
        final cat = res.plate!.category;
        expect(cat, isNotNull);
        expect(cat!['key'], key);
        expect(cat['ev'], ev);
        final j = res.plate!.toJson();
        expect(j['category'], same(cat));
        expect((j['category'] as Map).keys.toSet(), resultKeys);
        jsonEncode(j['category']);
      });
    }

    test('a VIN has no category, and the key is still present', () {
      final j = field('vin', 'MAL907KE1E6602791', (0, 0, 10, 10)).toJson();
      expect(j.containsKey('category'), isTrue);
      expect(j['category'], isNull);
    });

    test('fused photos take the category from the best photo, judged against the fused text layout', () {
      final (img0, b) = sceneWith(plateCrop(yellow, black));
      final res = SceneReader(FakeEngine([b], lp: makeLogp('MH12AB1234'), raw: 'MH12AB1234'), multiView: false).readFrames([img0, img0], domains: {'plate'});
      expect(res.plate!.views, 2);
      expect(res.plate!.notes.first, contains('fused 2 photos'));
      expect(res.plate!.category, isNotNull);
      expect(res.plate!.category!['key'], 'commercial');
      expect(res.plate!.toJson()['category']['key'], 'commercial');
    });
  });

  // ================================================================ aspect normalisation (CTC time-step budget)
  group('aspect normalisation', () {
    test('a condensed plate crop is stretched to the budget (the real HR88B8888 box: 2.6:1 gave only 16 CTC steps)', () {
      final out = normaliseAspect(zeros(99, 258));
      expect((out.height, out.width), (99, 396)); // 4.0 * 99
      expect(out.rgb.length, 99 * 396 * 3);
    });

    test('wide text is never touched: the very same object comes back', () {
      for (final (h, w) in [(48, 200), (50, 200), (48, 600), (30, 1200)]) {
        // (50, 200) is exactly 4.0
        final crop = zeros(h, w);
        expect(identical(normaliseAspect(crop), crop), isTrue, reason: '$h x $w');
      }
    });

    test('the stretch is capped at maxStretch', () {
      final out = normaliseAspect(zeros(40, 40)); // 1:1 would need x4, capped at x2.5
      expect((out.height, out.width), (40, 100));
    });

    test('the new width rounds half UP, not to even', () {
      expect(normaliseAspect(zeros(4, 5)).width, 13); // 5 * 2.5 = 12.5 -> 13 (round-half-even would give 12)
    });

    test('the constants are the ones the Python reference uses', () {
      expect((minAspect, maxStretch), (4.0, 2.5));
    });

    test('every line crop the recogniser sees meets the budget (two condensed boxes that also form a region)', () {
      final eng = FakeEngine([box(150, 100, 110, 52), box(140, 160, 130, 56)], lp: makeLogp('MH12AB1234'), raw: 'MH12AB1234');
      SceneReader(eng, multiView: false).read(sceneImg(), domains: {'plate'});
      expect(eng.seen, contains((52, 208))); // the line crops, stretched to exactly 4 : 1
      expect(eng.seen, contains((56, 224)));
      expect(eng.seen, isNot(contains((52, 110)))); // ... and never recognised unstretched
      expect(eng.seen, isNot(contains((56, 130))));
      expect(eng.seen.every((hw) => hw.$2 / hw.$1 >= 3.8), isTrue, reason: '${eng.seen}'); // the pair crop is content-dependent and hits the x2.5 cap (3.86 here, 3.94 in the Python run: its noise image differs)
    });

    test('two stacked rows of one plate are stretched before they are recognised', () {
      final lp = makeLogp('MH12AB1234');
      final eng = FakeEngine([box(100, 100, 120, 40), box(100, 150, 120, 40)], lp: lp, raw: 'MH12AB1234');
      SceneReader(eng, multiView: false).read(sceneImg(), domains: {'plate'});
      expect(eng.seen, isNotEmpty);
      expect(eng.seen.every((hw) => hw.$2 / hw.$1 >= minAspect - 0.05), isTrue, reason: 'sizes seen: ${eng.seen}');
    });

    test('fused photos re-read stretched crops too, and keep the fused posterior', () {
      final eng = FakeEngine([box(100, 100, 120, 40)], lp: makeLogp('TN71JE5250'), raw: 'TN71JE5250'); // 3.0 : 1
      final res = SceneReader(eng, multiView: false).readFrames([sceneImg(), sceneImg()], domains: {'plate'});
      expect(res.plate, isNotNull);
      expect(res.plate!.views, 2);
      expect(res.plate!.notes[0], contains('fused 2 photos'));
      expect(eng.seen, contains((40, 160))); // observe AND the fusion re-read are stretched
      expect(eng.seen, isNot(contains((40, 120))));
      expect(res.plate!.posterior, isNotNull);
      expect(res.plate!.toJson()['unknown_candidates'], isA<List>());
    });
  });

  // ================================================================ unknown-slot candidates (the 'innate ability')
  group('unknown-slot candidates', () {
    test('an unknown letter slot lists only legal letters, ranked (HR88?8888: the slot after the RTO MUST be a letter)', () {
      final lp = lpWithSlot('HR88B8888', 4, {'?': 0.55, 'B': 0.30, '8': 0.10, 'R': 0.03, 'D': 0.02});
      final out = slotCandidates('HR88?8888', lp, plateGrammar);
      expect([for (final e in out) e['position']], [4]);
      final opts = (out[0]['options'] as List).cast<Map<String, dynamic>>();
      expect([for (final o in opts) o['char']], ['B', 'R', 'D']); // ranked; the digit 8 is never offered
      final ps = [for (final o in opts) o['p'] as double];
      expect(ps, [...ps]..sort((a, b) => b.compareTo(a)));
      expect(ps[0], closeTo(0.768, 0.01)); // softmax over the LEGAL characters of ln(peak) + the grammar's log-prior (0.758 on the peaks alone)
      expect(out[0]['suggested'], 'B'); // dominant: >= 0.6 and 0.3 ahead of the next
    });

    test('two close letters give options but no suggestion', () {
      final lp = lpWithSlot('HR88B8888', 4, {'?': 0.5, 'B': 0.25, 'D': 0.22});
      final e = slotCandidates('HR88?8888', lp, plateGrammar).single;
      expect([for (final o in e['options'] as List) (o as Map)['char']], ['B', 'D']);
      expect(e['suggested'], isNull);
    });

    test('options below five percent are dropped', () {
      final lp = lpWithSlot('HR88B8888', 4, {'?': 0.7, 'B': 0.2});
      final e = slotCandidates('HR88?8888', lp, plateGrammar).single;
      expect([for (final o in e['options'] as List) (o as Map)['char']], ['B']); // the other 25 letters are < 5% each
      expect(e['suggested'], 'B');
    });

    test('a digit-only slot offers only digits', () {
      final lp = lpWithSlot('HR88B8888', 7, {'?': 0.5, '8': 0.3, '3': 0.1, 'B': 0.05});
      final e = slotCandidates('HR88B88?8', lp, plateGrammar).single; // inside the 4-digit number: no letter is legal
      final chars = [for (final o in e['options'] as List) (o as Map)['char'] as String];
      expect(e['position'], 7);
      expect(chars.take(2), ['8', '3']);
      expect(chars.every((c) => _isDigitChar(c)), isTrue);
    });

    test('a slot where the grammar allows a letter or a digit offers both kinds, ranked', () {
      final lp = lpWithSlot('HR88B8888', 5, {'?': 0.5, '8': 0.3, '3': 0.1, 'B': 0.05});
      final e = slotCandidates('HR88B?888', lp, plateGrammar).single; // second series letter OR first number digit
      // the grammar's prior makes a number digit likelier than a second series letter here, so B (5%) falls away
      expect([for (final o in e['options'] as List) (o as Map)['char']].take(2), ['8', '3']);
      expect(e['suggested'], '8'); // ... and with the letters out of the running the digit dominates (>= 0.6 and 0.3 ahead)
    });

    List<String> offered(String text, int i, Map<String, double> masses, Grammar g) {
      final out = slotCandidates(text, lpWithSlot(text.replaceAll('?', 'X'), i, masses), g);
      expect(out, isNotEmpty, reason: '$text $masses');
      return [for (final o in out.single['options'] as List) (o as Map)['char'] as String];
    }

    test('the PATTERN decides: slots that must be letters offer letters only, even when the digits have more evidence', () {
      // evidence says 8 (40%) > B (20%), but the character after the RTO digits is a series LETTER
      expect(offered('HR88?8888', 4, {'?': 0.3, '8': 0.4, 'B': 0.2, 'R': 0.05}, plateGrammar).every((c) => !_isDigitChar(c)), isTrue);
      final a = offered('MH12?B1234', 4, {'?': 0.3, '8': 0.4, 'A': 0.2, 'K': 0.05}, plateGrammar);
      expect(a.first, 'A');
      expect(a.every((c) => !_isDigitChar(c)), isTrue, reason: '$a');
      final b = offered('MH12A?1234', 5, {'?': 0.3, '8': 0.4, 'B': 0.2, 'K': 0.05}, plateGrammar);
      expect(b.first, 'B');
      expect(b.every((c) => !_isDigitChar(c)), isTrue, reason: '$b');
    });

    test('a slot inside the number offers digits only, even when a letter has more evidence', () {
      final d = offered('MH12AB12?4', 8, {'?': 0.3, 'B': 0.4, '3': 0.2, '8': 0.05}, plateGrammar);
      expect(d.first, '3');
      expect(d.every(_isDigitChar), isTrue, reason: '$d');
    });

    test('a VIN unknown at the model-year position ranks a valid year code above an invalid one at equal evidence', () {
      // position 10 (index 9): U and Z and 0 are never model-year codes; E (2014) is
      final got = offered('MAL907KE1?6602791', 9, {'?': 0.4, 'E': 0.25, 'U': 0.25}, vinGrammar);
      expect(got.first, 'E', reason: '$got');
      final withZero = offered('MAL907KE1?6602791', 9, {'?': 0.4, 'L': 0.25, '0': 0.25}, vinGrammar);
      expect(withZero.first, 'L', reason: '$withZero');
    });

    test('an unknown in the VIN serial ranks a digit above a letter at equal evidence', () {
      // the last four characters of a VIN are digits (a letter there is not a plausible serial)
      final got = offered('MAL907KE1E66027?1', 15, {'?': 0.4, '9': 0.25, 'K': 0.25}, vinGrammar);
      expect(got.first, '9', reason: '$got');
    });

    test('inferred but known characters get candidates too', () {
      final lp = lpWithSlot('HR88B8888', 4, {'B': 0.5, 'R': 0.2, 'D': 0.1});
      final out = slotCandidates('HR88B8888', lp, plateGrammar, inferred: [4]);
      expect([for (final e in out) e['position']], [4]);
      expect([for (final o in out[0]['options'] as List) (o as Map)['char']].take(2), ['B', 'R']);
      expect(slotCandidates('HR88B8888', lp, plateGrammar), isEmpty); // nothing unknown, nothing weak: nothing to offer
    });

    test('VIN slots offer the folded posterior, so I / O / Q never appear', () {
      final lp = foldVin(lpWithSlot('MAL907KE1E6602791', 3, {'?': 0.5, '9': 0.2, 'O': 0.15, 'Q': 0.1, '8': 0.05}));
      final e = slotCandidates('MAL?07KE1E6602791', lp, vinGrammar).single;
      final got = {for (final o in e['options'] as List) (o as Map)['char'] as String};
      expect(got, containsAll(['9', '0']));
      expect(got.intersection({'I', 'O', 'Q'}), isEmpty);
    });

    test('no candidates without an alignment', () {
      expect(slotCandidates('?' * 40, makeLogp('ABC'), plateGrammar), isEmpty); // more characters than frames
      expect(slotCandidates('HR88?8888', null, plateGrammar), isEmpty); // no posterior kept
      expect(slotCandidates('', makeLogp('ABC'), plateGrammar), isEmpty);
    });

    test('the unknown slot reaches the reading JSON through the scene reader', () {
      final lp = lpWithSlot('HR88B8888', 4, {'?': 0.55, 'B': 0.30, '8': 0.10, 'R': 0.03, 'D': 0.02});
      final eng = FakeEngine([box(100, 100, 300, 60)], lp: lp, raw: 'HR88B8888');
      final res = SceneReader(eng, multiView: false).read(sceneImg(), domains: {'plate'});
      final d = res.plate!.toJson();
      expect(d['value'], 'HR88?8888');
      expect(d['status'], isNot('accept'));
      final e = (d['unknown_candidates'] as List).single as Map<String, dynamic>;
      expect(e['position'], 4);
      expect((d['value'] as String)[e['position'] as int], '?');
      final opts = (e['options'] as List).cast<Map<String, dynamic>>();
      expect(opts.first['char'], 'B');
      expect(opts.any((o) => _isDigitChar(o['char'] as String)), isFalse);
      expect(e['suggested'], 'B');
      final clean = SceneReader(FakeEngine([box(100, 100, 300, 60)], lp: makeLogp('TN71JE5250'), raw: 'TN71JE5250'), multiView: false)
          .read(sceneImg(), domains: {'plate'});
      expect(clean.plate!.toJson()['value'], 'TN71JE5250');
      expect(clean.plate!.toJson()['unknown_candidates'], isEmpty);
    });
  });

  // ================================================================ screen context
  group('screen context', () {
    const attackLines = [
      'DE?CX?1', 'E-Ringi', 'Kafka R', 'Quest27', 'AskGartner Home', 'e-Ringi',
      'https://www.bing.com/images/search?view=detailV2&ccid=3SUMWcMG&id=6770EFC1A91614CEA936',
      "HR88B8888 Becomes India's Most Expensive Car Registration Number Sold ...",
      'newsable.asianetnews.com 1280 x 720 jpeg 11 mos ago', 'Search', 'HR8888', 'IND', 'HR88BS', 'REGISTR',
      'Related searc', 'California', 'See more details', 'Car Number Plate in Mumbai', 'Car Number Plate in Gurgaon',
      'Renewal of RC for 15-year-old cars to', '4444 Number Plate Price and', 'Procedure of renewing vehicle registration',
      'Registration Process in India', 'How to Che', //
    ];

    test('attack-like screen text (the real monitor photo) scores the maximum', () {
      final ctx = screenContext(attackLines, excluded: {10, 11}); // the plate lines themselves are not context
      expect(ctx['url'], isTrue);
      expect(ctx['text_lines'] as int, greaterThanOrEqualTo(14));
      expect(ctx['ui_words'], ['DETAILS', 'IMAGES', 'RELATED', 'SEARCH']);
      expect(ctx['score'], 3.0);
      expect((ctx['reasons'] as List).length, 3);
      expect(ctx.keys.toSet(), {'score', 'url', 'ui_words', 'text_lines', 'reasons'});
    });

    test('a plain plate scene has no screen context', () {
      expect(screenContext(['MARUTI SUZUKI', 'MH12AB1234'], excluded: {1}), {...noScreen, 'text_lines': 1});
      expect(screenContext([]), noScreen);
    });

    test('a dealer-sticker URL alone weighs one point', () {
      final ctx = screenContext(['www.sharmamotors.com', 'MH12AB1234'], excluded: {1});
      expect(ctx['url'], isTrue);
      expect(ctx['ui_words'], isEmpty);
      expect(ctx['score'], 1.0);
    });

    test('many lines alone weigh half a point, and short lines do not count', () {
      final ctx = screenContext([for (var i = 0; i < 20; i++) 'SHOP $i', 'AB', '1', '?!']);
      expect(ctx['text_lines'], 20);
      expect(ctx['score'], 0.5);
      expect(ctx['url'], isFalse);
      expect(screenContext([for (var i = 0; i < 13; i++) 'SHOP $i'])['score'], 0.0); // 13 < 14
    });

    test('three DISTINCT interface words are needed', () {
      expect(screenContext(['Search', 'Settings'])['score'], 0.0);
      expect(screenContext(['Search', 'Settings', 'Download'])['score'], 1.5);
      expect(screenContext(['search search search'])['ui_words'], ['SEARCH']); // a repeated word is ONE word
    });

    test('URL detection rules', () {
      for (final t in ['http://x', 'HTTPS://A.B', 'www.example', 'visit example.com now', 'a.in', 'b.org/x', 'c.net', 'd.co']) {
        expect(screenContext([t])['url'], isTrue, reason: t);
      }
      for (final t in ['INDIA', 'CONTACT 98765', 'x.INDIA', 'SECOND', 'A.COMPANY']) {
        expect(screenContext([t])['url'], isFalse, reason: t);
      }
    });

    test('excluded lines and non-ASCII text do not count', () {
      expect(screenContext(['https://a.b'], excluded: {0})['score'], 0.0);
      expect(screenContext(['汉字汉字汉字汉字'])['text_lines'], 0); // only ASCII letters / digits count
    });

    FakeEngine contextEngine(String sideRaw) => FakeEngine(
          [box(100, 100, 300, 60), box(20, 330, 120, 20)],
          lp: makeLogp('TN71JE5250'),
          lps: {(20, 120): makeLogp('IMAGESBING')},
          raws: {(60, 300): 'TN71JE5250 https://www.x.com', (20, 120): sideRaw},
        );

    test('the scene reader attaches the context and excludes the chosen plate lines', () {
      final res = SceneReader(contextEngine('MARUTI SUZUKI'), multiView: false).read(sceneImg(), domains: {'plate'});
      expect(res.plate!.toJson()['value'], 'TN71JE5250');
      // the URL is on the PLATE's own line: not scene context. Only the badge line counts, as text.
      expect(res.screenContext, {...noScreen, 'text_lines': 1});
      expect(res.toJson()['screen_context'], res.screenContext);
      final res2 = SceneReader(contextEngine('https://www.bing.com/images'), multiView: false).read(sceneImg(), domains: {'plate'});
      expect(res2.screenContext['url'], isTrue);
      expect(res2.screenContext['score'], 1.0);
      expect(res2.screenContext['reasons'], ['a web address (URL) is visible in the scene']);
    });

    test('readFrames keeps the strongest context (the first of equals)', () {
      final rd = SceneReader(contextEngine('https://www.bing.com/images'), multiView: false);
      final res = rd.readFrames([sceneImg(), sceneImg()], domains: {'plate'});
      expect(res.screenContext['url'], isTrue);
      expect(res.perImage.length, 2);
      final single = rd.readFrames([sceneImg()], domains: {'plate'});
      expect(single.screenContext, res.screenContext);
    });
  });

  group('golden replay of the Python scene reader', goldenTests);
}

// ================================================================================ golden
class ReplayEngine implements RecEngine {
  ReplayEngine(this.events, this.outs) : _decoded = List<Float64List?>.filled(outs.length, null);
  final List<Map<String, dynamic>> events;
  final List<Map<String, dynamic>> outs;
  final List<Float64List?> _decoded;
  final List<String> problems = [];
  int next = 0;

  Map<String, dynamic>? _take(String kind) {
    if (next >= events.length) {
      problems.add('unexpected extra $kind call (the Python made only ${events.length} engine calls)');
      return null;
    }
    final e = events[next++];
    if (e['k'] != kind) {
      problems.add('call #${next - 1}: Dart called $kind, Python called ${e['k']}');
      return null;
    }
    return e;
  }

  @override
  List<TextBox> detect(RgbImage rgb) {
    final e = _take('detect');
    if (e == null) return const [];
    final size = (e['size'] as List).cast<int>();
    if (size[0] != rgb.width || size[1] != rgb.height) problems.add('detect: image ${rgb.width}x${rgb.height} but recorded ${size[0]}x${size[1]}');
    return [for (final b in (e['boxes'] as List).cast<Map<String, dynamic>>()) TextBox([for (final v in b['quad'] as List) (v as num).toDouble()], (b['score'] as num).toDouble())];
  }

  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) {
    final e = _take('rec');
    if (e == null) return [for (final _ in crops) _noOut()];
    final shapes = (e['shapes'] as List).map((s) => (s as List).cast<int>()).toList();
    final ids = (e['ids'] as List).cast<int>();
    if (shapes.length != crops.length) {
      problems.add('rec call #${next - 1}: Dart sent ${crops.length} crops, Python sent ${shapes.length}');
      return [for (final _ in crops) _noOut()];
    }
    for (var i = 0; i < crops.length; i++) {
      if (crops[i].height != shapes[i][0] || crops[i].width != shapes[i][1]) {
        problems.add('rec call #${next - 1} crop $i: Dart ${crops[i].height}x${crops[i].width}, Python ${shapes[i][0]}x${shapes[i][1]}');
      }
    }
    return [for (final id in ids) RecOut(Float32List(0), outs[id]['t'] as int, id)];
  }

  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) {
    if (temperature != 1.0) problems.add('temperature $temperature (goldens are recorded at 1.0)');
    if (r.classes < 0) {
      // a desynchronised call: hand back a blank so the run can continue and report everything
      final d = Float64List(4 * kVocab);
      for (var t = 0; t < 4; t++) {
        d[t * kVocab] = 0;
        for (var v = 1; v < kVocab; v++) {
          d[t * kVocab + v] = -20;
        }
      }
      return d;
    }
    return _decoded[r.classes] ??= _decode(outs[r.classes]);
  }

  /// The recorded Python engine had no RapidOCR-style `greedy` (its Recorder wraps only the log-probs), so
  /// the reader there fell back to the vocabulary-limited greedy text; do the same unless the recording
  /// carries the raw text ('raw').
  @override
  String greedyText(RecOut r) {
    if (r.classes < 0) return '';
    final raw = outs[r.classes]['raw'];
    if (raw is String) return raw;
    final d = toLogp(r);
    return greedy(Logp(r.t, d)).$1;
  }

  static Float64List _decode(Map<String, dynamic> o) {
    final raw = Uint8List.fromList(zlib.decode(base64Decode(o['z'] as String)));
    final f32 = raw.buffer.asFloat32List(raw.offsetInBytes, raw.lengthInBytes ~/ 4);
    return Float64List.fromList([for (final v in f32) v]);
  }
}

/// A stand-in for a call that could not be matched to a recording (the run continues and reports it).
RecOut _noOut() => RecOut(Float32List(0), 4, -1);

RgbImage decodePng(String b64) {
  final im = img.decodePng(base64Decode(b64));
  if (im == null) throw StateError('PNG did not decode');
  return RgbImage(im.width, im.height, Uint8List.fromList(im.convert(numChannels: 3).getBytes(order: img.ChannelOrder.rgb)));
}

bool near(Object? a, Object? b, double tol) => (a as num).toDouble() - (b as num).toDouble() <= tol && (b).toDouble() - (a).toDouble() <= tol;

void cmpValue(List<String> diffs, String path, Object? actual, Object? expected, {double tol = 1e-6}) {
  if (expected is Map) {
    if (actual is! Map) {
      diffs.add('$path: expected a map, got $actual');
      return;
    }
    for (final k in expected.keys) {
      cmpValue(diffs, '$path.$k', actual[k], expected[k], tol: tol);
    }
    for (final k in actual.keys) {
      if (!expected.containsKey(k)) diffs.add('$path.$k: unexpected key');
    }
  } else if (expected is List) {
    if (actual is! List || actual.length != expected.length) {
      diffs.add('$path: expected $expected, got $actual');
      return;
    }
    for (var i = 0; i < expected.length; i++) {
      cmpValue(diffs, '$path[$i]', actual[i], expected[i], tol: tol);
    }
  } else if (expected is num) {
    if (actual is! num || !near(actual, expected, tol)) diffs.add('$path: expected $expected, got $actual');
  } else if (actual != expected) {
    diffs.add('$path: expected ${jsonEncode(expected)}, got ${jsonEncode(actual)}');
  }
}

/// Compares one FieldReading (as `toJson`) with Python's `as_dict`. Quads are compared to 0.2 px (they
/// are rounded to 0.1), everything else to 1e-6.
List<String> compareField(String label, Map<String, dynamic>? actual, Map<String, dynamic>? expected) {
  final diffs = <String>[];
  if (expected == null || actual == null) {
    if (expected != actual) diffs.add('$label: expected ${expected == null ? 'none' : (expected['masked'])}, got ${actual == null ? 'none' : (actual['masked'])}');
    return diffs;
  }
  for (final k in expected.keys) {
    final p = '$label.$k';
    if (k == 'quads') {
      // Python nests each quad as 4 [x, y] pairs; the Dart contract keeps 8 flat numbers (same values)
      final flat = [for (final q in expected[k] as List) [for (final pt in q as List) ...(pt as List)]];
      cmpValue(diffs, p, actual[k], flat, tol: 0.2);
    } else if (k == 'quality') {
      final a = actual[k] as Map<String, dynamic>?, e = expected[k] as Map<String, dynamic>?;
      if (e == null || a == null) {
        if (e != a) diffs.add('$p: expected $e, got $a');
      } else {
        for (final kk in e.keys) {
          cmpValue(diffs, '$p.$kk', a[kk], e[kk], tol: kk == 'metrics' ? 1e-2 : 1e-6);
        }
      }
    } else {
      cmpValue(diffs, p, actual[k], expected[k]);
    }
  }
  for (final k in actual.keys) {
    if (!expected.containsKey(k)) diffs.add('$label.$k: key the Python does not produce');
  }
  return diffs;
}

void goldenTests() {
  final file = File('test/fixtures/golden/scene_golden.json');
  if (!file.existsSync()) {
    test('scene_golden.json is missing - run ml/scripts/make_golden_scene.py', () => fail('golden file missing'));
    return;
  }
  final data = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;
  final cases = (data['cases'] as List).cast<Map<String, dynamic>>();
  final byName = {for (final c in cases) c['name'] as String: c};

  RgbImage imageOf(Map<String, dynamic> entry) {
    if (entry.containsKey('ref')) return imageOf(((byName[entry['ref']]!['images'] as List)[0] as Map).cast<String, dynamic>());
    return decodePng(entry['png'] as String);
  }

  for (final c in cases) {
    final name = c['name'] as String;
    final runs = (c['runs'] as List).cast<Map<String, dynamic>>();
    for (var ri = 0; ri < runs.length; ri++) {
      final run = runs[ri];
      final args = (run['args'] as Map).cast<String, dynamic>();
      test('$name run $ri ${args['domains']} legacy=${args['allow_legacy']}', () {
        final engine = ReplayEngine((run['events'] as List).cast<Map<String, dynamic>>(), (run['outs'] as List).cast<Map<String, dynamic>>());
        final rd = SceneReader(engine);
        final images = [for (final e in (c['images'] as List).cast<Map<String, dynamic>>()) imageOf(e)];
        final domains = {for (final d in args['domains'] as List) d as String};
        final exp = (run['expected'] as Map).cast<String, dynamic>();
        final diffs = <String>[];
        final sw = Stopwatch()..start();
        if (c['kind'] == 'single') {
          final res = rd.read(images[0], domains: domains, allowLegacyVin: args['allow_legacy'] as bool);
          final ms = sw.elapsedMilliseconds;
          diffs
            ..addAll(compareField('plate', res.plate?.toJson(), (exp['plate'] as Map?)?.cast<String, dynamic>()))
            ..addAll(compareField('vin', res.vin?.toJson(), (exp['vin'] as Map?)?.cast<String, dynamic>()));
          cmpValue(diffs, 'candidates', res.candidates, exp['candidates']);
          cmpValue(diffs, 'screen_context', res.screenContext, exp['screen_context']);
          cmpValue(diffs, 'lines', res.lines, exp['lines']);
          // ignore: avoid_print
          print('  $name run $ri: $ms ms (Dart JIT, replay engine, python recorded ${run['python_ms']} ms incl. ONNX)');
        } else {
          final res = rd.readFrames(images, domains: domains, allowLegacyVin: args['allow_legacy'] as bool);
          diffs
            ..addAll(compareField('plate', res.plate?.toJson(), (exp['plate'] as Map?)?.cast<String, dynamic>()))
            ..addAll(compareField('vin', res.vin?.toJson(), (exp['vin'] as Map?)?.cast<String, dynamic>()));
          cmpValue(diffs, 'conflicts', res.conflicts, exp['conflicts']);
          cmpValue(diffs, 'screen_context', res.screenContext, exp['screen_context']);
          final per = (exp['per_image'] as List).cast<Map<String, dynamic>>();
          if (res.perImage.length != per.length) {
            diffs.add('per_image: ${res.perImage.length} vs ${per.length}');
          } else {
            for (var i = 0; i < per.length; i++) {
              diffs
                ..addAll(compareField('img$i.plate', res.perImage[i].plate?.toJson(), (per[i]['plate'] as Map?)?.cast<String, dynamic>()))
                ..addAll(compareField('img$i.vin', res.perImage[i].vin?.toJson(), (per[i]['vin'] as Map?)?.cast<String, dynamic>()));
              cmpValue(diffs, 'img$i.candidates', res.perImage[i].candidates, per[i]['candidates']);
              cmpValue(diffs, 'img$i.screen_context', res.perImage[i].screenContext, per[i]['screen_context']);
            }
          }
        }
        if (engine.next != engine.events.length) diffs.add('replay: Dart made ${engine.next} engine calls, Python made ${engine.events.length}');
        expect([...engine.problems, ...diffs], isEmpty, reason: 'differences from the Python reference:\n${[...engine.problems, ...diffs].join('\n')}');
      });
    }
  }
}
