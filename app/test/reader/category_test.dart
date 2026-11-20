// Vehicle category (plate_category.dart): unit tests ported from ml/tests/test_category.py (hand-made crops with
// known colours), the dealer grammar, and the scene-reader helpers. The golden comparison against the Python
// reference lives in category_golden_test.dart.
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/grammar.dart';
import 'package:vinplate_tester/reader/plate_category.dart';

typedef Rgb = (int, int, int);
const Rgb white = (248, 248, 246), black = (18, 18, 18), yellow = (255, 206, 22), green = (22, 150, 72), blue = (0, 72, 172), red = (196, 28, 36);

/// A plate-like crop: [bg] everywhere, ink bars (about 25 % of the width) over the middle 60 % of the rows.
RgbImage stripPlate(Rgb bg, Rgb ink, {int h = 48, int w = 240, int bar = 4, int period = 16}) {
  final data = Uint8List(h * w * 3);
  for (var i = 0; i < h * w; i++) {
    data[i * 3] = bg.$1;
    data[i * 3 + 1] = bg.$2;
    data[i * 3 + 2] = bg.$3;
  }
  final y0 = (0.2 * h).toInt(), y1 = (0.8 * h).toInt();
  for (var x = 0; x < w; x += period) {
    for (var y = y0; y < y1; y++) {
      for (var xx = x; xx < x + bar && xx < w; xx++) {
        final i = (y * w + xx) * 3;
        data[i] = ink.$1;
        data[i + 1] = ink.$2;
        data[i + 2] = ink.$3;
      }
    }
  }
  return RgbImage(w, h, data);
}

RgbImage flat(int h, int w, int v) => RgbImage(w, h, Uint8List(h * w * 3)..fillRange(0, h * w * 3, v));

Map<String, dynamic>? plateParsed(String t) => parsePlate(t);

const resultKeys = {'key', 'label', 'confidence', 'bg', 'ink', 'reasons', 'alternatives', 'ev', 'bh'};

void closeRgb(List<double> got, Rgb want, {double atol = 1}) {
  expect((got[0] - want.$1).abs() <= atol && (got[1] - want.$2).abs() <= atol && (got[2] - want.$3).abs() <= atol, isTrue, reason: '$got vs $want');
}

List<String> altKeys(Map<String, dynamic> r) => [for (final a in r['alternatives'] as List) (a as Map)['key'] as String];

void main() {
  group('constants', () {
    test('labels are the exact strings of the spec', () {
      expect(categoryLabels, {
        'private': 'Private vehicle',
        'commercial': 'Commercial / transport vehicle',
        'rental': 'Rental / self-drive vehicle',
        'ev_private': 'Electric vehicle (private)',
        'ev_commercial': 'Electric vehicle (commercial)',
        'temporary': 'Temporary registration',
        'trade': 'Trade certificate / dealer vehicle',
        'diplomatic': 'Diplomatic / consular vehicle',
        'military': 'Military / defence vehicle',
        'unknown': 'Type not determined',
      });
    });

    test('rule tables only name known categories and colours', () {
      const colours = {'black', 'white', 'grey', 'red', 'yellow', 'green', 'blue', 'other'};
      for (final (bg, ink, key, factor) in colourRules) {
        expect(colours.contains(bg) && (ink == null || colours.contains(ink)) && categoryLabels.containsKey(key) && factor > 0 && factor <= 1, isTrue, reason: '$bg/$ink/$key');
      }
      layoutRules.forEach((layout, v) {
        expect(categoryLabels.containsKey(v.$1), isTrue, reason: layout);
        expect(colours.containsAll(v.$2), isTrue, reason: layout);
      });
      categoryAlternatives.forEach((key, alts) {
        expect(categoryLabels.containsKey(key) && alts.length <= 2 && !alts.contains(key) && alts.every(categoryLabels.containsKey), isTrue, reason: key);
      });
    });

    test('the colour rule table has the first-match order of the reference', () {
      expect([for (final r in colourRules) '${r.$1}/${r.$2}/${r.$3}/${r.$4}'], [
        'blue/null/diplomatic/0.7',
        'green/yellow/ev_commercial/1.0',
        'green/null/ev_private/1.0',
        'yellow/red/temporary/1.0',
        'yellow/null/commercial/1.0',
        'black/yellow/rental/1.0',
        'grey/yellow/rental/1.0',
        'red/null/trade/1.0',
        'white/null/private/1.0',
      ]);
      expect(altBelow, 0.75);
    });
  });

  group('colour naming', () {
    const cases = <(Rgb, String)>[
      ((18, 18, 18), 'black'), ((0, 0, 0), 'black'), ((60, 60, 70), 'black'),
      ((248, 248, 246), 'white'), ((255, 255, 255), 'white'), ((200, 200, 200), 'white'),
      ((128, 128, 128), 'grey'), ((110, 112, 115), 'grey'),
      ((196, 28, 36), 'red'), ((255, 0, 10), 'red'), ((255, 10, 0), 'red'), ((230, 60, 60), 'red'),
      ((255, 206, 22), 'yellow'), ((240, 190, 40), 'yellow'), ((200, 160, 20), 'yellow'),
      ((22, 150, 72), 'green'), ((40, 170, 90), 'green'),
      ((0, 72, 172), 'blue'), ((24, 54, 150), 'blue'), ((70, 120, 230), 'blue'), ((0, 120, 220), 'blue'),
      // the REAL Indian EV plate is a teal (hue 174-175 clean, 184-190 washed out through a monitor): green, never blue
      ((24, 181, 166), 'green'), ((30, 162, 149), 'green'), ((132, 189, 193), 'green'), ((151, 199, 208), 'green'), ((0, 166, 140), 'green'),
      ((255, 244, 214), 'white'), // a pale cream is a white plate under warm light, not yellow
      ((230, 30, 200), 'other'), // magenta
    ];
    for (final (rgb, name) in cases) {
      test('name_colour $rgb is $name', () {
        expect(nameColour(rgb.$1.toDouble(), rgb.$2.toDouble(), rgb.$3.toDouble()).$1, name);
      });
    }

    test('exact value / saturation boundaries', () {
      expect(nameColour(71, 0, 0).$1, 'black'); // v = 0.278 < 0.28: black whatever the hue
      expect(nameColour(0, 80, 0).$1, 'green'); // v = 0.3137, s = 1: a dark but saturated green keeps its hue
      expect(nameColour(72, 72, 72).$1, 'black'); // v = 0.282, s = 0: below the 0.40 grey line
      expect(nameColour(101, 101, 101).$1, 'black'); // v = 0.396
      expect(nameColour(102, 102, 102).$1, 'grey'); // v = 0.40 exactly
      expect(nameColour(157, 157, 157).$1, 'grey'); // v = 0.6157
      expect(nameColour(159, 159, 159).$1, 'white'); // v = 0.6235
      expect(nameColour(240, 230, 200).$1, 'white'); // s = 0.167 < 0.22: a pale cream is white
      expect(nameColour(250, 230, 185).$1, 'white'); // s = 0.26, v = 0.98: still the cream band (s < 0.30)
      expect(nameColour(250, 225, 150).$1, 'yellow'); // s = 0.40: yellow
    });

    test('confidence is half at a boundary and full far from one', () {
      final far = nameColour(255, 255, 255).$2;
      expect(far == 1.0 || far > 0.95, isTrue);
      expect(nameColour(18, 18, 18).$2, greaterThan(0.95));
      final edge = nameColour(72, 71, 72).$2; // v = 0.282, one hair above the black boundary
      expect(edge >= 0.5 && edge < 0.52, isTrue, reason: '$edge');
      for (final rgb in <Rgb>[(200, 120, 40), (90, 90, 90), (250, 250, 100), (17, 71, 171)]) {
        final c = nameColour(rgb.$1.toDouble(), rgb.$2.toDouble(), rgb.$3.toDouble()).$2;
        expect(c >= 0.5 && c <= 1.0, isTrue, reason: '$rgb $c');
      }
    });

    test('hue wraps at red and ties resolve r, then g, then b', () {
      expect(nameColour(255, 0, 40).$1, 'red');
      expect(nameColour(255, 40, 0).$1, 'red');
      expect(nameColour(200, 200, 40).$1, 'yellow'); // r == g maximum: 60 degrees
      expect(nameColour(40, 200, 200).$1, 'green'); // g == b maximum: 180 degrees: cyan / teal is on the GREEN side (a real EV plate is teal)
    });
  });

  group('2-means', () {
    test('the background is the bigger cluster', () {
      var t = twoMeans(stripPlate(yellow, black));
      closeRgb(t.bg, yellow);
      closeRgb(t.ink!, black);
      expect(t.dist, greaterThan(300));
      t = twoMeans(stripPlate(black, yellow)); // a dark plate: bg is still the bigger cluster
      closeRgb(t.bg, black);
      closeRgb(t.ink!, yellow);
    });

    test('one colour has no ink; two colours under 40 apart are one colour; deterministic', () {
      final f = twoMeans(flat(30, 90, 200));
      expect(f.ink, isNull);
      expect(f.dist, 0.0);
      closeRgb(f.bg, (200, 200, 200), atol: 1e-9);
      expect(twoMeans(stripPlate((200, 200, 200), (215, 205, 200))).ink, isNull); // 22 apart
      final a = twoMeans(stripPlate(green, white)), b = twoMeans(stripPlate(green, white));
      expect(a.bg, b.bg);
      expect(a.ink, b.ink);
      expect(a.dist, b.dist);
    });

    test('a huge crop is sampled on a strided grid and still gives the right answer', () {
      final big = stripPlate(white, black, h: 480, w: 2400, bar: 40, period: 160); // step = 16
      final t = twoMeans(big);
      closeRgb(t.bg, white);
      closeRgb(t.ink!, black);
    });
  });

  group('result shape + colour rules', () {
    test('shape is exact and JSON serialisable', () {
      final r = classifyPlate(stripPlate(yellow, black), plateParsed('MH12AB1234'));
      expect(r.keys.toSet(), resultKeys);
      expect(r['key'], 'commercial');
      expect(r['label'], categoryLabels['commercial']);
      expect(r['ev'], false);
      expect(r['bh'], false);
      final c = r['confidence'] as double;
      expect(c >= 0 && c <= 1, isTrue);
      expect(c, double.parse(c.toStringAsFixed(3)));
      expect((r['bg'] as Map).keys.toSet(), {'name', 'rgb'});
      expect((r['bg'] as Map)['name'], 'yellow');
      expect((r['bg'] as Map)['rgb'], [255, 206, 22]);
      expect(((r['ink'] as Map)['rgb'] as List).every((v) => v is int), isTrue);
      expect((r['ink'] as Map)['name'], 'black');
      expect((r['reasons'] as List).take(2).toList(), ['plate background looks yellow', 'text looks black']);
      jsonEncode(r);
    });

    const rules = <(Rgb, Rgb, String)>[
      (white, black, 'private'), (yellow, black, 'commercial'), (black, yellow, 'rental'),
      (green, white, 'ev_private'), (green, yellow, 'ev_commercial'), (yellow, red, 'temporary'),
      (red, white, 'trade'), (blue, white, 'diplomatic'),
    ];
    for (final (bg, ink, key) in rules) {
      test('colour rule $bg / $ink -> $key', () {
        final text = key == 'diplomatic' ? '123CD45' : 'MH12AB1234'; // a diplomatic plate carries a CD / CC / UN code
        final r = classifyPlate(stripPlate(bg, ink), plateParsed(text));
        expect(r['key'], key);
        expect(r['label'], categoryLabels[key]);
        expect(r['ev'], key.startsWith('ev_'));
      });
    }

    test('a blue plate with an ordinary registration is never called diplomatic (text-pattern guard)', () {
      // a real photo of a green-plate EV was called diplomatic because its teal colour was named blue
      for (final text in ['MH12AB1234', 'KA01MN4120', '22BH1234AA']) {
        final r = classifyPlate(stripPlate(blue, white), plateParsed(text));
        expect(r['key'], 'unknown', reason: text);
        expect(r['reasons'], contains('diplomatic plates carry a CD / CC / UN code; this text is an ordinary registration, so the blue colour is probably a colour cast'));
        expect(altKeys(r), ['ev_private', 'private']);
        expect(r['ev'], isFalse);
        expect(r['confidence'] as double, lessThan(0.5));
      }
      final y = classifyPlate(stripPlate(blue, (255, 214, 40)), plateParsed('MH12AB1234'));
      expect(y['key'], 'unknown');
      expect(altKeys(y), ['ev_commercial', 'ev_private']);
      final bh = classifyPlate(stripPlate(blue, white), plateParsed('22BH1234AA'));
      expect(bh['bh'], isTrue);
      expect((bh['reasons'] as List)[2], 'format NN BH NNNN AA = Bharat series');
    });

    test('blue stays diplomatic when the text is a diplomatic code or unreadable', () {
      final dip = classifyPlate(stripPlate(blue, white), plateParsed('123CD45'));
      expect(dip['key'], 'diplomatic');
      expect(dip['reasons'], isNot(contains('blue plates with ordinary text are unusual')));
      final none = classifyPlate(stripPlate(blue, white), null); // no text to contradict the colour
      expect(none['key'], 'diplomatic');
      expect(none['reasons'], contains('blue plates with ordinary text are unusual'));
      expect(none['confidence'] as double, lessThan(dip['confidence'] as double));
    });

    // measured on the user's photo of a green-plate Toyota Camry (KA01MN4120): clean, and photographed off a monitor
    const realEv = <(Rgb, Rgb)>[
      ((24, 181, 166), (246, 246, 246)), ((30, 162, 149), (246, 246, 246)), ((132, 189, 193), (250, 250, 250)), ((151, 199, 208), (240, 240, 240)),
    ];
    for (final (bg, ink) in realEv) {
      test('the real teal EV plate $bg is an EV, not a diplomatic plate', () {
        for (final text in <String?>['KA01MN4120', null]) {
          final r = classifyPlate(stripPlate(bg, ink), text == null ? null : plateParsed(text));
          expect(r['key'], 'ev_private', reason: '$text');
          expect(r['ev'], isTrue);
          expect((r['bg'] as Map)['name'], 'green');
          if (text != null) expect(altKeys(r), isNot(contains('diplomatic')));
        }
      });
    }

    test('an ordinary registration never gets diplomatic as an alternative', () {
      const crops = <(Rgb, Rgb)>[(green, white), (black, yellow), ...realEv, ((90, 160, 110), (240, 240, 240)), ((40, 90, 40), (250, 250, 250))];
      for (final (bg, ink) in crops) {
        final r = classifyPlate(stripPlate(bg, ink), plateParsed('KA01MN4120'));
        expect(altKeys(r), isNot(contains('diplomatic')), reason: '$bg $ink');
        expect(r['key'], isNot('diplomatic'), reason: '$bg $ink');
      }
    });

    test('unmatched colour pairs are unknown, with the dark alternatives', () {
      var r = classifyPlate(stripPlate(black, white), plateParsed('MH12AB1234'));
      expect(r['key'], 'unknown');
      expect(r['label'], categoryLabels['unknown']);
      expect(r['reasons'], contains('no rule matches this colour pair'));
      expect(altKeys(r), ['military', 'rental']);
      expect(r['ev'], false);
      r = classifyPlate(stripPlate((140, 60, 160), white), null); // a purple plate
      expect(r['key'], 'unknown');
      expect(altKeys(r), ['private', 'commercial']);
    });

    test('a one-colour crop is named but barely trusted', () {
      final r = classifyPlate(flat(30, 90, 250), null);
      expect(r['ink'], isNull);
      expect(r['reasons'], contains('text colour could not be separated from the background'));
      expect(r['confidence'] as double, lessThan(0.34));
      expect(r['alternatives'], isNotEmpty);
    });

    test('an unusable crop is unknown, not an exception', () {
      for (final crop in <RgbImage?>[null, RgbImage(0, 10, Uint8List(0)), RgbImage(10, 0, Uint8List(0))]) {
        final r = classifyPlate(crop, null);
        expect(r.keys.toSet(), resultKeys);
        expect(r['key'], 'unknown');
        expect(r['confidence'], 0.0);
        expect(r['bg'], isNull);
        expect(r['ink'], isNull);
        expect(r['reasons'], ['no rule matches this colour pair']);
      }
    });

    test('a one-pixel crop does not crash', () {
      expect(classifyPlate(flat(1, 1, 0), null)['key'], 'unknown');
    });
  });

  group('layout vs colour', () {
    double baseConf(RgbImage c) => classifyPlate(c, null)['confidence'] as double;

    test('the text layout wins over colour and agreement adds a tenth', () {
      final r = classifyPlate(stripPlate(blue, white), plateParsed('123CD45'));
      expect(r['key'], 'diplomatic');
      expect(r['reasons'], containsAll(['format NNN CD NN = diplomatic mission code', 'colour agrees with the plate format']));
      final base = baseConf(stripPlate(blue, white)) / 0.7; // undo the blue rule's x0.7
      expect((r['confidence'] as double), closeTo(base + 0.1 > 1 ? 1.0 : base + 0.1, 0.003));
    });

    test('a disagreeing colour multiplies the confidence by 0.6 and says so', () {
      final w = stripPlate(white, black);
      final r = classifyPlate(w, plateParsed('123CD45'));
      expect(r['key'], 'diplomatic'); // the text pattern decides
      expect(r['reasons'], contains('plate colour (white) does not match the diplomatic format'));
      expect(r['reasons'], isNot(contains('colour agrees with the plate format')));
      expect(r['confidence'] as double, closeTo(baseConf(w) * 0.6, 0.003));
      expect(altKeys(r).first, 'private'); // what the colour alone says comes first
    });

    test('military and dealer layouts', () {
      final mil = plateParsed('12A123456B')!;
      expect(mil['layout'], 'military');
      expect(classifyPlate(stripPlate((26, 26, 30), white), mil)['key'], 'military');
      expect(classifyPlate(stripPlate(white, black), mil)['key'], 'military');
      final dealer = plateParsed('DLR012345');
      expect(classifyPlate(stripPlate(red, white), dealer)['key'], 'trade');
      final r = classifyPlate(stripPlate(white, black), dealer);
      expect(r['key'], 'trade');
      expect(r['reasons'], contains('plate colour (white) does not match the dealer format'));
      expect(r['ev'], false);
      expect(r['bh'], false);
    });

    test('Bharat series is a flag, not a category', () {
      final bh = plateParsed('22BH1234AA');
      final w = classifyPlate(stripPlate(white, black), bh);
      expect(w['key'], 'private');
      expect(w['bh'], true);
      expect(w['reasons'], containsAll(['format NN BH NNNN AA = Bharat series', 'colour agrees with the plate format']));
      final plain = classifyPlate(stripPlate(white, black), plateParsed('MH12AB1234'));
      expect(w['confidence'] as double, greaterThanOrEqualTo(plain['confidence'] as double));
      expect(plain['bh'], false);
      final y = classifyPlate(stripPlate(yellow, black), bh); // a BH plate that is not white: still a flag, less sure
      expect(y['key'], 'commercial');
      expect(y['bh'], true);
      expect(y['reasons'], contains('Bharat-series plates are white with black text'));
      expect(y['confidence'] as double, lessThan(classifyPlate(stripPlate(yellow, black), null)['confidence'] as double));
      final unk = classifyPlate(stripPlate(black, white), bh);
      expect(unk['key'], 'unknown');
      expect(unk['bh'], true);
      expect((unk['reasons'] as List)[2], 'format NN BH NNNN AA = Bharat series'); // inserted after the two colour reasons
    });
  });

  group('confidence + alternatives', () {
    test('alternatives appear only when the confidence is low and never repeat the answer', () {
      final clean = classifyPlate(stripPlate(yellow, black), plateParsed('MH12AB1234'));
      expect(clean['confidence'] as double, greaterThanOrEqualTo(altBelow));
      expect(clean['alternatives'], isEmpty);
      final dim = classifyPlate(stripPlate((230, 100, 40), (10, 10, 10)), plateParsed('MH12AB1234')); // orange-ish
      expect(dim['key'], 'commercial');
      expect(dim['confidence'] as double, lessThan(altBelow));
      expect(altKeys(dim), categoryAlternatives['commercial']!.take(2).toList());
      for (final a in dim['alternatives'] as List) {
        expect((a as Map).keys.toSet(), {'key', 'label'});
        expect(a['label'], categoryLabels[a['key']]);
        expect(a['key'], isNot(dim['key']));
      }
    });

    test('confidence falls with the separation between text and background', () {
      final strong = classifyPlate(stripPlate(white, black), null)['confidence'] as double;
      final weak = classifyPlate(stripPlate(white, (150, 150, 150)), null)['confidence'] as double; // grey text on white
      expect(weak, lessThan(strong));
    });

    test('classification is deterministic', () {
      final crop = stripPlate(green, yellow);
      expect(jsonEncode(classifyPlate(crop, null)), jsonEncode(classifyPlate(stripPlate(green, yellow), null)));
    });
  });

  group('dealer grammar (DLR NN NNNN)', () {
    test('parses with the reference segmentation (longest first digit group first)', () {
      expect(plateParsed('DLR012345'), {'layout': 'dealer', 'plate': 'DLR012345', 'log_prior': -2.0, 'series': 'DLR', 'rto': '01', 'number': '2345', 'display': 'DLR 01 2345'});
      expect(plateParsed('DLR12345')!['rto'], '12');
      expect(plateParsed('DLR12345')!['number'], '345');
      expect(plateParsed('DLR1234')!['rto'], '1');
      expect(plateParsed('DLR1234')!['display'], 'DLR 1 234');
      expect(plateParsed('DLR01234')!['display'], 'DLR 01 234');
    });

    test('DLR is required and the digits are bounded', () {
      expect(plateParsed('DLR12'), isNull);
      expect(plateParsed('DLR123'), isNull);
      expect(plateParsed('DLR1234567'), isNull);
      expect(plateParsed('012345'), isNull);
    });

    test('an ordinary Delhi plate is still a standard plate', () {
      expect(plateParsed('DL1CAB1234')!['layout'], 'standard');
      expect(plateParsed('DL12AB1234')!['layout'], 'standard');
    });

    test('the dealer layout is accepted by the plate grammar', () {
      expect(plateGrammar.isValid('DLR012345'), isTrue);
    });
  });
}
