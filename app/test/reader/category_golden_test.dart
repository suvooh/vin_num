// The Dart vehicle-category classifier against the Python reference (ml/vinplate/category.py), replayed on the
// SAME crops. Fixture: test/fixtures/golden/category_cases.json, written by ml/scripts/make_golden_category.py.
// key / label / ev / bh / reasons / alternatives / colour names / rgb ints must match EXACTLY; the confidence is
// rounded to 3 decimals in both and must agree to 1e-3; the 2-means centres (float64) to 1e-9.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/grammar.dart';
import 'package:vinplate_tester/reader/plate_category.dart';

RgbImage decodePng(String b64) {
  final im = img.decodePng(base64Decode(b64))!;
  return RgbImage(im.width, im.height, Uint8List.fromList(im.convert(numChannels: 3).getBytes(order: img.ChannelOrder.rgb)));
}

void main() {
  final file = File('test/fixtures/golden/category_cases.json');
  if (!file.existsSync()) {
    test('category_cases.json is missing - run ml/scripts/make_golden_category.py', () => fail('golden file missing'));
    return;
  }
  final data = (jsonDecode(file.readAsStringSync()) as Map).cast<String, dynamic>();
  final cases = (data['cases'] as List).cast<Map<String, dynamic>>();
  final colours = (data['colours'] as List).cast<Map<String, dynamic>>();
  final meta = (data['meta'] as Map).cast<String, dynamic>();

  test('fixture is present and large enough to mean something', () {
    expect(cases.length, greaterThanOrEqualTo(60));
    expect(colours.length, greaterThanOrEqualTo(200));
    final keys = {for (final c in cases) ((c['expected'] as Map)['key'] as String)};
    expect(keys, containsAll(['private', 'commercial', 'rental', 'ev_private', 'ev_commercial', 'temporary', 'trade', 'diplomatic', 'military', 'unknown']));
  });

  test('the rule tables and reason strings are the ones the fixture was written from', () {
    expect((meta['labels'] as Map).cast<String, String>(), categoryLabels);
    expect(meta['alt_below'], altBelow);
    final r = (meta['reasons'] as Map).cast<String, dynamic>();
    final one = classifyPlate(_flat(30, 90, 250), null);
    expect(one['reasons'], [(r['bg'] as String).replaceFirst('{}', 'white'), r['one_colour']]);
    final dis = classifyPlate(_strip((248, 248, 246), (18, 18, 18)), parsePlate('123CD45'));
    expect(dis['reasons'], contains((r['disagree'] as String).replaceFirst('{}', 'white').replaceFirst('{}', 'diplomatic')));
    expect(dis['reasons'], contains((r['layout'] as Map)['diplomatic']));
    final bh = classifyPlate(_strip((248, 248, 246), (18, 18, 18)), parsePlate('22BH1234AA'));
    expect(bh['reasons'], containsAll([r['bh'], r['agree'], r['ink'].replaceFirst('{}', 'black')]));
  });

  test('name_colour: every colour (random + boundary cases) is named and scored like the reference', () {
    var nameOk = 0;
    final bad = <String>[];
    for (final c in colours) {
      final rgb = (c['rgb'] as List).cast<num>();
      final (name, conf) = nameColour(rgb[0].toDouble(), rgb[1].toDouble(), rgb[2].toDouble());
      if (name == c['name'] && (conf - (c['conf'] as num)).abs() <= 1e-9) {
        nameOk++;
      } else {
        bad.add('$rgb: got $name/$conf want ${c['name']}/${c['conf']}');
      }
    }
    // ignore: avoid_print
    print('name_colour golden: $nameOk/${colours.length} identical');
    expect(bad, isEmpty);
  });

  test('two_means: the bg / ink centres and their distance match the reference', () {
    var ok = 0;
    final bad = <String>[];
    for (final c in cases) {
      final crop = decodePng(c['png'] as String);
      expect([crop.height, crop.width], [c['h'], c['w']], reason: '${c['id']} size');
      final tm = twoMeans(crop);
      final k = (c['kmeans'] as Map).cast<String, dynamic>();
      final problems = <String>[];
      _vec('bg', tm.bg, k['bg'], problems);
      if (k['ink'] == null) {
        if (tm.ink != null) problems.add('ink: expected none, got ${tm.ink}');
      } else if (tm.ink == null) {
        problems.add('ink: expected ${k['ink']}, got none');
      } else {
        _vec('ink', tm.ink!, k['ink'], problems);
      }
      if ((tm.dist - (k['dist'] as num)).abs() > 1e-9) problems.add('dist ${tm.dist} vs ${k['dist']}');
      problems.isEmpty ? ok++ : bad.add('${c['id']}: ${problems.join('; ')}');
    }
    // ignore: avoid_print
    print('two_means golden: $ok/${cases.length} identical');
    expect(bad, isEmpty);
  });

  test('classify_plate: every crop is classified exactly like the reference', () {
    var exact = 0;
    final bad = <String>[];
    final sw = Stopwatch();
    for (final c in cases) {
      final crop = decodePng(c['png'] as String);
      final text = c['text'] as String?;
      final parsed = text == null ? null : parsePlate(text);
      // the grammar must give the layout the fixture was written with (this covers the dealer template)
      final wantLayout = c['layout'] as String?;
      if (parsed?['layout'] != wantLayout) {
        bad.add('${c['id']}: layout ${parsed?['layout']} vs $wantLayout');
        continue;
      }
      sw.start();
      final got = classifyPlate(crop, parsed);
      sw.stop();
      final diffs = _diff('', got, c['expected']);
      diffs.isEmpty ? exact++ : bad.add('${c['id']}: ${diffs.take(6).join('; ')}');
    }
    // ignore: avoid_print
    print('classify_plate golden: $exact/${cases.length} identical; '
        '${(sw.elapsedMicroseconds / cases.length / 1000).toStringAsFixed(2)} ms per classification (JIT, host CPU)');
    expect(bad, isEmpty);
  });
}

RgbImage _flat(int h, int w, int v) => RgbImage(w, h, Uint8List(h * w * 3)..fillRange(0, h * w * 3, v));

RgbImage _strip((int, int, int) bg, (int, int, int) ink, {int h = 48, int w = 240, int bar = 4, int period = 16}) {
  final d = Uint8List(h * w * 3);
  for (var i = 0; i < h * w; i++) {
    d[i * 3] = bg.$1;
    d[i * 3 + 1] = bg.$2;
    d[i * 3 + 2] = bg.$3;
  }
  for (var x = 0; x < w; x += period) {
    for (var y = (0.2 * h).toInt(); y < (0.8 * h).toInt(); y++) {
      for (var xx = x; xx < x + bar && xx < w; xx++) {
        final i = (y * w + xx) * 3;
        d[i] = ink.$1;
        d[i + 1] = ink.$2;
        d[i + 2] = ink.$3;
      }
    }
  }
  return RgbImage(w, h, d);
}

void _vec(String what, List<double> got, Object? want, List<String> problems) {
  final w = (want as List).cast<num>();
  for (var i = 0; i < 3; i++) {
    final d = (got[i] - w[i]).abs();
    if (d > 1e-9 * (w[i].abs() > 1 ? w[i].abs() : 1)) problems.add('$what[$i] ${got[i]} vs ${w[i]}');
  }
}

/// Structural diff: strings / ints / bools / nulls exactly, doubles to 1e-3 (the confidence is rounded to 3 decimals).
List<String> _diff(String path, Object? got, Object? want) {
  if (want is Map) {
    if (got is! Map) return ['$path: expected a map, got $got'];
    final out = <String>[];
    if (got.keys.toSet().difference(want.keys.toSet()).isNotEmpty || want.keys.toSet().difference(got.keys.toSet()).isNotEmpty) {
      out.add('$path keys ${got.keys.toList()} vs ${want.keys.toList()}');
    }
    for (final k in want.keys) {
      out.addAll(_diff('$path.$k', got[k], want[k]));
    }
    return out;
  }
  if (want is List) {
    if (got is! List) return ['$path: expected a list, got $got'];
    if (got.length != want.length) return ['$path length ${got.length} vs ${want.length}: $got vs $want'];
    return [for (var i = 0; i < want.length; i++) ..._diff('$path[$i]', got[i], want[i])];
  }
  if (want is double && got is double) return (got - want).abs() <= 1e-3 + 1e-12 ? const [] : ['$path: $got vs $want'];
  if (want is num && got is num && want is! double) return got == want ? const [] : ['$path: $got vs $want'];
  return got == want ? const [] : ['$path: $got vs $want'];
}
