import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/rows.dart';

Map<String, dynamic> golden() => (jsonDecode(File('test/fixtures/golden/image_rows.json').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage img(Map<String, dynamic> o) => RgbImage(o['w'] as int, o['h'] as int, base64Decode(o['rgb'] as String));

void main() {
  final g = golden();

  test('looksTwoRow follows the 2.6 aspect rule', () {
    for (final a in (g['aspect'] as List).cast<Map<String, dynamic>>()) {
      expect(looksTwoRow(a['h'] as int, a['w'] as int), a['two_row'], reason: '${a['w']}x${a['h']}');
    }
  });

  final cases = (g['cases'] as List).cast<Map<String, dynamic>>();
  for (var i = 0; i < cases.length; i++) {
    final c = cases[i];
    test('splitRows case $i (${c['rows']}-row ${(c['crop'] as Map)['w']}x${(c['crop'] as Map)['h']}) cuts where Python cuts', () {
      final crop = img(c['crop'] as Map<String, dynamic>);
      expect(looksTwoRow(crop.height, crop.width), c['looks_two_row']);
      final p = splitRows(crop);
      expect(p, hasLength(2));
      expect(p[0].height, c['top_h']);
      expect(p[1].height, c['bottom_h']);
      // the pieces are exact slices of the crop: the top from row 0, the bottom up to the last row
      final stride = crop.width * 3;
      expect(p[0].rgb, orderedEquals(crop.rgb.sublist(0, p[0].height * stride)));
      expect(p[1].rgb, orderedEquals(crop.rgb.sublist((crop.height - p[1].height) * stride)));
      expect(p[0].width, crop.width);
    });
  }

  test('a hard-edged two-row image is cut in the gap with the stated overlap', () {
    const w = 90, h = 60;
    final px = Uint8List(w * h * 3)..fillRange(0, w * h * 3, 230);
    for (final rows in [(8, 24), (36, 52)]) {
      for (var y = rows.$1; y < rows.$2; y++) {
        for (var x = 5; x < 85; x++) {
          // glyph-like texture: solid bars have no vertical edges inside them, so the quietest band
          // would be a tie between the gap and the inside of a bar (the first tie wins, as in Python)
          if ((x * 7 + y * 13) % 3 == 0) continue;
          final i = (y * w + x) * 3;
          px[i] = px[i + 1] = px[i + 2] = 20;
        }
      }
    }
    final p = splitRows(RgbImage(w, h, px));
    final m = (0.12 * h).toInt();
    final cut = p[0].height - m; // the cut row, from the top piece
    expect(cut, inInclusiveRange(24, 36), reason: 'cut must land in the white gap between the rows');
    expect(h - p[1].height, cut - m);
    expect(p[0].height, greaterThan(24)); // the top piece keeps its row whole
    expect(p[1].height, greaterThan(24)); // and so does the bottom piece
  });

  test('tiny crops do not throw', () {
    final p = splitRows(RgbImage(10, 4, Uint8List(120)));
    expect(p, hasLength(2));
  });
}
