// Tests for everything in pp_engine.dart that does not need the ONNX Runtime native library.
// PpEngine.detect / PpEngine.recProbsMany themselves (the ORT calls) can only run on a device.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/pp_engine.dart';

Map<String, dynamic> load(String name) => (jsonDecode(File('test/fixtures/golden/$name').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage img(Map<String, dynamic> o) => RgbImage(o['w'] as int, o['h'] as int, base64Decode(o['rgb'] as String));

Float32List f32(String b64) {
  final b = base64Decode(b64);
  return Float32List.view(Uint8List.fromList(b).buffer);
}

List<String> realChars() => ['', ...const LineSplitter().convert(File('assets/models/ocr_charset.txt').readAsStringSync()), ' '];

void main() {
  group('detector post-process (boxesFromProbMap) matches the Python reference', () {
    final g = load('image_geometry.json')['boxes'] as Map<String, dynamic>;
    final boxes = boxesFromProbMap(f32(g['prob'] as String), g['rw'] as int, g['rh'] as int, g['w'] as int, g['h'] as int);
    final exp = (g['boxes'] as List).cast<Map<String, dynamic>>();

    test('the same boxes survive the size / score filters, in the same order', () {
      expect(exp.length, greaterThanOrEqualTo(2));
      expect(boxes.length, exp.length);
    });

    test('quads agree to 1e-6 pixel and scores to 1e-5', () {
      for (var i = 0; i < exp.length; i++) {
        final q = [for (final v in exp[i]['quad'] as List) (v as num).toDouble()];
        for (var k = 0; k < 8; k++) {
          expect(boxes[i].quad[k], closeTo(q[k], 1e-6), reason: 'box $i coordinate $k');
        }
        expect(boxes[i].score, closeTo((exp[i]['score'] as num).toDouble(), 1e-5));
      }
    });

    test('the tilted line comes out tilted and the flat one flat', () {
      final angles = [for (final b in boxes) b.angle];
      expect(angles.any((a) => a.abs() < 1), isTrue);
      expect(angles.any((a) => (a + 9).abs() < 3), isTrue);
    });

    test('an empty probability map gives no boxes', () {
      expect(boxesFromProbMap(Float32List(64 * 32), 64, 32, 128, 64), isEmpty);
    });
  });

  group('detector pre-process', () {
    final p = load('image_geometry.json')['preprocess'] as Map<String, dynamic>;

    test('sizes: short side up to 736, long side capped at 1920, multiples of 32', () {
      for (final c in (p['det_sizes'] as List).cast<Map<String, dynamic>>()) {
        final h = c['h'] as int, w = c['w'] as int;
        if (h * w > 1000 * 1700) continue; // the large frames are covered below; keep this test fast
        final r = detPreprocess(RgbImage(w, h, Uint8List(w * h * 3)));
        expect([r.rw, r.rh], [c['rw'], c['rh']], reason: '${w}x$h');
        expect(r.rw % 32, 0);
        expect(r.rh % 32, 0);
        expect(r.data.length, 3 * r.rw * r.rh);
      }
    });

    test('a large frame is capped to 1920 on its long side', () {
      final c = (p['det_sizes'] as List).cast<Map<String, dynamic>>().firstWhere((c) => c['h'] == 2000);
      final r = detPreprocess(RgbImage(c['w'] as int, c['h'] as int, Uint8List((c['w'] as int) * (c['h'] as int) * 3)));
      expect([r.rw, r.rh], [c['rw'], c['rh']]);
      expect(math.max(r.rw, r.rh), lessThanOrEqualTo(1920));
    });

    test('tensor values (BGR, ImageNet mean/std, CHW) match the Python', () {
      final t = p['det_tensor'] as Map<String, dynamic>;
      final r = detPreprocess(img(t['image'] as Map<String, dynamic>));
      expect([r.rw, r.rh], [t['rw'], t['rh']]);
      final plane = r.rw * r.rh;
      final means = (t['mean'] as List).cast<num>();
      for (var c = 0; c < 3; c++) {
        var s = 0.0;
        for (var i = 0; i < plane; i++) {
          s += r.data[c * plane + i];
        }
        expect(s / plane, closeTo(means[c].toDouble(), 1e-3), reason: 'channel $c mean');
      }
      for (final s in (t['samples'] as List).cast<List>()) {
        expect(r.data[(s[0] as int) * plane + (s[1] as int) * r.rw + (s[2] as int)], closeTo((s[3] as num).toDouble(), 1e-3));
      }
    });
  });

  group('recogniser pre-process', () {
    final p = load('image_geometry.json')['preprocess'] as Map<String, dynamic>;

    test('width is ceil(48 w / h) clamped to [8, 1600]', () {
      for (final c in (p['rec_widths'] as List).cast<Map<String, dynamic>>()) {
        final r = recPreprocess(RgbImage(c['w'] as int, c['h'] as int, Uint8List((c['w'] as int) * (c['h'] as int) * 3)));
        expect(r.w, c['nw'], reason: '${c['w']}x${c['h']}');
        expect(r.data.length, 3 * 48 * r.w);
      }
    });

    test('pixels equal the Pillow-resized crop, in BGR order, scaled to [-1, 1]', () {
      final rc = p['rec_crop'] as Map<String, dynamic>;
      final r = recPreprocess(img(rc['image'] as Map<String, dynamic>));
      expect(r.w, rc['nw']);
      final exp = img(rc['resized'] as Map<String, dynamic>);
      final plane = 48 * r.w;
      var worst = 0.0;
      for (var i = 0; i < plane; i++) {
        for (var c = 0; c < 3; c++) {
          // channel c of the tensor is BGR: B = rgb byte 2
          final back = (r.data[c * plane + i] * 0.5 + 0.5) * 255;
          worst = math.max(worst, (back - exp.rgb[3 * i + (2 - c)]).abs());
        }
      }
      expect(worst, lessThan(1e-3));
    });
  });

  group('VocabProjection', () {
    test('real charset: 6623 lines + blank + space; only a few dozen classes are kept', () {
      final chars = realChars();
      expect(chars.length, 6625);
      final v = VocabProjection(chars);
      final golden = load('image_vocab.json');
      expect(chars.length - 2, golden['n_lines']);
      expect(v.keptClasses, golden['n_kept']);
      expect(v.keptClasses, lessThan(200));
    });

    test('matches the Python dense projection on realistic probabilities (T = 1 and T = 2)', () {
      final golden = load('image_vocab.json');
      final chars = realChars();
      final v = VocabProjection(chars);
      final probs = f32(golden['probs'] as String);
      final t = golden['t'] as int;
      for (final (key, temp) in [('logp_t1', 1.0), ('logp_t2', 2.0)]) {
        final got = v.toLogp(probs, t, temperature: temp);
        final exp = [for (final x in golden[key] as List) (x as num).toDouble()];
        expect(got.length, exp.length);
        var worst = 0.0;
        for (var i = 0; i < exp.length; i++) {
          worst = math.max(worst, (got[i] - exp[i]).abs());
        }
        expect(worst, lessThan(1e-4), reason: '$key: worst abs diff of the log-probabilities');
      }
    });

    final tiny = ['', 'A', 'b', '0', 'l', '中', '-', 'Q', ' '];
    Float32List row(Map<int, double> p) {
      final r = Float32List(tiny.length);
      p.forEach((k, v) => r[k] = v);
      return r;
    }

    test('hand-checked mapping: case folds, space is blank, CJK / punctuation are unknown, l spreads', () {
      final v = VocabProjection(tiny);
      int idx(String c) => vocabAlphabet.indexOf(c) + 1;
      // one step: 0.4 on 'b', 0.2 on blank, 0.1 on ' ', 0.1 on 'l', 0.1 on '中', 0.1 on '-'
      final lp = v.toLogp(row({2: .4, 0: .2, 8: .1, 4: .1, 5: .1, 6: .1}), 1);
      double p(int i) => math.exp(lp[i]);
      expect(p(idx('B')), closeTo(0.4, 1e-6));
      expect(p(vocabBlank), closeTo(0.3, 1e-6)); // blank + space
      expect(p(idx('L')), closeTo(0.034, 1e-6));
      expect(p(idx('1')), closeTo(0.033, 1e-6));
      expect(p(idx('I')), closeTo(0.033, 1e-6));
      expect(p(vocabUnk), closeTo(0.2, 1e-6)); // 中 and - : everything that is not kept
      var sum = 0.0;
      for (var i = 0; i < vocabSize; i++) {
        sum += math.exp(lp[i]);
      }
      expect(sum, closeTo(1.0, 1e-9));
    });

    test('temperature flattens but keeps a normalised distribution and the same winner', () {
      final v = VocabProjection(tiny);
      final probs = row({1: .9, 2: .05, 0: .05});
      final a = v.toLogp(probs, 1), b = v.toLogp(probs, 1, temperature: 3);
      var sa = 0.0, sb = 0.0;
      for (var i = 0; i < vocabSize; i++) {
        sa += math.exp(a[i]);
        sb += math.exp(b[i]);
      }
      expect(sa, closeTo(1, 1e-9));
      expect(sb, closeTo(1, 1e-9));
      final iA = vocabAlphabet.indexOf('A') + 1;
      expect(math.exp(b[iA]), lessThan(math.exp(a[iA])));
      expect(math.exp(b[iA]), greaterThan(math.exp(b[vocabAlphabet.indexOf('B') + 1]))); // still the winner
    });

    test('probabilities are never zero (log stays finite)', () {
      final lp = VocabProjection(tiny).toLogp(row({1: 1.0}), 1);
      for (final x in lp) {
        expect(x.isFinite, isTrue);
      }
    });
  });

  group('greedyDecode', () {
    test('best path: repeats collapse, blanks separate, unknown classes are skipped safely', () {
      final chars = ['', 'M', 'H', '1', ' '];
      // steps: M M blank M H H blank 1
      const seq = [1, 1, 0, 1, 2, 2, 0, 3];
      final probs = Float32List(seq.length * chars.length);
      for (var s = 0; s < seq.length; s++) {
        probs[s * chars.length + seq[s]] = 1;
      }
      expect(greedyDecode(RecOut(probs, seq.length, chars.length), chars), 'MMH1');
    });
  });

  test('vocabulary constants agree with the decoder alphabet (blank, 36 characters, unknown)', () {
    expect(vocabAlphabet.length, 36);
    expect(vocabSize, 38);
    expect(vocabUnk, vocabAlphabet.length + 1);
    expect(vocabBlank, 0);
  });
}
