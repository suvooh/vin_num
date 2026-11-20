// Unit tests for the pieces the golden file does not isolate: vocabulary, Logp, the NFA (incl. the
// '?' wildcard), numpy-compatible sorting, and a timing report for the decoder.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/reader/ctc.dart';
import 'package:vinplate_tester/reader/decoder.dart';
import 'package:vinplate_tester/reader/grammar.dart';
import 'package:vinplate_tester/reader/types.dart';
import 'package:vinplate_tester/reader/vocab.dart';

/// A peaky posterior for [text]: one spike frame per character, blank elsewhere (3 frames / char).
Logp spikes(String text, {int fpc = 3}) {
  final rows = <List<double>>[];
  List<double> row(int hot, double p) {
    final r = List<double>.filled(kVocab, (1 - p) / (kVocab - 1));
    r[hot] = p;
    return [for (final x in r) math.log(x)];
  }

  for (final ch in text.split('')) {
    for (var f = 0; f < fpc; f++) {
      rows.add(f == 0 ? row(charToId(ch), 0.95) : row(blankId, 0.95));
    }
    rows.add(row(blankId, 0.95));
  }
  return Logp.fromRows(rows);
}

void main() {
  group('vocab', () {
    test('ids round-trip and the unreadable class is last', () {
      expect(charToId('0'), 1);
      expect(charToId('A'), 11);
      expect(charToId('Z'), 36);
      expect(charToId('?'), 37);
      expect(decodeLabels(encodeText('MH12AB?234')), 'MH12AB?234');
      expect(decodeLabels([0, 11, 0, 12]), 'AB');
      expect(kChars.length + 1, kVocab);
    });

    test('characters outside the alphabet are refused', () {
      expect(() => charToId('a'), throwsArgumentError);
      expect(() => encodeText('MH-12'), throwsArgumentError);
    });
  });

  group('Logp', () {
    test('slice is clipped like a Python slice', () {
      final lp = spikes('AB');
      expect(lp.slice(0, 3).t, 3);
      expect(lp.slice(-5, 3).t, 3);
      expect(lp.slice(2, 999).t, lp.t - 2);
      expect(lp.slice(5, 2).t, 0);
      expect(lp.slice(1, 3).at(0, charToId('A')), lp.at(1, charToId('A')));
    });

    test('trimBlank keeps one margin frame; all-blank keeps a single frame', () {
      final lp = spikes('A');
      final padded = Logp.concat([lp]); // already trimmed
      expect(padded.t, lessThanOrEqualTo(lp.t));
      final blank = Logp.fromRows([for (var i = 0; i < 6; i++) [0.0, ...List<double>.filled(kVocab - 1, -20.0)]]);
      expect(blank.trimBlank().t, 1);
    });

    test('concat joins two rows without the blank margins that look like a gap', () {
      final a = spikes('MH12'), b = spikes('AB1234');
      final j = Logp.concat([a, b]);
      expect(j.t, lessThan(a.t + b.t));
      expect(readLine(j, plateGrammar).text, 'MH12AB1234');
    });
  });

  group('grammar NFA', () {
    test('stateCodes matches stateNames', () => expect(stateCodes, stateNames.keys.toSet()));

    test('wildcard ? stays alive while SOME completion is legal', () {
      expect(plateGrammar.isValid('M?12AB1234'), isTrue); // MH, MP, ML, MN, MZ
      expect(plateGrammar.isValid('??12AB1234'), isTrue);
      expect(plateGrammar.isValid('MH12AB123?'), isTrue);
      expect(plateGrammar.isValid('Q?12AB1234'), isFalse); // no state code starts with Q
      expect(plateGrammar.isValid('MH12AB12345'), isFalse); // 5-digit number
      var s = plateGrammar.start();
      expect(plateGrammar.step(s, 'Q'), isNull);
      final m = plateGrammar.step(s, 'M')!;
      expect(plateGrammar.step(m, '?'), isNotNull);
    });

    test('equal state sets are one object (interning), and step is cached', () {
      final s = plateGrammar.start();
      final a = plateGrammar.step(plateGrammar.step(s, 'M')!, 'H');
      final b = plateGrammar.step(plateGrammar.step(s, 'M')!, 'H');
      expect(identical(a, b), isTrue);
    });

    test('accepting needs a complete layout', () {
      expect(plateGrammar.isValid('MH'), isFalse);
      expect(plateGrammar.isValid('MH1'), isFalse); // a state code and an RTO digit, no number yet
      expect(plateGrammar.isValid('MH12'), isTrue); // MH + RTO 1 + number 2: legal, but the prior makes it very unlikely
      expect(plateGrammar.logPrior('MH12')!, lessThan(-5.0));
      expect(plateGrammar.isValid('MH121'), isTrue); // MH 12 1 (legacy short number): legal, penalised by the prior
      expect(plateGrammar.logPrior('MH121')!, lessThan(plateGrammar.logPrior('MH121234')!));
      expect(vinGrammar.isValid('MAL907KE1E6602791'), isTrue);
      expect(vinGrammar.isValid('MAL907KE1E660279'), isTrue); // 16 chars: legacy template
      expect(vinGrammar.isValid('MAL90'), isFalse);
    });

    test('delhi class letter and bharat series display', () {
      expect(parsePlate('DL1CAB1234')!['display'], 'DL 1C AB 1234');
      expect(parsePlate('22BH1234AA')!['display'], '22 BH 1234 AA');
      expect(parsePlate('MH12AB1234')!['display'], 'MH 12 AB 1234');
      expect(parsePlate('M?12AB1234')!['state_name'], isNull);
      expect(parsePlate('NOTAPLATE'), isNull);
    });

    test('vin check digit and year code', () {
      expect(vinCheckDigitOk('1M8GDM9AXKP042788'), isTrue);
      expect(vinCheckDigitOk('1M8GDM9A1KP042788'), isFalse);
      expect(vinYearCandidates('MAL907KE1E6602791', nowYear: 2026), [1984, 2014]); // E: 1984 or 2014, both plausible
      expect(vinYearCandidates('MAL907KE1A6602791', nowYear: 2026), [2010]); // A: 1980 is before the standard
      expect(vinYearCandidates('MAL907KE156602791', nowYear: 2026), [2005]); // a digit code: 2001-2009 only
      expect(vinYearCandidates('MAL907KE1Y6602791', nowYear: 2026), [2000]); // Y: 2000 (2030 is in the future)
      expect(vinYearCandidates('MAL907KE1I6602791', nowYear: 2026), isEmpty); // I is not a year code
    });
  });

  group('sorting helpers', () {
    test('stableSortedBy keeps insertion order on ties (Python sorted)', () {
      final out = stableSortedBy(['a', 'b', 'c', 'd', 'e'], (s) => s == 'c' ? 0.0 : 1.0);
      expect(out, ['c', 'a', 'b', 'd', 'e']);
    });

    test('argsortNumpy reproduces numpy.argsort tie order (vectors captured from numpy 2.5)', () {
      expect(argsortNumpy(List<double>.filled(37, 5.0)),
          [0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 19, 35, 18, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 36]);
      final v = List<double>.filled(37, 5.0);
      v[32] = 9;
      v[36] = 9;
      expect(argsortNumpy(v), [0, 20, 21, 22, 23, 24, 25, 19, 26, 28, 29, 30, 31, 33, 34, 27, 35, 18, 16, 1, 2, 3, 4, 5, 6, 17, 7, 9, 10, 11, 12, 13, 14, 15, 8, 32, 36]);
      final w = List<double>.filled(37, 5.0);
      w[0] = 9;
      w[36] = 9;
      expect(argsortNumpy(w), [18, 19, 20, 21, 22, 23, 24, 35, 25, 27, 28, 29, 30, 31, 32, 26, 33, 17, 15, 1, 2, 3, 4, 5, 6, 16, 7, 9, 10, 11, 12, 13, 14, 8, 34, 0, 36]);
      expect(argsortNumpy(List<double>.filled(16, 5.0)), List<int>.generate(16, (i) => i)); // <= 16 elements: insertion sort, stable
      expect(argsortNumpy(List<double>.filled(17, 5.0)).take(5), [0, 14, 13, 12, 11]); // 17 elements: first partition
    });

    test('argsortNumpy sorts correctly in general', () {
      final r = math.Random(4);
      for (final n in [1, 2, 5, 16, 17, 37, 100]) {
        final v = [for (var i = 0; i < n; i++) r.nextDouble()];
        final idx = argsortNumpy(v);
        for (var i = 1; i < n; i++) {
          expect(v[idx[i - 1]] <= v[idx[i]], isTrue);
        }
        expect(idx.toSet().length, n);
      }
    });

    test('pairwiseSum equals a plain sum within rounding', () {
      final r = math.Random(1);
      for (final n in [1, 7, 8, 37, 129, 1000]) {
        final x = [for (var i = 0; i < n; i++) r.nextDouble() - 0.5];
        expect((pairwiseSum(x, 0, n, 1) - x.fold<double>(0, (a, b) => a + b)).abs(), lessThan(1e-9));
      }
    });
  });

  group('ctc primitives', () {
    test('logSoftmax rows sum to 1 and respect temperature', () {
      final logits = Float64List.fromList([for (var i = 0; i < 2 * kVocab; i++) (i % 7).toDouble()]);
      final a = logSoftmax(logits, 2);
      final b = logSoftmax(logits, 2, temperature: 2.0);
      for (final lp in [a, b]) {
        for (var t = 0; t < 2; t++) {
          final s = [for (var v = 0; v < kVocab; v++) math.exp(lp.at(t, v))].fold<double>(0, (x, y) => x + y);
          expect(s, closeTo(1.0, 1e-12));
        }
      }
      expect(math.exp(b.at(0, 6)), lessThan(math.exp(a.at(0, 6)))); // flatter
    });

    test('greedy / loglik / align agree on a clean read', () {
      final lp = spikes('MH12AB1234');
      expect(greedy(lp).$1, 'MH12AB1234');
      final labs = encodeText('MH12AB1234');
      expect(ctcLoglik(lp, labs), greaterThan(ctcLoglik(lp, encodeText('MH12AB1235'))));
      final al = ctcAlign(lp, labs);
      expect(al.length, 10);
      expect([for (final a in al) a.$1], [for (var i = 0; i < 10; i++) i * 4]);
      expect(suspectedGaps(al, lp.t), isEmpty);
    });

    test('a missing character leaves a wide gap that is flagged', () {
      // 'MH12', a hole as wide as one character, then 'B1234'
      final a = spikes('MH12');
      final hole = Logp.fromRows([for (var i = 0; i < 4; i++) [math.log(0.95), ...List<double>.filled(kVocab - 1, math.log(0.05 / 37))]]);
      final b = spikes('B1234');
      final all = Float64List.fromList([...a.data, ...hole.data, ...b.data]);
      final lp = Logp(a.t + hole.t + b.t, all);
      final r = readLine(lp, plateGrammar);
      expect(r.text, 'MH12B1234');
      expect(r.suspectedGaps, [4]);
      expect(r.status, isNot('accept'));
      expect(r.masked, 'MH12?B1234');
    });

    test('a clean plate and a clean VIN are accepted; junk is rejected', () {
      final p = readLine(spikes('TN71JE5250'), plateGrammar);
      expect((p.text, p.status), ('TN71JE5250', 'accept'));
      expect(p.masked, 'TN71JE5250');
      final v = readLine(spikes('MAL907KE1E6602791'), vinGrammar);
      expect(v.text, 'MAL907KE1E6602791');
      final junk = readLine(spikes('XQZW'), plateGrammar);
      expect(junk.status, 'reject');
    });
  });

  test('decoder speed (JIT debug build: an upper bound; release/AOT on a phone is faster)', () {
    final g = (jsonDecode(File('test/fixtures/golden/decoder_cases.json').readAsStringSync()) as Map).cast<String, dynamic>();
    final cases = (g['read_line'] as List).cast<Map<String, dynamic>>();
    for (final kind in ['plate', 'vin']) {
      final c = cases.lastWhere((c) => c['grammar'] == kind && (c['id'] as String).startsWith('real'));
      final lp = Logp.fromFlat((c['logp'] as List).cast<num>());
      final gram = kind == 'plate' ? plateGrammar : vinGrammar;
      readLine(lp, gram); // warm up
      final sw = Stopwatch()..start();
      const reps = 5;
      for (var i = 0; i < reps; i++) {
        readLine(lp, gram);
      }
      // ignore: avoid_print
      print('readLine $kind: T=${lp.t}  ${(sw.elapsedMicroseconds / reps / 1000).toStringAsFixed(1)} ms/call');
      final sw2 = Stopwatch()..start();
      for (var i = 0; i < reps; i++) {
        fuseFrames([lp, lp, lp], gram);
      }
      // ignore: avoid_print
      print('fuseFrames x3 $kind: ${(sw2.elapsedMicroseconds / reps / 1000).toStringAsFixed(1)} ms/call');
    }
  });
}
