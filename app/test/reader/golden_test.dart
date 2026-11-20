// The Dart decoder + grammar against the Python reference (ml/vinplate), replayed on the SAME
// inputs. Fixture: test/fixtures/golden/decoder_cases.json, written by ml/scripts/make_golden.py.
// Text / status / masked / inferred / gaps / reasons / n-best texts must match EXACTLY; floats to 1e-6
// relative.

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/reader/ctc.dart';
import 'package:vinplate_tester/reader/decoder.dart';
import 'package:vinplate_tester/reader/grammar.dart';
import 'package:vinplate_tester/reader/types.dart';
import 'package:vinplate_tester/reader/vocab.dart';

late Map<String, dynamic> gold;

Grammar grammarOf(String n) => n == 'plate' ? plateGrammar : vinGrammar;
Logp lpOf(dynamic flat) => Logp.fromFlat((flat as List).cast<num>());

void closeTo(num got, num want, String what, {double tol = 1e-6}) {
  final d = (got - want).abs();
  expect(d <= tol * (want.abs() > 1 ? want.abs() : 1), isTrue, reason: '$what: got $got, want $want');
}

void expectReading(Reading r, Map<String, dynamic> e, String id) {
  expect(r.text, e['text'], reason: '$id text');
  expect(r.valid, e['valid'], reason: '$id valid');
  expect(r.status, e['status'], reason: '$id status (${r.reasons} vs ${e['reasons']})');
  expect(r.masked, e['masked'], reason: '$id masked');
  expect(r.inferred, (e['inferred'] as List).cast<int>(), reason: '$id inferred');
  expect(r.suspectedGaps, (e['suspected_gaps'] as List).cast<int>(), reason: '$id gaps');
  expect(r.greedyText, e['greedy_text'], reason: '$id greedy');
  expect(r.reasons, (e['reasons'] as List).cast<String>(), reason: '$id reasons');
  expect([for (final (t, _) in r.nBest) t], [for (final n in e['n_best'] as List) (n as List)[0]], reason: '$id n_best texts');
  closeTo(r.confidence, e['confidence'] as num, '$id confidence');
  closeTo(r.seqPosterior, e['seq_posterior'] as num, '$id seq_posterior');
  closeTo(r.grammarCost, e['grammar_cost'] as num, '$id grammar_cost');
  final cc = (e['char_conf'] as List).cast<num>();
  expect(r.charConf.length, cc.length, reason: '$id char_conf length');
  for (var i = 0; i < cc.length; i++) {
    closeTo(r.charConf[i], cc[i], '$id char_conf[$i]');
  }
  final nb = e['n_best'] as List;
  expect(r.nBest.length, nb.length);
  for (var i = 0; i < nb.length; i++) {
    closeTo(r.nBest[i].$2, (nb[i] as List)[1] as num, '$id n_best[$i] p');
  }
}

void expectMap(Object? got, Object? want, String path) {
  if (want is Map) {
    expect(got, isA<Map>(), reason: path);
    expect((got as Map).keys.toSet(), want.keys.toSet(), reason: '$path keys');
    for (final k in want.keys) {
      expectMap(got[k], want[k], '$path.$k');
    }
  } else if (want is List) {
    expect(got, isA<List>(), reason: path);
    expect((got as List).length, want.length, reason: '$path length');
    for (var i = 0; i < want.length; i++) {
      expectMap(got[i], want[i], '$path[$i]');
    }
  } else if (want is num && got is num) {
    closeTo(got, want, path);
  } else {
    expect(got, want, reason: path);
  }
}

void main() {
  setUpAll(() {
    gold = (jsonDecode(File('test/fixtures/golden/decoder_cases.json').readAsStringSync()) as Map).cast<String, dynamic>();
    vinNowYearOverride = gold['year_now'] as int;
  });

  test('fixture is present and large enough to mean something', () {
    expect((gold['read_line'] as List).length, greaterThanOrEqualTo(60));
    expect((gold['fuse'] as List).length, greaterThanOrEqualTo(15));
    expect((gold['grammar'] as List).length, greaterThanOrEqualTo(100));
  });

  group('read_line (simulated + real posteriors)', () {
    test('every case matches the Python reference', () {
      var n = 0;
      final failures = <String>[];
      for (final c in (gold['read_line'] as List).cast<Map<String, dynamic>>()) {
        final id = c['id'] as String;
        try {
          final r = readLine(lpOf(c['logp']), grammarOf(c['grammar'] as String));
          expectReading(r, (c['expected'] as Map).cast<String, dynamic>(), id);
          n++;
        } on TestFailure catch (e) {
          failures.add('$id: ${e.message}');
        }
      }
      // ignore: avoid_print
      print('read_line golden: $n/${(gold['read_line'] as List).length} identical');
      expect(failures, isEmpty, reason: failures.take(8).join('\n'));
    });

    test('sub-functions: greedy, forward likelihood, alignment, alternatives, gaps, beam', () {
      final failures = <String>[];
      var checked = 0;
      for (final c in (gold['read_line'] as List).cast<Map<String, dynamic>>()) {
        final aux = c['aux'] as Map<String, dynamic>?;
        if (aux == null) continue;
        final id = c['id'] as String;
        final lp = lpOf(c['logp']);
        final text = (c['expected'] as Map)['text'] as String;
        try {
          final (gt, gconf) = greedy(lp);
          final wg = aux['greedy'] as List;
          expect(gt, wg[0], reason: '$id greedy text');
          expectMap(gconf, wg[1], '$id greedy conf');
          final labs = encodeText(text);
          closeTo(ctcLoglik(lp, labs), aux['ctc_loglik_text'] as num, '$id loglik(text)');
          if (aux['ctc_loglik_greedy'] != null) closeTo(ctcLoglik(lp, encodeText(gt)), aux['ctc_loglik_greedy'] as num, '$id loglik(greedy)');
          final spans = ctcAlign(lp, labs);
          expectMap([for (final (a, b, p) in spans) [a, b, p]], aux['align'], '$id align');
          expectMap(suspectedGaps(spans, lp.t), aux['gaps'], '$id gaps');
          expectMap([for (final alt in alternatives(lp, text, k: 3)) [for (final (ch, p) in alt) [ch, p]]], aux['alternatives'], '$id alternatives');
          final beam = prefixBeamSearch(lp, grammarOf(c['grammar'] as String), beam: 32).take(4).toList();
          expectMap([for (final h in beam) [h.text, h.ctcLogp, h.prior, h.score]], aux['beam'], '$id beam');
          checked++;
        } on TestFailure catch (e) {
          failures.add('$id: ${e.message}');
        }
      }
      // ignore: avoid_print
      print('sub-function golden: $checked cases identical, ${failures.length} differ');
      expect(failures, isEmpty, reason: failures.take(8).join('\n'));
    });
  });

  group('fuse_frames', () {
    test('every case matches (incl. ragged lengths and weights)', () {
      var n = 0;
      final failures = <String>[];
      for (final c in (gold['fuse'] as List).cast<Map<String, dynamic>>()) {
        final id = c['id'] as String;
        try {
          final lps = [for (final f in c['logps'] as List) lpOf(f)];
          final w = (c['weights'] as List?)?.cast<num>().map((x) => x.toDouble()).toList();
          final r = fuseFrames(lps, grammarOf(c['grammar'] as String), weights: w);
          expectReading(r, (c['expected'] as Map).cast<String, dynamic>(), id);
          final fz = c['fused'] as Map<String, dynamic>?;
          if (fz != null) {
            final f = entropyFuse(lps, weights: w);
            expect(f.t, fz['t'], reason: '$id fused T');
            closeTo(f.data.fold<double>(0, (s, x) => s + x), fz['sum'] as num, '$id fused sum', tol: 1e-6);
            expectMap([for (var v = 0; v < kVocab; v++) f.at(0, v)], fz['row0'], '$id fused row0');
            expectMap([for (var v = 0; v < kVocab; v++) f.at(f.t ~/ 2, v)], fz['row_mid'], '$id fused row_mid');
          }
          n++;
        } on TestFailure catch (e) {
          failures.add('$id: ${e.message}');
        }
      }
      // ignore: avoid_print
      print('fuse golden: $n/${(gold['fuse'] as List).length} identical');
      expect(failures, isEmpty, reason: failures.take(8).join('\n'));
    });
  });

  test('Logp.concat = trim_blank + concatenate (two-row plate)', () {
    for (final c in (gold['concat'] as List).cast<Map<String, dynamic>>()) {
      final parts = [for (final f in c['logps'] as List) lpOf(f)];
      final j = Logp.concat(parts);
      expect(j.t, c['joined_t'], reason: c['id'] as String);
      closeTo(j.data.fold<double>(0, (s, x) => s + x), c['joined_sum'] as num, '${c['id']} sum');
      expectReading(readLine(j, plateGrammar), (c['expected'] as Map).cast<String, dynamic>(), c['id'] as String);
    }
  });

  group('grammar', () {
    test('is_valid, log_prior, parse_plate / parse_vin for every string', () {
      final failures = <String>[];
      for (final c in (gold['grammar'] as List).cast<Map<String, dynamic>>()) {
        final s = c['s'] as String;
        for (final name in ['plate', 'vin']) {
          final want = (c[name] as Map).cast<String, dynamic>();
          final g = grammarOf(name);
          try {
            expect(g.isValid(s), want['valid'], reason: '$name valid "$s"');
            final p = g.logPrior(s);
            if (want['prior'] == null) {
              expect(p, isNull, reason: '$name prior "$s"');
            } else {
              closeTo(p!, want['prior'] as num, '$name prior "$s"');
            }
            expectMap(name == 'plate' ? parsePlate(s) : parseVin(s), want['parsed'], '$name parse "$s"');
          } on TestFailure catch (e) {
            failures.add(e.message ?? '$name "$s"');
          }
        }
      }
      expect(failures, isEmpty, reason: failures.take(8).join('\n'));
    });

    test('vin_check_digit_ok', () {
      for (final c in (gold['vin_check'] as List).cast<Map<String, dynamic>>()) {
        expect(vinCheckDigitOk(c['s'] as String), c['ok'], reason: 'check digit of "${c['s']}"');
      }
    });

    test('vin_year_candidates (year pinned from the fixture)', () {
      final now = gold['year_now'] as int;
      for (final c in (gold['vin_year'] as List).cast<Map<String, dynamic>>()) {
        expect(vinYearCandidates(c['s'] as String, nowYear: now), (c['years'] as List).cast<int>(), reason: c['s'] as String);
      }
    });
  });
}
