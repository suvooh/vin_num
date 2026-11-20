/// From recogniser posteriors to a READING: constrained decode, candidate rescoring, multi-frame
/// fusion, and an honest account of how much of the answer was inferred. Port of
/// `ml/vinplate/decode/reader.py`.
///
/// Design rules (they matter for an identity / fraud workflow): never return a bare string, only
/// the evidence behind each character; a character the pixels did not support is `inferred`, a
/// character that may have been dropped is reported as a suspected gap; `grammarCost` says how far
/// the constraint bent the network's own reading; `status` is the one field to branch on.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'ctc.dart';
import 'grammar.dart';
import 'types.dart';
import 'vocab.dart';

/// nats the grammar may bend the raw reading before a human should look
const double reviewCost = 6.0;

/// beyond this the "reading" is the grammar talking, not the image
const double rejectCost = 12.0;

class Reading {
  Reading({
    required this.text,
    required this.valid,
    required this.status,
    required this.confidence,
    required this.seqPosterior,
    required this.charConf,
    required this.inferred,
    required this.suspectedGaps,
    required this.grammarCost,
    required this.greedyText,
    required this.nBest,
    required this.reasons,
  });

  final String text;
  final bool valid; // accepted by the grammar

  /// 'accept' | 'review' | 'reject'. Mutable: callers may downgrade (never upgrade) it.
  String status;

  /// min(sequence posterior, geometric-mean character evidence). Uncalibrated.
  final double confidence;
  final double seqPosterior;
  final List<double> charConf;

  /// positions whose pixel evidence was weak (mutable: callers may add ambiguous ones)
  List<int> inferred;

  /// insertion points where a character may be missing
  final List<int> suspectedGaps;
  final double grammarCost;
  final String greedyText;
  final List<(String, double)> nBest;
  List<String> reasons;
  bool calibrated = false;

  bool get needsReview => status != 'accept';

  /// `text` with every inferred character replaced by '?' and a '?' inserted at each suspected
  /// gap. THIS, not `text`, is what to display whenever status is not 'accept'.
  String get masked {
    final chars = [for (var i = 0; i < text.length; i++) inferred.contains(i) ? '?' : text[i]];
    for (final k in (List<int>.of(suspectedGaps)..sort((a, b) => b.compareTo(a)))) {
      chars.insert(math.min(k, chars.length), '?');
    }
    return chars.join();
  }

  Map<String, dynamic> toJson() => {
        'text': text,
        'valid': valid,
        'status': status,
        'confidence': confidence,
        'seq_posterior': seqPosterior,
        'char_conf': charConf,
        'inferred': inferred,
        'suspected_gaps': suspectedGaps,
        'grammar_cost': grammarCost,
        'greedy_text': greedyText,
        'n_best': [for (final (t, p) in nBest) [t, p]],
        'reasons': reasons,
        'masked': masked,
      };
}

// ------------------------------------------------------------------- candidate generation
List<int> _weakest(List<double> charConf, int maxPos, double confBelow) {
  final idx = stableSortedBy(List<int>.generate(charConf.length, (i) => i), (i) => charConf[i]).take(maxPos);
  return [for (final i in idx) if (charConf[i] < confBelow) i];
}

/// Single-character substitutions at the weakest positions, using the alternatives the network
/// itself favoured. If exactly ONE character is unreadable every legal character is tried there,
/// which is what lets a mandatory check digit pin it down.
List<String> _substitutions(Logp lp, String text, List<double> charConf, Grammar? g,
    {int maxPos = 6, int alts = 3, double confBelow = 0.9, double flatBelow = 0.25}) {
  final weak = _weakest(charConf, maxPos, confBelow);
  if (weak.isEmpty) return [];
  final out = <String>[];
  final flat = [for (final i in weak) if (charConf[i] < flatBelow) i];
  if (flat.length == 1) {
    final i = flat[0];
    for (var k = 0; k < alphabet.length; k++) {
      final cand = text.substring(0, i) + alphabet[k] + text.substring(i + 1);
      if (cand != text && (g == null || g.isValid(cand))) out.add(cand);
    }
  }
  final alt = alternatives(lp, text, k: alts);
  for (final i in weak) {
    for (final (ch, _) in alt[i]) {
      final cand = text.substring(0, i) + ch + text.substring(i + 1);
      if (!out.contains(cand) && (g == null || g.isValid(cand))) out.add(cand);
    }
  }
  return out;
}

/// If exactly one gap looks like a dropped character, try every legal character there.
List<String> _insertions(String text, List<int> gaps, Grammar? g) {
  if (gaps.length != 1) return [];
  final k = gaps[0];
  final out = <String>[];
  for (var c = 0; c < alphabet.length; c++) {
    final cand = text.substring(0, k) + alphabet[c] + text.substring(k);
    if (g == null || g.isValid(cand)) out.add(cand);
  }
  return out;
}

/// Exactly one '?': let the grammar / checksum narrow it, under a UNIFORM prior over the alphabet
/// (the pixels said nothing). A filled reading only beats the '?' reading when the prior - in
/// practice a mandatory check digit - is worth more than log(36) nats. Mutates (and re-sorts) [pool].
Hypothesis _fillUnknown(Hypothesis best, List<Hypothesis> pool, Grammar? g, double priorWeight) {
  if (g == null || '?'.allMatches(best.text).length != 1) return best;
  final i = best.text.indexOf('?');
  final uniform = -math.log(alphabet.length);
  final have = {for (final h in pool) h.text};
  for (var k = 0; k < alphabet.length; k++) {
    final cand = best.text.substring(0, i) + alphabet[k] + best.text.substring(i + 1);
    if (have.contains(cand) || !g.isValid(cand)) continue;
    pool.add(_score(cand, best.ctcLogp + uniform, g, priorWeight));
  }
  final sorted = stableSortedBy(pool, (h) => -h.score);
  pool
    ..clear()
    ..addAll(sorted);
  return pool[0];
}

Hypothesis _score(String text, double ll, Grammar? g, double priorWeight) {
  var prior = 0.0;
  if (g != null) {
    final p = g.logPrior(text);
    prior = p ?? -50.0;
  }
  return Hypothesis(text, ll, prior, true, ll + priorWeight * prior);
}

(List<double>, List<int>) _analyse(Logp lp, String text) {
  final spans = ctcAlign(lp, encodeText(text));
  return ([for (final s in spans) s.$3], suspectedGaps(spans, lp.t));
}

// ------------------------------------------------------------------------- the verdict
/// Python's `format(x, '.<d>f')`: round-half-even on an exact tie (Dart's toStringAsFixed rounds
/// half away from zero), so reason strings are identical.
String _fixed(double x, int d) {
  final scale = math.pow(10, d).toDouble();
  final scaled = x * scale;
  final fl = scaled.floorToDouble();
  if (scaled - fl == 0.5 && scaled / scale == x) {
    final even = (fl % 2 == 0) ? fl : fl + 1;
    return (even / scale).toStringAsFixed(d);
  }
  return x.toStringAsFixed(d);
}

double _round(double x, int d) {
  final m = math.pow(10, d).toDouble();
  // Python round(): half to even on the binary value; an exact tie is the only place this differs
  final scaled = x * m;
  final fl = scaled.floorToDouble();
  if (scaled - fl == 0.5) return ((fl % 2 == 0) ? fl : fl + 1) / m;
  return scaled.roundToDouble() / m;
}

Reading _finish(
  Hypothesis best,
  List<Hypothesis> pool,
  List<double> charConf,
  List<int> gaps,
  String greedyText,
  double cost, {
  required double inferBelow,
  required int maxInferred,
  required double minSeq,
  required bool valid,
}) {
  final post = posteriorOver(pool);
  final seq = pool.isEmpty ? 0.0 : post[pool.indexOf(best)];
  final inferred = <int>[
    for (var i = 0; i < charConf.length; i++)
      if (charConf[i] < inferBelow || best.text[i] == unk) i
  ];
  final ev = [for (var i = 0; i < charConf.length; i++) best.text[i] == unk ? 0.0 : charConf[i]];
  double gm = 0.0;
  if (ev.isNotEmpty) {
    final logs = [for (final e in ev) math.log(e.clamp(1e-6, 1.0))];
    gm = math.exp(pairwiseSum(logs, 0, logs.length, 1) / logs.length);
  }
  final reasons = <String>[];
  var review = false;
  var reject = false;
  if (!valid) {
    reject = true;
    reasons.add('no legal layout fits this read');
  }
  if (cost > rejectCost) {
    reject = true;
    reasons.add('grammar bent the raw reading by ${_fixed(cost, 0)} nats: output is mostly inferred');
  } else if (cost > reviewCost) {
    review = true;
    reasons.add('grammar bent the raw reading by ${_fixed(cost, 1)} nats');
  }
  if (seq < minSeq) {
    review = true;
    reasons.add('ambiguous: top reading holds only ${_fixed(seq * 100, 0)}% of the probability');
  }
  if (best.text.contains(unk)) {
    // A '?' that survived to the output is simply not read.
    review = true;
    reasons.add('${'?'.allMatches(best.text).length} character(s) unreadable and not recoverable');
  }
  if (inferred.length > maxInferred) {
    review = true;
    reasons.add('${inferred.length} characters rest on weak pixel evidence');
  }
  if (gaps.isNotEmpty) {
    review = true;
    reasons.add('possible dropped character at position(s) [${gaps.join(', ')}]');
  }
  if (best.text != greedyText && valid && !reject) {
    reasons.add("grammar/priors changed the network's raw reading");
  }
  final status = reject ? 'reject' : (review ? 'review' : 'accept');
  final ranked = stableSortedBy([for (var i = 0; i < pool.length; i++) (pool[i], post[i])], (e) => -e.$2).take(5);
  return Reading(
    text: best.text,
    valid: valid,
    status: status,
    confidence: math.min(seq, gm),
    seqPosterior: seq,
    charConf: [for (final c in charConf) _round(c, 3)],
    inferred: inferred,
    suspectedGaps: gaps,
    grammarCost: _round(cost, 2),
    greedyText: greedyText,
    nBest: [for (final (h, p) in ranked) (h.text, _round(p, 4))],
    reasons: reasons,
  );
}

Reading _noLegal(List<Logp> lps, double inferBelow, int maxInferred, double minSeq) {
  final g = [for (final lp in lps) greedy(lp).$1];
  var text = '';
  if (g.isNotEmpty) {
    // most frequent greedy text; ties -> first seen (the reference's set order is unspecified)
    var bestCount = 0;
    for (final t in g) {
      final c = g.where((x) => x == t).length;
      if (c > bestCount) {
        bestCount = c;
        text = t;
      }
    }
  }
  final h = Hypothesis(text, -1e9, 0.0, false, 0.0);
  return _finish(h, [h], List<double>.filled(text.length, 0.0), [], text, 99.0,
      inferBelow: inferBelow, maxInferred: maxInferred, minSeq: minSeq, valid: false);
}

/// Decode ONE line: grammar-constrained beam search, candidate rescoring, honest verdict.
Reading readLine(
  Logp lp,
  Grammar? g, {
  int beam = 32,
  double priorWeight = 1.0,
  double inferBelow = 0.5,
  int maxInferred = 1,
  double minSeq = 0.85,
}) {
  final gText = greedy(lp).$1;
  final hyps = prefixBeamSearch(lp, g, beam: beam, priorWeight: priorWeight);
  if (hyps.isEmpty) return _noLegal([lp], inferBelow, maxInferred, minSeq);

  final pool = <String, Hypothesis>{for (final h in hyps) h.text: h};
  void add(List<String> cands) {
    for (final c in cands) {
      if (!pool.containsKey(c)) pool[c] = _score(c, ctcLoglik(lp, encodeText(c)), g, priorWeight);
    }
  }

  for (var n = 0; n < hyps.length && n < 2; n++) {
    final h = hyps[n];
    final (cc, gaps) = _analyse(lp, h.text);
    add(_substitutions(lp, h.text, cc, g));
    if (identical(h, hyps[0])) add(_insertions(h.text, gaps, g));
  }
  final ranked = stableSortedBy(pool.values, (h) => -h.score);
  final best = _fillUnknown(ranked[0], ranked, g, priorWeight);
  final (cc, gaps) = _analyse(lp, best.text);
  return _finish(best, ranked, cc, gaps, gText, _cost(lp, gText, best),
      inferBelow: inferBelow, maxInferred: maxInferred, minSeq: minSeq, valid: true);
}

double _cost(Logp lp, String gText, Hypothesis best) {
  if (gText.isEmpty) return 0.0;
  return math.max(0.0, ctcLoglik(lp, encodeText(gText)) - best.ctcLogp);
}

// ----------------------------------------------------------------------------- multi-frame
/// Uncertainty-weighted average of time-aligned posteriors. At each step, frames that are unsure
/// (a glared character: high entropy) are down-weighted exponentially against frames that are
/// sure. Frames of different length are resampled (nearest step) to the longest.
Logp entropyFuse(List<Logp> lps, {List<double>? weights, double tau = 0.5}) {
  var T = 0;
  for (final lp in lps) {
    if (lp.t > T) T = lp.t;
  }
  final ws = weights ?? List<double>.filled(lps.length, 1.0);
  final num = Float64List(T * kVocab);
  final den = Float64List(T);
  final p = List<double>.filled(kVocab, 0.0);
  final plp = List<double>.filled(kVocab, 0.0);
  for (var f = 0; f < lps.length; f++) {
    final lp = lps[f];
    for (var t = 0; t < T; t++) {
      final src = lp.t == T ? t : math.min((t * lp.t / T).floor(), lp.t - 1);
      final off = src * kVocab;
      for (var v = 0; v < kVocab; v++) {
        final l = lp.data[off + v];
        p[v] = math.exp(l);
        plp[v] = p[v] * l;
      }
      final h = -pairwiseSum(plp, 0, kVocab, 1);
      final w = ws[f] * math.exp(-h / tau);
      for (var v = 0; v < kVocab; v++) {
        num[t * kVocab + v] += w * p[v];
      }
      den[t] += w;
    }
  }
  final out = Float64List(T * kVocab);
  for (var t = 0; t < T; t++) {
    for (var v = 0; v < kVocab; v++) {
      out[t * kVocab + v] = math.log(num[t * kVocab + v] / den[t] + 1e-12);
    }
  }
  return Logp(T, out);
}

/// Read ONE physical identifier from several frames (burst / video / photometric views). Glare and
/// smudges move, so different frames lose different characters: posterior fusion recovers a
/// character that is clear in at least one frame, and candidates from every frame (and the fused
/// posterior) are rescored under ALL frames (product of experts), each frame's penalty capped so
/// one washed-out frame cannot veto the rest.
Reading fuseFrames(
  List<Logp> lps,
  Grammar? g, {
  List<double>? weights,
  int beam = 16,
  double priorWeight = 1.0,
  double perFrameCap = 12.0,
  int maxCandidates = 24,
  double inferBelow = 0.5,
  int maxInferred = 1,
  double minSeq = 0.85,
}) {
  final w = weights ?? List<double>.filled(lps.length, 1.0);
  final fused = entropyFuse(lps, weights: w);
  final gText = greedy(fused).$1;
  final poolTexts = <String>{}; // insertion-ordered
  for (final src in [fused, ...lps]) {
    final hyps = prefixBeamSearch(src, g, beam: beam, priorWeight: priorWeight).take(4).toList();
    poolTexts.addAll(hyps.map((h) => h.text));
    if (hyps.isNotEmpty) {
      final (cc, gaps) = _analyse(src, hyps[0].text);
      poolTexts.addAll(_substitutions(src, hyps[0].text, cc, g));
      poolTexts.addAll(_insertions(hyps[0].text, gaps, g));
    }
  }
  if (poolTexts.isEmpty) return _noLegal(lps, inferBelow, maxInferred, minSeq);

  final ref = [
    for (final lp in lps)
      () {
        final t = greedy(lp).$1;
        return t.isNotEmpty ? ctcLoglik(lp, encodeText(t)) : -1e9;
      }()
  ];
  var texts = poolTexts.toList();
  if (texts.length > maxCandidates) {
    // cheap pre-rank under the fused posterior alone; only the best get the per-frame rescoring
    final pre = stableSortedBy(texts, (t) => -_score(t, ctcLoglik(fused, encodeText(t)), g, priorWeight).score);
    texts = pre.take(maxCandidates).toList();
  }
  var pool = <Hypothesis>[];
  for (final text in texts) {
    final lab = encodeText(text);
    var ll = 0.0;
    for (var f = 0; f < lps.length; f++) {
      ll += w[f] * math.max(ctcLoglik(lps[f], lab), ref[f] - perFrameCap);
    }
    pool.add(_score(text, ll, g, priorWeight));
  }
  pool = stableSortedBy(pool, (h) => -h.score);
  final best = _fillUnknown(pool[0], pool, g, priorWeight);

  // Evidence per character = strongest support in the fused posterior or any single frame.
  final perFrame = [for (final lp in lps) _analyse(lp, best.text).$1];
  final (ccF, gaps) = _analyse(fused, best.text);
  final cc = [
    for (var i = 0; i < ccF.length; i++) perFrame.fold<double>(ccF[i], (m, pf) => math.max(m, pf[i])),
  ];
  // cost: how far the constraint moved us from the fused posterior's own best reading
  final cost = gText.isNotEmpty
      ? math.max(0.0, ctcLoglik(fused, encodeText(gText)) - ctcLoglik(fused, encodeText(best.text)))
      : 0.0;
  return _finish(best, pool, cc, gaps, gText, cost,
      inferBelow: inferBelow, maxInferred: maxInferred, minSeq: minSeq, valid: true);
}
