/// CTC decoding: greedy, grammar-constrained prefix beam search, forced alignment, exact sequence
/// likelihood. Port of `ml/vinplate/decode/ctc.py`; all inputs are log-softmax outputs ([Logp]).
///
/// Two details make results identical to the numpy reference rather than merely similar:
///  * ties: numpy's `argsort` (introsort, not stable) decides which of several equally likely
///    classes are beam candidates / alternatives, so [argsortNumpy] ports that exact algorithm;
///    Python's stable `sorted` is mirrored by [stableSortedBy];
///  * summation order: numpy sums with a pairwise algorithm, mirrored by [pairwiseSum].
library;

import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'grammar.dart';
import 'types.dart';
import 'vocab.dart';

const double kNeg = -1e30;
const double _ln2 = 0.6931471805599453;

double _log1p(double x) {
  final u = 1.0 + x;
  if (u == 1.0) return x;
  return math.log(u) * x / (u - 1.0);
}

/// `np.logaddexp(x, y)`.
double lse(double x, double y) {
  if (x == y) return x + _ln2;
  final tmp = x - y;
  if (tmp > 0) return x + _log1p(math.exp(-tmp));
  if (tmp <= 0) return y + _log1p(math.exp(tmp));
  return tmp; // NaN
}

/// numpy's pairwise summation over `n` elements at `off`, `stride` apart.
double pairwiseSum(List<double> a, int off, int n, int stride) {
  if (n < 8) {
    var res = 0.0;
    for (var i = 0; i < n; i++) {
      res += a[off + i * stride];
    }
    return res;
  } else if (n <= 128) {
    final r = List<double>.generate(8, (j) => a[off + j * stride]);
    var i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (var j = 0; j < 8; j++) {
        r[j] += a[off + (i + j) * stride];
      }
    }
    var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) {
      res += a[off + i * stride];
    }
    return res;
  } else {
    var n2 = n ~/ 2;
    n2 -= n2 % 8;
    return pairwiseSum(a, off, n2, stride) + pairwiseSum(a, off + n2 * stride, n - n2, stride);
  }
}

/// Python's `sorted(items, key=key)`: stable, ascending.
List<T> stableSortedBy<T>(Iterable<T> items, double Function(T) key) {
  final tmp = [for (final it in items) it];
  final keys = [for (final it in tmp) key(it)];
  final idx = List<int>.generate(tmp.length, (i) => i)
    ..sort((a, b) {
      final c = keys[a].compareTo(keys[b]);
      return c != 0 ? c : a.compareTo(b);
    });
  return [for (final i in idx) tmp[i]];
}

/// `np.argsort(v)` for a float64 vector (default kind: quicksort = numpy's introsort `aquicksort_`).
/// Ported step for step because tied values (many classes share one floor probability) come out in
/// an implementation-defined order that the decoder's candidate sets depend on.
List<int> argsortNumpy(List<double> v) {
  final n = v.length;
  final a = List<int>.generate(n, (i) => i);
  if (n < 2) return a;
  const small = 15; // numpy: SMALL_QUICKSORT = 15 (verified against numpy 2.5 on 4080 arrays)
  var pl = 0, pr = n - 1;
  final stack = <int>[];
  final depthStack = <int>[];
  var msb = 0;
  for (var u = n >> 1; u > 0; u >>= 1) {
    msb++;
  }
  var cdepth = msb * 2;
  void swap(int i, int j) {
    final t = a[i];
    a[i] = a[j];
    a[j] = t;
  }

  while (true) {
    if (cdepth < 0) throw StateError('argsortNumpy: depth limit (heapsort fallback not ported)');
    while ((pr - pl) > small) {
      final pm = pl + ((pr - pl) >> 1);
      if (v[a[pm]] < v[a[pl]]) swap(pm, pl);
      if (v[a[pr]] < v[a[pm]]) swap(pr, pm);
      if (v[a[pm]] < v[a[pl]]) swap(pm, pl);
      final vp = v[a[pm]];
      var pi = pl;
      var pj = pr - 1;
      swap(pm, pj);
      while (true) {
        do {
          pi++;
        } while (v[a[pi]] < vp);
        do {
          pj--;
        } while (vp < v[a[pj]]);
        if (pi >= pj) break;
        swap(pi, pj);
      }
      final pk = pr - 1;
      swap(pi, pk);
      // push the largest partition on the stack
      if (pi - pl < pr - pi) {
        stack.add(pi + 1);
        stack.add(pr);
        pr = pi - 1;
      } else {
        stack.add(pl);
        stack.add(pi - 1);
        pl = pi + 1;
      }
      depthStack.add(--cdepth);
    }
    // insertion sort
    for (var pi = pl + 1; pi <= pr; pi++) {
      final vi = a[pi];
      final vp = v[vi];
      var pj = pi;
      var pk = pi - 1;
      while (pj > pl && vp < v[a[pk]]) {
        a[pj--] = a[pk--];
      }
      a[pj] = vi;
    }
    if (stack.isEmpty) break;
    pr = stack.removeLast();
    pl = stack.removeLast();
    cdepth = depthStack.removeLast();
  }
  return a;
}

/// `log_softmax` of raw logits [t * kVocab], per frame, with optional temperature.
Logp logSoftmax(Float64List logits, int t, {double temperature = 1.0}) {
  final out = Float64List(t * kVocab);
  for (var i = 0; i < t; i++) {
    final off = i * kVocab;
    var mx = double.negativeInfinity;
    for (var v = 0; v < kVocab; v++) {
      final x = logits[off + v] / temperature;
      out[off + v] = x;
      if (x > mx) mx = x;
    }
    var sum = 0.0;
    final tmp = List<double>.generate(kVocab, (v) => math.exp(out[off + v] - mx));
    sum = pairwiseSum(tmp, 0, kVocab, 1);
    final ls = math.log(sum);
    for (var v = 0; v < kVocab; v++) {
      out[off + v] = out[off + v] - mx - ls;
    }
  }
  return Logp(t, out);
}

/// Best-path decode: text and the probability of each emitted character.
(String, List<double>) greedy(Logp lp) {
  final out = StringBuffer();
  final conf = <double>[];
  var prev = blankId;
  for (var t = 0; t < lp.t; t++) {
    final off = t * kVocab;
    var idx = 0;
    var mx = lp.data[off];
    for (var v = 1; v < kVocab; v++) {
      if (lp.data[off + v] > mx) {
        mx = lp.data[off + v];
        idx = v;
      }
    }
    final prob = math.exp(mx);
    if (idx != blankId && idx != prev) {
      out.write(kChars[idx - 1]);
      conf.add(prob);
    } else if (idx != blankId && idx == prev && conf.isNotEmpty) {
      conf[conf.length - 1] = math.max(conf.last, prob);
    }
    prev = idx;
  }
  return (out.toString(), conf);
}

/// log P(labels | lp), summed over all CTC alignments (forward algorithm).
double ctcLoglik(Logp lp, List<int> labels) {
  final T = lp.t;
  final L = labels.length;
  if (L == 0) return pairwiseSum(lp.data, blankId, T, kVocab);
  if (T < L) return kNeg;
  final S = 2 * L + 1;
  final ext = List<int>.filled(S, blankId);
  for (var k = 0; k < L; k++) {
    ext[2 * k + 1] = labels[k];
  }
  final skip = List<bool>.filled(S, false);
  for (var s = 2; s < S; s++) {
    skip[s] = ext[s] != blankId && ext[s] != ext[s - 2];
  }
  var alpha = Float64List(S)..fillRange(0, S, kNeg);
  var next = Float64List(S);
  alpha[0] = lp.data[blankId];
  alpha[1] = lp.data[ext[1]];
  for (var t = 1; t < T; t++) {
    final off = t * kVocab;
    for (var s = 0; s < S; s++) {
      final a1 = s >= 1 ? alpha[s - 1] : kNeg;
      final a2 = (s >= 2 && skip[s]) ? alpha[s - 2] : kNeg;
      next[s] = lse(lse(alpha[s], a1), a2) + lp.data[off + ext[s]];
    }
    final tmp = alpha;
    alpha = next;
    next = tmp;
  }
  return lse(alpha[S - 1], alpha[S - 2]);
}

/// Viterbi forced alignment. Per label: (first_frame, last_frame, max_prob).
List<(int, int, double)> ctcAlign(Logp lp, List<int> labels) {
  final T = lp.t;
  final L = labels.length;
  if (L == 0 || T < L) return List.filled(L, (0, 0, 0.0));
  final S = 2 * L + 1;
  final ext = List<int>.filled(S, blankId);
  for (var k = 0; k < L; k++) {
    ext[2 * k + 1] = labels[k];
  }
  final skip = List<bool>.filled(S, false);
  for (var s = 2; s < S; s++) {
    skip[s] = ext[s] != blankId && ext[s] != ext[s - 2];
  }
  final score = Float64List(T * S)..fillRange(0, T * S, kNeg);
  final back = Int8List(T * S);
  score[0] = lp.data[blankId];
  score[1] = lp.data[ext[1]];
  for (var t = 1; t < T; t++) {
    final off = t * kVocab;
    final prev = (t - 1) * S;
    for (var s = 0; s < S; s++) {
      final s0 = score[prev + s];
      final s1 = s >= 1 ? score[prev + s - 1] : kNeg;
      final s2 = (s >= 2 && skip[s]) ? score[prev + s - 2] : kNeg;
      // argmax with the first maximum winning, like numpy
      var arg = 0;
      var best = s0;
      if (s1 > best) {
        arg = 1;
        best = s1;
      }
      if (s2 > best) {
        arg = 2;
        best = s2;
      }
      score[t * S + s] = best + lp.data[off + ext[s]];
      back[t * S + s] = arg;
    }
  }
  var s = score[(T - 1) * S + S - 1] >= score[(T - 1) * S + S - 2] ? S - 1 : S - 2;
  final path = List<int>.filled(T, 0);
  for (var t = T - 1; t >= 0; t--) {
    path[t] = s;
    s -= back[t * S + s];
  }
  final out = <(int, int, double)>[];
  for (var k = 0; k < L; k++) {
    var first = -1, last = -1;
    var mx = double.negativeInfinity;
    for (var t = 0; t < T; t++) {
      if (path[t] == 2 * k + 1) {
        if (first < 0) first = t;
        last = t;
        final p = math.exp(lp.data[t * kVocab + labels[k]]);
        if (p > mx) mx = p;
      }
    }
    out.add(first < 0 ? (0, 0, 0.0) : (first, last, mx));
  }
  return out;
}

class Hypothesis {
  Hypothesis(this.text, this.ctcLogp, [this.prior = 0.0, this.valid = true, this.score = 0.0]);
  final String text;
  final double ctcLogp;
  final double prior;
  final bool valid;
  final double score;
}

class _Beam {
  _Beam(this.pb, this.pnb, this.gs);
  double pb, pnb;
  final GState? gs;
}

/// CTC prefix beam search. With a grammar only grammar-legal prefixes are extended, and only
/// complete (accepting) strings are returned; an empty list means "nothing legal survived the beam".
/// [blankSkip]: on a step whose blank probability is at least this, nothing is emitted.
List<Hypothesis> prefixBeamSearch(
  Logp lp,
  Grammar? grammar, {
  int beam = 32,
  int topK = 12,
  double minLogp = -14.0,
  double priorWeight = 1.0,
  double blankSkip = 0.995,
}) {
  final logBlankSkip = math.log(blankSkip);
  var beams = <String, _Beam>{};
  beams[''] = _Beam(0.0, kNeg, grammar?.start());
  final neg = List<double>.filled(kVocab - 1, 0.0);
  for (var t = 0; t < lp.t; t++) {
    final off = t * kVocab;
    final blank = lp.data[off + blankId];
    final cand = <int>[];
    if (blank < logBlankSkip) {
      for (var i = 0; i < kVocab - 1; i++) {
        neg[i] = -lp.data[off + 1 + i];
      }
      final order = argsortNumpy(neg);
      for (var k = 0; k < topK && k < order.length; k++) {
        final c = order[k] + 1;
        if (lp.data[off + c] > minLogp) cand.add(c);
      }
    }
    final nxt = <String, _Beam>{};
    _Beam slot(String prefix, GState? gs) => nxt.putIfAbsent(prefix, () => _Beam(kNeg, kNeg, gs));

    for (final e in beams.entries) {
      final prefix = e.key;
      final b = e.value;
      final ptot = lse(b.pb, b.pnb);
      final s = slot(prefix, b.gs);
      s.pb = lse(s.pb, ptot + lp.data[off + blankId]);
      if (prefix.isNotEmpty) {
        s.pnb = lse(s.pnb, b.pnb + lp.data[off + charToId(prefix[prefix.length - 1])]);
      }
      for (final c in cand) {
        final ch = kChars[c - 1];
        GState? ns;
        if (grammar != null) {
          ns = grammar.step(b.gs!, ch);
          if (ns == null) continue;
        }
        final add = ((prefix.isNotEmpty && prefix[prefix.length - 1] == ch) ? b.pb : ptot) + lp.data[off + c];
        final s2 = slot(prefix + ch, ns);
        s2.pnb = lse(s2.pnb, add);
      }
    }
    final ranked = stableSortedBy(nxt.entries, (kv) => -lse(kv.value.pb, kv.value.pnb)).take(beam);
    beams = LinkedHashMap<String, _Beam>.fromEntries(ranked);
  }

  final hyps = <Hypothesis>[];
  for (final e in beams.entries) {
    final prefix = e.key;
    if (prefix.isEmpty) continue;
    if (grammar != null && !grammar.accepting(e.value.gs!)) continue;
    var prior = 0.0;
    if (grammar != null) {
      final p = grammar.logPrior(prefix);
      if (p == null) continue;
      prior = p;
    }
    final ll = lse(e.value.pb, e.value.pnb);
    hyps.add(Hypothesis(prefix, ll, prior, true, ll + priorWeight * prior));
  }
  return stableSortedBy(hyps, (h) => -h.score);
}

/// Per-character evidence: peak posterior of that label in its aligned frames.
List<double> charConfidence(Logp lp, String text) => [for (final a in ctcAlign(lp, encodeText(text))) a.$3];

/// For each character, the [k] best OTHER characters the frames in its span favoured.
List<List<(String, double)>> alternatives(Logp lp, String text, {int k = 3}) {
  final labels = encodeText(text);
  final spans = ctcAlign(lp, labels);
  final out = <List<(String, double)>>[];
  for (var n = 0; n < labels.length; n++) {
    final lab = labels[n];
    final (a, b, _) = spans[n];
    final peak = List<double>.filled(kVocab - 1, double.negativeInfinity); // classes 1..37
    for (var t = a; t <= b && t < lp.t; t++) {
      for (var v = 1; v < kVocab; v++) {
        final p = math.exp(lp.data[t * kVocab + v]);
        if (p > peak[v - 1]) peak[v - 1] = p;
      }
    }
    peak[lab - 1] = -1.0;
    peak[kChars.length - 1] = -1.0; // '?' is a statement of ignorance, not a substitute
    final order = argsortNumpy([for (final p in peak) -p]);
    out.add([for (var j = 0; j < k && j < order.length; j++) (kChars[order[j]], peak[order[j]])]);
  }
  return out;
}

double _median(List<double> xs) {
  final s = List<double>.of(xs)..sort();
  final n = s.length;
  if (n.isOdd) return s[n ~/ 2];
  return (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

/// Where a character may have been DROPPED: a gap much wider than the typical pitch. Returns
/// insertion indices (k = "between character k-1 and k"; 0 = before the first, n = after the last).
/// `factor` is 1.9 because plates print wider gaps between their groups (1.5-1.75 pitches measured
/// on PP-OCR posteriors) while one missing character gives >= 2.0.
List<int> suspectedGaps(List<(int, int, double)> spans, int T, {double factor = 1.9, double edgeFactor = 2.3}) {
  final n = spans.length;
  if (n < 4) return [];
  final starts = [for (final s in spans) s.$1.toDouble()];
  final diffs = [for (var i = 1; i < n; i++) starts[i] - starts[i - 1]];
  final pitch = _median(diffs);
  if (pitch <= 0) return [];
  final out = <int>[];
  for (var k = 0; k < diffs.length; k++) {
    if (diffs[k] > factor * pitch) out.add(k + 1);
  }
  if (starts[0] > edgeFactor * pitch) out.add(0);
  if ((T - 1 - spans.last.$2) > edgeFactor * pitch) out.add(n);
  return out;
}

/// Beam-normalised posterior of each hypothesis (mass outside the beam is ignored).
List<double> posteriorOver(List<Hypothesis> hyps) {
  if (hyps.isEmpty) return [];
  var mx = double.negativeInfinity;
  for (final h in hyps) {
    if (h.score > mx) mx = h.score;
  }
  final p = [for (final h in hyps) math.exp(h.score - mx)];
  final sum = pairwiseSum(p, 0, p.length, 1);
  return [for (final x in p) x / sum];
}
