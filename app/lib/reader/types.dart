/// Shared types of the on-device reader: the recogniser's output as log-probabilities over the
/// plate / VIN alphabet. Pure Dart (no Flutter), so everything in `lib/reader` can run in an isolate.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Vocabulary size: 0 = CTC blank, 1..36 = '0'-'9','A'-'Z', 37 = '?' (unreadable).
const int kVocab = 38;

/// Class i (1-based) <-> kChars[i-1].
const String kChars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?';

/// Log-probabilities [t][kVocab], row-major, natural log. Mirrors the numpy `[T, V]` arrays of
/// `ml/vinplate` (float64).
class Logp {
  Logp(this.t, this.data) : assert(data.length == t * kVocab, 'data must hold t * $kVocab values');

  /// Builds from nested rows (tests, small fixtures).
  factory Logp.fromRows(List<List<double>> rows) {
    final d = Float64List(rows.length * kVocab);
    for (var i = 0; i < rows.length; i++) {
      assert(rows[i].length == kVocab);
      for (var v = 0; v < kVocab; v++) {
        d[i * kVocab + v] = rows[i][v];
      }
    }
    return Logp(rows.length, d);
  }

  /// Builds from a flat list (T * kVocab values).
  factory Logp.fromFlat(List<num> flat) {
    assert(flat.length % kVocab == 0);
    final d = Float64List(flat.length);
    for (var i = 0; i < flat.length; i++) {
      d[i] = flat[i].toDouble();
    }
    return Logp(flat.length ~/ kVocab, d);
  }

  final int t;
  final Float64List data;

  double at(int ti, int v) => data[ti * kVocab + v];

  /// Frames [from, to), clipped to [0, t] the way a Python slice is.
  Logp slice(int from, int to) {
    final a = from.clamp(0, t);
    final b = to.clamp(0, t);
    if (b <= a) return Logp(0, Float64List(0));
    return Logp(b - a, Float64List.fromList(data.sublist(a * kVocab, b * kVocab)));
  }

  /// `vinplate.decode.lines.trim_blank`: drops leading / trailing frames that are confidently
  /// blank, keeping [keep] frames of margin. An all-blank input leaves its first frame.
  Logp trimBlank({int keep = 1, double blankP = 0.9}) {
    var first = -1, last = -1;
    for (var i = 0; i < t; i++) {
      if (math.exp(data[i * kVocab]) < blankP) {
        if (first < 0) first = i;
        last = i;
      }
    }
    if (first < 0) return slice(0, 1);
    return slice(first - keep, last + keep + 1);
  }

  /// `concat_lines`: each part is blank-trimmed first (otherwise the junction of two rows of a
  /// 2-row plate looks like a wide gap and trips the dropped-character detector), then joined.
  static Logp concat(List<Logp> parts) {
    final trimmed = [for (final p in parts) p.trimBlank()];
    final total = trimmed.fold<int>(0, (s, p) => s + p.t);
    final d = Float64List(total * kVocab);
    var off = 0;
    for (final p in trimmed) {
      d.setRange(off, off + p.data.length, p.data);
      off += p.data.length;
    }
    return Logp(total, d);
  }
}
