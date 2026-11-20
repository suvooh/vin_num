/// From a full photo to a plate / VIN READING: detect -> deskew -> recognise -> grammar decode.
/// Port of `ml/vinplate/scene.py` (current form).
///
/// The recogniser's posteriors are kept, so a damaged / glared / scratched character comes out as
/// `?` (an explicit "unreadable") instead of a confident wrong letter; the Indian plate grammar and
/// the ISO 3779 VIN grammar resolve AMBIGUITY (0/O, 5/S, ...) and may pin a SINGLE unreadable
/// character when a mandatory check digit makes it unique (always flagged as inferred); photometric
/// views of one crop and several photos are fused on the posteriors; clutter around the text
/// ("IND", "TC/20", a "VIN:" label, a second sticker line) is handled by decoding sub-windows; 2-row
/// plates and plates the detector split into pieces are rejoined.
///
/// It deliberately does NOT invent characters. `FieldReading.status` (accept / review / reject) is
/// the one field to branch on, and `reading.masked` - not `reading.text` - is what to display
/// whenever the status is not `accept`.
///
/// Pure Dart (no Flutter): runs inside the worker isolate. The recogniser is injected through
/// [RecEngine] so the whole decision logic is testable without ONNX Runtime.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/image_ops.dart' show PilFilter, resizePil, roundHalfEven;
import '../engine/types.dart';
import 'ctc.dart';
import 'decoder.dart';
import 'enhance.dart';
import 'geometry.dart';
import 'grammar.dart';
import 'plate_category.dart';
import 'pp_engine.dart' show PpEngine, RecOut;
import 'quality.dart';
import 'rows.dart';
import 'types.dart';
import 'vocab.dart';

// ======================================================================== constants
/// characters actually supported by pixels, per domain
const Map<String, int> minEvidence = {'plate': 6, 'vin': 9};

/// glyphs a damaged / blurred / glared character turns into each other
const List<String> lookalikes = ['O0QD', 'I1L', '5S', '8B', '2Z', '6G'];

/// alternative look-alike evidence that makes a character ambiguous. Measured on 113 look-alike
/// positions of synthetic reads: all 3 WRONG characters had >= 0.2; 0.02 also flagged 8 correct
/// ones, 0.1 flags 3. Only 3 wrong characters - re-fit on real data.
const double lookalikeP = 0.1;

/// below this a "reading" is noise, not a finding (provisional)
const double minConfidence = 0.15;

const Map<String, int> statusRank = {'accept': 2, 'review': 1, 'reject': 0};

/// The recogniser maps a crop to ~one time-step per 8 px of its 48-px-high resize. HSRP's tall,
/// condensed font gives a plate crop an aspect of ~2.6, i.e. ~16 steps for 9 characters - fewer than
/// CTC needs to separate repeated characters ("88", "8888"), so it dropped and merged them (a real
/// photo read HR88B8888 as HR8888). A crop narrower than [minAspect] x its height is stretched
/// horizontally (never beyond [maxStretch]) before recognition: measured on that photo, x1.5-2.0
/// reads it perfectly at 97-98% confidence.
const double minAspect = 4.0, maxStretch = 2.5;

/// Unknown-slot candidates (see [FieldReading.unknownCandidates]): options below this share of the
/// legal alphabet are not worth showing; one is `suggested` when it dominates.
const double candidateMinP = 0.05, suggestMinP = 0.6, suggestMinMargin = 0.3;

/// weight of the grammar's log-prior against the log of the recogniser's peak evidence when ranking candidates
const double candidatePriorWeight = 1.0;

/// Words of a web / operating-system interface. A real car scene has no reason to contain several.
const List<String> uiWords = [
  'SEARCH', 'RELATED', 'SETTINGS', 'DOWNLOAD', 'SUBSCRIBE', 'COPILOT', 'GOOGLE', 'CHROME', 'YOUTUBE', 'FACEBOOK',
  'WHATSAPP', 'SCREENSHOT', 'DASHBOARD', 'NOTIFICATIONS', 'EXTENSIONS', 'BOOKMARKS', 'DETAILS', 'IMAGES', 'WINDOWS',
  'OUTLOOK', 'TEAMS', //
];

/// A line of text counts towards "this scene is full of text" with at least this many alphanumerics.
const int sceneTextMinAlnum = 4;

/// Screen-context score weights and the line count that triggers the "many lines" cue.
const double ctxUrlWeight = 1.0, ctxWordsWeight = 1.5, ctxLinesWeight = 0.5;
const int ctxWordsNeeded = 3, ctxManyLines = 14;

final RegExp _urlTld = RegExp(r'\.(COM|ORG|NET|IN|CO)(?![A-Z0-9_])');

const Map<String, String> _confusable = {
  '0': 'O', 'O': '0', '1': 'I', 'I': '1', '5': 'S', 'S': '5', '8': 'B', 'B': '8', '2': 'Z',
  'Z': '2', '6': 'G', 'G': '6', '4': 'A', 'A': '4', //
};

final int _idI = kChars.indexOf('I') + 1;
final int _idO = kChars.indexOf('O') + 1;
final int _idQ = kChars.indexOf('Q') + 1;
final int _id1 = kChars.indexOf('1') + 1;
final int _id0 = kChars.indexOf('0') + 1;

/// a run of letters that long is a word, not a chassis number
final RegExp _wordRun = RegExp(r'[A-Z]{5,}');

/// Words printed on the same stickers / plates as a VIN. 17 characters of "TYRE PRESSURE 32 PSI"
/// is a legal-looking VIN to every grammar, so a read containing one of these is clutter, not a VIN.
/// A VIN whose ISO check digit passes is never filtered by it.
const List<String> labelWords = [
  'PRESSURE', 'TYRE', 'TIRE', 'FRONT', 'GVWR', 'GAWR', 'KERB', 'WEIGHT', 'MANUFACT', 'ENGINE',
  'CAPACITY', 'SEATING', 'CERTIF', 'WARNING', 'CAUTION', 'BATTERY', 'COLOUR', 'MOTORS', 'LIMITED',
  'VEHICLE', 'MODEL', 'MADE', 'INDIA', 'TOTAL', 'LOAD', //
];

// ========================================================================= helpers
/// Python's `round(x, d)`: half to even on the binary value (only an exact tie differs from
/// rounding half away from zero).
double pyRound(double x, int d) {
  final m = math.pow(10, d).toDouble();
  final scaled = x * m;
  final fl = scaled.floorToDouble();
  if (scaled - fl == 0.5) return ((fl % 2 == 0) ? fl : fl + 1) / m;
  return scaled.roundToDouble() / m;
}

/// Python's `sorted(items, key=...)` for a comparator: STABLE (equal items keep their order).
List<T> _stableSort<T>(Iterable<T> items, int Function(T a, T b) cmp) {
  final dec = <(int, T)>[];
  var i = 0;
  for (final it in items) {
    dec.add((i++, it));
  }
  dec.sort((a, b) {
    final c = cmp(a.$2, b.$2);
    return c != 0 ? c : a.$1.compareTo(b.$1);
  });
  return [for (final d in dec) d.$2];
}

int _cmpRank((int, int, double) a, (int, int, double) b) {
  if (a.$1 != b.$1) return a.$1.compareTo(b.$1);
  if (a.$2 != b.$2) return a.$2.compareTo(b.$2);
  return a.$3.compareTo(b.$3);
}

bool _isAlnum(String c) => (c.compareTo('0') >= 0 && c.compareTo('9') <= 0) || (c.compareTo('A') >= 0 && c.compareTo('Z') <= 0);
bool _isDigit(String c) => c.compareTo('0') >= 0 && c.compareTo('9') <= 0;

/// ASCII letters and digits of either case (the screen-context line count ignores other scripts).
int _asciiAlnumCount(String t) {
  var n = 0;
  for (final u in t.codeUnits) {
    if ((u >= 48 && u <= 57) || (u >= 65 && u <= 90) || (u >= 97 && u <= 122)) n++;
  }
  return n;
}

/// Stretches a too-narrow line crop horizontally (see [minAspect]). Sizes are exact and shared with
/// the Python reference: `newW = floor(w * s + 0.5)`, Pillow bicubic.
RgbImage normaliseAspect(RgbImage crop) {
  final aspect = crop.width / crop.height;
  if (aspect >= minAspect) return crop;
  final s = math.min(maxStretch, minAspect / aspect);
  final newW = (crop.width * s + 0.5).floor();
  if (newW <= crop.width) return crop;
  return resizePil(crop, newW, crop.height, PilFilter.bicubic);
}

/// Python `[x for x in ...]` first-maximum semantics of `max(key=...)`.
T _firstMax<T>(Iterable<T> items, int Function(T a, T b) cmp) {
  T? best;
  var first = true;
  for (final it in items) {
    if (first || cmp(it, best as T) > 0) {
      best = it;
      first = false;
    }
  }
  return best as T;
}

/// ISO 3779 forbids I, O and Q in a VIN, so a recogniser that reports one of them is certain to have
/// misread a 1 or a 0: move that probability mass onto the digit. Strict 17-character path only.
/// (Operation order as in numpy: exp, then `p1 += pI`, `p0 += pO + pQ`, floor the three at 1e-9,
/// normalise with numpy's pairwise row sum, log.)
Logp foldVin(Logp lp) {
  final n = lp.t * kVocab;
  final p = Float64List(n);
  for (var i = 0; i < n; i++) {
    p[i] = math.exp(lp.data[i]);
  }
  for (var r = 0; r < lp.t; r++) {
    final b = r * kVocab;
    p[b + _id1] += p[b + _idI];
    p[b + _id0] += p[b + _idO] + p[b + _idQ];
    p[b + _idI] = 1e-9;
    p[b + _idO] = 1e-9;
    p[b + _idQ] = 1e-9;
  }
  final out = Float64List(n);
  for (var r = 0; r < lp.t; r++) {
    final b = r * kVocab;
    final s = pairwiseSum(p, b, kVocab, 1);
    for (var v = 0; v < kVocab; v++) {
      out[b + v] = math.log(p[b + v] / s);
    }
  }
  return Logp(lp.t, out);
}

/// Two characters that are one common look-alike substitution away from a state code.
bool nearState(String head) {
  if (head.length != 2) return false;
  final a = [head[0], if (_confusable.containsKey(head[0])) _confusable[head[0]]!];
  final b = [head[1], if (_confusable.containsKey(head[1])) _confusable[head[1]]!];
  for (final x in a) {
    for (final y in b) {
      if (stateCodes.contains('$x$y')) return true;
    }
  }
  return false;
}

/// Two masked readings of the same identifier: equal length and no position where both are known
/// and different ('?' matches anything), allowing a single character of slack (a misread).
bool readsCompatible(String a, String b) {
  if (a.length != b.length) return false;
  var bad = 0;
  for (var i = 0; i < a.length; i++) {
    final x = a[i], y = b[i];
    if (x != '?' && y != '?' && x != y) bad++;
  }
  return bad <= 1;
}

// ===================================================================== geometry
(double, double) _axisU(TextBox b) {
  final w = math.max(b.width, 1e-6);
  return ((b.quad[2] - b.quad[0]) / w, (b.quad[3] - b.quad[1]) / w);
}

(double, double) _axisV(TextBox b) {
  final h = math.max(b.height, 1e-6);
  return ((b.quad[6] - b.quad[0]) / h, (b.quad[7] - b.quad[1]) / h);
}

/// [b] is a text row directly below [a] (2-row plate / wrapped sticker).
bool isStacked(TextBox a, TextBox b) {
  final u = _axisU(a), v = _axisV(a);
  final ca = a.center, cb = b.center;
  final dx = cb.$1 - ca.$1, dy = cb.$2 - ca.$2;
  final du = dx * u.$1 + dy * u.$2, dv = dx * v.$1 + dy * v.$2;
  final h = (a.height + b.height) / 2;
  return dv > 0.5 * h &&
      dv - h < 1.0 * h &&
      du.abs() < 0.6 * math.max(a.width, b.width) &&
      0.55 < a.height / b.height &&
      a.height / b.height < 1.8 &&
      0.3 < a.width / b.width &&
      a.width / b.width < 3.4 &&
      (a.angle - b.angle).abs() < 15;
}

/// [b] follows [a] on the same text row (a plate the detector split, a state code, a label).
bool isSideBySide(TextBox a, TextBox b) {
  final u = _axisU(a), v = _axisV(a);
  final ca = a.center, cb = b.center;
  final dx = cb.$1 - ca.$1, dy = cb.$2 - ca.$2;
  final du = dx * u.$1 + dy * u.$2, dv = dx * v.$1 + dy * v.$2;
  final h = (a.height + b.height) / 2;
  return du > 0 &&
      du - (a.width + b.width) / 2 < 1.4 * h &&
      dv.abs() < 0.5 * h &&
      0.6 < a.height / b.height &&
      a.height / b.height < 1.7 &&
      (a.angle - b.angle).abs() < 15;
}

/// Boxes close enough to belong to one plate: the gap between them is small relative to the text.
bool isNear(TextBox a, TextBox b) {
  final (ax0, ay0, ax1, ay1) = a.bbox;
  final (bx0, by0, bx1, by1) = b.bbox;
  final gx = math.max(0, math.max(ax0, bx0) - math.min(ax1, bx1));
  final gy = math.max(0, math.max(ay0, by0) - math.min(ay1, by1));
  final h = math.max(a.height, b.height);
  return gx < 1.6 * h && gy < 0.9 * h;
}

List<int> _boxList((int, int, int, int) b) => [b.$1, b.$2, b.$3, b.$4];

(int, int, int, int) _unionBbox(List<TextBox> boxes) {
  var x0 = 1 << 62, y0 = 1 << 62, x1 = -(1 << 62), y1 = -(1 << 62);
  for (final b in boxes) {
    final (a, bb, c, d) = b.bbox;
    x0 = math.min(x0, a);
    y0 = math.min(y0, bb);
    x1 = math.max(x1, c);
    y1 = math.max(y1, d);
  }
  return (x0, y0, x1, y1);
}

// ========================================================================== engine seam
/// What the reader needs from the recogniser. [PpRecEngine] adapts the ONNX-backed [PpEngine]; tests
/// use a replay fake.
abstract interface class RecEngine {
  List<TextBox> detect(RgbImage rgb);
  List<RecOut> recProbsMany(List<RgbImage> crops);
  Float64List toLogp(RecOut r, {double temperature = 1.0});

  /// RapidOCR-style best-path text WITH punctuation and case (the plate / VIN vocabulary drops both).
  /// Only used as context: the screen-context cue reads web addresses and interface words from it.
  String greedyText(RecOut r);
}

class PpRecEngine implements RecEngine {
  PpRecEngine(this.engine);
  final PpEngine engine;

  @override
  List<TextBox> detect(RgbImage rgb) => engine.detect(rgb);

  @override
  List<RecOut> recProbsMany(List<RgbImage> crops) => engine.recProbsMany(crops);

  @override
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => engine.toLogp(r, temperature: temperature);

  @override
  String greedyText(RecOut r) => engine.greedyText(r);
}

// ================================================================================ data
/// One detected text line with its crop and (lazily computed) per-view posteriors.
class LineObs {
  LineObs(this.box, this.crop, this.probs0, this.lp0, this.text0, {this.raw0 = ''});
  final TextBox box;
  final RgbImage crop; // already aspect-normalised ([normaliseAspect])
  final RecOut probs0;
  final Logp lp0;
  final String text0;

  /// the recogniser's own text with punctuation and case (context only; see [RecEngine.greedyText])
  final String raw0;

  /// view name -> posteriors; insertion-ordered ('orig' first), like the Python dict.
  final Map<String, Logp> viewLp = {};
  bool viewsReady = false;

  int get nAlnum {
    var n = 0;
    for (var i = 0; i < text0.length; i++) {
      if (_isAlnum(text0[i])) n++;
    }
    return n;
  }

  int get nDigit {
    var n = 0;
    for (var i = 0; i < text0.length; i++) {
      if (_isDigit(text0[i])) n++;
    }
    return n;
  }
}

class FieldReading {
  FieldReading({
    required this.domain,
    required this.reading,
    required this.parsed,
    required this.box,
    required this.quads,
    this.image = 0,
    this.views = 1,
    this.rows = 1,
    this.evidence = 0,
    this.quality,
    this.notes = const [],
    this.lines = const [],
    this.posterior,
    this.category,
  });

  final String domain; // 'plate' | 'vin'
  final Reading reading;
  final Map<String, dynamic>? parsed;
  final (int, int, int, int) box; // union bbox in image pixels
  final List<List<double>> quads; // one deskewed quad (8 numbers) per text line used
  final int image; // index of the photo it came from
  final int views; // photometric views fused
  final int rows;
  final int evidence; // characters backed by pixels
  final QualityReport? quality;
  final List<String> notes;
  final List<LineObs> lines; // the lines it was read from

  /// the (fused / sub-window) posterior the reading was decoded from; null when unknown
  final Logp? posterior;

  /// What kind of vehicle the plate is (see [classifyPlate]); null for a VIN or a rejected read.
  final Map<String, dynamic>? category;

  List<Map<String, dynamic>>? _cands;

  String get status => reading.status;

  (int, int, double) rank() => (statusRank[reading.status]!, math.min(evidence, 17), pyRound(reading.confidence, 3));

  /// For every character that is unknown (`?`) or weak (`inferred`): which characters could it be?
  /// The network's evidence at that character's frames is restricted to the characters the GRAMMAR
  /// allows in that slot (a series slot takes letters only, a digit slot digits only), renormalised,
  /// and the top options are returned, so a `?` becomes "probably B (93%), else R, D" instead of a
  /// blank. A dominant option is flagged `suggested`; it is still only a suggestion: the person
  /// confirms it. Entry: {position, options: [{char, p}], suggested: char | null}.
  List<Map<String, dynamic>> unknownCandidates() => _cands ??= _computeCandidates();

  List<Map<String, dynamic>> _computeCandidates() =>
      slotCandidates(reading.text, posterior, domain == 'plate' ? plateGrammar : vinGrammar, inferred: reading.inferred);

  /// Same keys as Python `FieldReading.as_dict`.
  Map<String, dynamic> toJson() {
    final r = reading;
    return {
      'domain': domain,
      'status': r.status,
      'value': r.text,
      'masked': r.masked,
      'confidence': pyRound(r.confidence, 3),
      'char_confidence': r.charConf,
      'inferred': r.inferred,
      'suspected_gaps': r.suspectedGaps,
      'grammar_cost': r.grammarCost,
      'raw_read': r.greedyText,
      'alternatives': [for (final (t, p) in r.nBest) {'text': t, 'p': p}],
      'reasons': r.reasons,
      'parsed': parsed,
      'box': [box.$1, box.$2, box.$3, box.$4],
      'quads': quads,
      'image': image,
      'views': views,
      'rows': rows,
      'evidence_chars': evidence,
      'quality': quality?.toJson(),
      'notes': notes,
      'calibrated': r.calibrated,
      'unknown_candidates': unknownCandidates(),
      'category': category,
    };
  }
}

/// Port of scene.slot_candidates. For every unknown (`?`) or weak ([inferred]) character of [text]: the
/// ranked characters the grammar allows in that slot, weighed by what the recogniser actually saw there
/// (the peak probability of each class over the frames CTC aligned to that character), renormalised over
/// the LEGAL characters. Entry: {position, options: [{char, p}] (top 3, p >= 5%), suggested: char | null};
/// `suggested` only when one option dominates (>= 0.6 and 0.3 ahead of the next). Positions index [text].
List<Map<String, dynamic>> slotCandidates(String text, Logp? posterior, Grammar g, {Iterable<int> inferred = const []}) {
  final lp = posterior;
  if (lp == null || text.isEmpty) return const [];
  final marked = inferred.toSet();
  final todo = [for (var i = 0; i < text.length; i++) if (text[i] == '?' || marked.contains(i)) i];
  if (todo.isEmpty) return const [];
  final List<(int, int, double)> spans;
  try {
    spans = ctcAlign(lp, encodeText(text));
  } on ArgumentError {
    return const [];
  }
  final out = <Map<String, dynamic>>[];
  for (final i in todo) {
    if (i >= spans.length) continue;
    final (a, b, pk) = spans[i];
    if (pk <= 0.0) continue; // no frames were aligned to this character
    // score of a candidate = ln(peak evidence over its frames) + the grammar's log-prior of the whole candidate
    // string, so the PATTERN decides too: a letter slot never offers a digit, and between two legal characters
    // the one that makes the likelier plate / VIN (a valid year code, a digit in the serial) ranks first.
    final kept = <(String, double)>[];
    for (var c = 1; c < kVocab - 1; c++) {
      // classes 1..36 are '0'-'9','A'-'Z'; 0 is the CTC blank and kVocab-1 is '?'
      var m = 0.0;
      for (var t = a; t <= b && t < lp.t; t++) {
        m = math.max(m, math.exp(lp.at(t, c)));
      }
      final ch = kChars[c - 1];
      final cand = text.substring(0, i) + ch + text.substring(i + 1);
      if (!g.isValid(cand)) continue;
      final prior = g.logPrior(cand);
      if (prior == null) continue;
      kept.add((ch, math.log(m + 1e-9) + candidatePriorWeight * prior));
    }
    if (kept.isEmpty) continue;
    final top = kept.fold(double.negativeInfinity, (s, e) => math.max(s, e.$2));
    final z = kept.fold(0.0, (s, e) => s + math.exp(e.$2 - top));
    final probs = [for (final (ch, sc) in kept) (ch, math.exp(sc - top) / z)];
    final ranked = _stableSort(probs, (x, y) => y.$2.compareTo(x.$2));
    final opts = [for (final (ch, p) in ranked.take(3)) if (p >= candidateMinP) (ch, p)];
    if (opts.isEmpty) continue;
    final p0 = opts[0].$2, p1 = opts.length > 1 ? opts[1].$2 : 0.0;
    out.add({
      'position': i,
      'options': [for (final (ch, p) in opts) {'char': ch, 'p': pyRound(p, 3)}],
      'suggested': (p0 >= suggestMinP && p0 - p1 >= suggestMinMargin) ? opts[0].$1 : null,
    });
  }
  return out;
}

/// "Is this a photo of a screen?" from the TEXT around the plate. A real car scene holds a plate and
/// maybe a badge; a browser or desktop holds dozens of lines, web addresses and interface words
/// (SEARCH, RELATED, SETTINGS ...). [rawLines] are the recogniser's own texts (punctuation and case
/// kept), [excluded] indexes the lines that belong to the plate / VIN itself.
/// score = 1.0 * url + 1.5 * (>= 3 interface words) + 0.5 * (>= 14 lines of text).
Map<String, dynamic> screenContext(List<String> rawLines, {Set<int> excluded = const {}}) {
  var textLines = 0;
  var url = false;
  final words = <String>{};
  for (var i = 0; i < rawLines.length; i++) {
    if (excluded.contains(i)) continue;
    final up = rawLines[i].toUpperCase();
    if (_asciiAlnumCount(rawLines[i]) >= sceneTextMinAlnum) textLines++;
    if (up.contains('HTTP') || up.contains('WWW') || _urlTld.hasMatch(up)) url = true;
    final letters = up.replaceAll(RegExp(r'[^A-Z]'), '');
    for (final w in uiWords) {
      if (letters.contains(w)) words.add(w);
    }
  }
  final sortedWords = words.toList()..sort();
  final manyWords = sortedWords.length >= ctxWordsNeeded;
  final manyLines = textLines >= ctxManyLines;
  final score = (url ? ctxUrlWeight : 0.0) + (manyWords ? ctxWordsWeight : 0.0) + (manyLines ? ctxLinesWeight : 0.0);
  return {
    'score': score,
    'url': url,
    'ui_words': sortedWords,
    'text_lines': textLines,
    'reasons': [
      if (url) 'a web address (URL) is visible in the scene',
      if (manyWords) 'browser / app interface words are visible: ${sortedWords.join(', ')}',
      if (manyLines) '$textLines separate lines of text are visible, far more than a vehicle scene normally has',
    ],
  };
}

Map<String, dynamic> _noContext() => {'score': 0.0, 'url': false, 'ui_words': <String>[], 'text_lines': 0, 'reasons': <String>[]};

class SceneResult {
  SceneResult({this.plate, this.vin, this.size = (0, 0)});
  FieldReading? plate, vin;
  List<Map<String, dynamic>> lines = [], candidates = [], conflicts = [];

  /// [screenContext] of this photo (all-zero when nothing suggests a screen)
  Map<String, dynamic> screenContext = _noContext();

  /// one result per photo ([SceneReader.readFrames] with several photos only)
  List<SceneResult> perImage = [];
  (int, int) size; // (width, height)
  Map<String, double> timings = {};
  List<String> notes = [];

  Map<String, dynamic> toJson() => {
        'plate': plate?.toJson(),
        'vin': vin?.toJson(),
        'lines': lines,
        'conflicts': conflicts,
        'image_size': [size.$1, size.$2],
        'timings_ms': timings,
        'notes': notes,
        'screen_context': screenContext,
      };
}

/// Best plate and best VIN such that no text line is used twice (a VIN sticker must not also be
/// reported as a plate, nor a plate as a chassis number). Exposed for tests.
(FieldReading?, FieldReading?) assignFields(List<FieldReading> pool) {
  bool ok(FieldReading f) =>
      // a read this weak is not a finding: keep it out of the result (it stays in `candidates`)
      f.evidence >= minEvidence[f.domain]! && f.status != 'reject' && f.reading.confidence >= minConfidence;

  int byRankDesc(FieldReading a, FieldReading b) => _cmpRank(b.rank(), a.rank());
  final plates = _stableSort(pool.where((f) => f.domain == 'plate' && ok(f)), byRankDesc);
  final vins = _stableSort(pool.where((f) => f.domain == 'vin' && ok(f)), byRankDesc);

  bool shares(FieldReading a, FieldReading b) {
    final ix = math.max(0, math.min(a.box.$3, b.box.$3) - math.max(a.box.$1, b.box.$1));
    final iy = math.max(0, math.min(a.box.$4, b.box.$4) - math.max(a.box.$2, b.box.$2));
    final small = math.min((a.box.$3 - a.box.$1) * (a.box.$4 - a.box.$2), (b.box.$3 - b.box.$1) * (b.box.$4 - b.box.$2));
    return ix * iy > 0.4 * math.max(small, 1);
  }

  FieldReading? bestP, bestV;
  var bestS = -1.0;
  final ps = plates.isEmpty ? <FieldReading?>[null] : plates.take(4).toList();
  final vs = vins.isEmpty ? <FieldReading?>[null] : vins.take(4).toList();
  for (final p in ps) {
    for (final v in vs) {
      if (p != null && v != null && shares(p, v)) continue;
      var s = 0.0;
      for (final f in [p, v]) {
        if (f != null) s += f.evidence * (1.0 + 0.5 * statusRank[f.status]!);
      }
      if (s > bestS) {
        bestP = p;
        bestV = v;
        bestS = s;
      }
    }
  }
  if (bestP == null && bestV == null && (plates.isNotEmpty || vins.isNotEmpty)) {
    return (plates.isNotEmpty ? plates[0] : null, plates.isNotEmpty ? null : vins[0]);
  }
  return (bestP, bestV);
}

// ================================================================================ reader
class SceneReader {
  SceneReader(
    this.engine, {
    this.multiView = true,
    this.temperature = 1.0,
    this.maxLines = 6,
    this.refineTop = 3,
    Set<String> views = const {'orig', 'stretch', 'deglare'},
  }) : views = Set<String>.of(views);

  final RecEngine engine;
  final bool multiView;
  final double temperature;
  final int maxLines;
  final int refineTop;

  /// Paired ablation (40 synthetic scenes): orig + stretch gave the same gain as all five views, so
  /// the default is orig + stretch + deglare (deglare only runs when a crop has clipped hot-spots).
  final Set<String> views;

  // ------------------------------------------------------------------------ observation
  ({List<LineObs> lines, Map<String, double> timings}) observe(RgbImage rgb) {
    final sw = Stopwatch()..start();
    final boxes = [for (final b in engine.detect(rgb)) if (b.height >= 7 && b.width >= 12) b];
    final tDetect = sw.elapsedMicroseconds / 1000.0;
    final crops = [for (final b in boxes) normaliseAspect(warpQuad(rgb, b.quad))];
    final probs = crops.isEmpty ? <RecOut>[] : engine.recProbsMany(crops);
    final lines = <LineObs>[];
    for (var i = 0; i < boxes.length; i++) {
      final lp = _logp(probs[i]);
      lines.add(LineObs(boxes[i], crops[i], probs[i], lp, greedy(lp).$1, raw0: engine.greedyText(probs[i])));
    }
    return (lines: lines, timings: {'detect': tDetect, 'recognise': sw.elapsedMicroseconds / 1000.0 - tDetect});
  }

  Logp _logp(RecOut r) => Logp(r.t, engine.toLogp(r, temperature: temperature));

  /// Posteriors of every photometric view shared by all [lines] (computed once per line).
  Map<String, List<Logp>> _viewLogps(List<LineObs> lines) {
    final todoC = <RgbImage>[];
    final todoK = <(LineObs, String)>[];
    for (final ln in lines) {
      ln.viewLp.putIfAbsent('orig', () => ln.lp0);
      if (!ln.viewsReady) {
        for (final (name, img) in variants(ln.crop, views: views)) {
          if (name != 'orig' && views.contains(name)) {
            todoC.add(img);
            todoK.add((ln, name));
          }
        }
        ln.viewsReady = true;
      }
    }
    if (todoC.isNotEmpty) {
      final outs = engine.recProbsMany(todoC);
      for (var i = 0; i < todoK.length; i++) {
        todoK[i].$1.viewLp[todoK[i].$2] = _logp(outs[i]);
      }
    }
    final names = [for (final n in lines[0].viewLp.keys) if (lines.every((ln) => ln.viewLp.containsKey(n))) n];
    return {for (final n in names) n: [for (final ln in lines) ln.viewLp[n]!]};
  }

  // ----------------------------------------------------------------------------- decode
  /// A single line is used as is; several are blank-trimmed and concatenated.
  static Logp join(List<Logp> lps) => lps.length == 1 ? lps[0] : Logp.concat(lps);

  /// Decode one domain from one or several (view-stacked) posteriors. null = not a finding for this
  /// domain (a VIN read that is label clutter / not a strict VIN when legacy is not allowed ...).
  /// Public for tests.
  (Reading, String)? decode(String domain, List<Logp> lpsByView, {required bool allowLegacy}) {
    if (domain == 'vin') {
      final strict = _decodeOne('vin', [for (final lp in lpsByView) foldVin(lp)]);
      if (parseVin(strict.$1.text)?['layout'] == 'vin17') {
        return isClutter(strict.$1.text) ? null : strict;
      }
      if (!allowLegacy) return null;
      // legacy / short chassis numbers: weaker structure, so never auto-accepted
      final legacy = _decodeOne('vin', lpsByView);
      final r = legacy.$1;
      var digits = 0;
      for (var i = 0; i < r.text.length; i++) {
        if (_isDigit(r.text[i])) digits++;
      }
      if (_wordRun.hasMatch(r.text) || digits < 0.3 * r.text.length || isClutter(r.text)) {
        return null; // reads like a word / label, not a chassis number
      }
      if (parseVin(r.text)?['layout'] == 'vin17') return legacy;
      if (r.status == 'accept') r.status = 'review';
      r.reasons.add('short / legacy chassis number: its structure cannot be validated');
      return legacy;
    }
    return _decodeOne(domain, lpsByView);
  }

  (Reading, String) _decodeOne(String domain, List<Logp> lpsByView) {
    final g = domain == 'plate' ? plateGrammar : vinGrammar;
    var reading = lpsByView.length == 1 ? readLine(lpsByView[0], g) : fuseFrames(lpsByView, g);
    var note = '';
    var used = lpsByView;
    if (reading.status == 'reject' || !reading.valid || _implausible(domain, reading)) {
      final win = _bestWindow(domain, lpsByView, g);
      if (win != null && _cmpPair((statusRank[win.$1.status]!, win.$1.confidence), (statusRank[reading.status]!, reading.confidence)) > 0) {
        reading = win.$1;
        used = win.$2;
        note = 'decoded a sub-window of the line (surrounding text ignored)';
      }
    }
    final fused = used.length == 1 ? used[0] : entropyFuse(used);
    final hardened = _harden(domain, reading, fused, g);
    _posteriorOf[hardened] = fused;
    return (hardened, note);
  }

  /// the posterior each decoded [Reading] came from (Python keeps it in `_last_post`)
  final Expando<Logp> _posteriorOf = Expando<Logp>('posterior');

  /// Decode-time safety policy. A reading may only stay `accept` if no character is ambiguous
  /// between look-alikes that the grammar would ALSO allow there (O/Q/0/D, I/1/L, 5/S, 8/B, 2/Z,
  /// 6/G) with real evidence for the alternative - such a character is marked unknown rather than
  /// shown as read - and (plates) it has series letters and a 4-digit number, and no I/O/Q series
  /// letter.
  Reading _harden(String domain, Reading r, Logp fused, Grammar g) {
    if (r.status == 'reject' || !r.valid || r.text.contains('?')) return r;
    final weak = <int>[];
    var alts = <List<(String, double)>>[];
    try {
      alts = alternatives(fused, r.text, k: 4);
    } on ArgumentError {
      alts = []; // includes RangeError; Python: except (KeyError, ValueError)
    }
    for (var i = 0; i < r.text.length && i < alts.length; i++) {
      final c = r.text[i];
      String? group;
      for (final x in lookalikes) {
        if (x.contains(c)) {
          group = x;
          break;
        }
      }
      if (group == null) continue;
      for (final (ch, p) in alts[i]) {
        if (group.contains(ch) && ch != c && p >= lookalikeP && g.isValid(r.text.substring(0, i) + ch + r.text.substring(i + 1))) {
          weak.add(i);
          break;
        }
      }
    }
    if (weak.isNotEmpty) {
      final merged = {...r.inferred, ...weak}.toList()..sort();
      r.inferred = merged;
      final shown = weak.toList()..sort();
      r.reasons.add('ambiguous look-alike character(s) at position(s) [${shown.join(', ')}]: shown as unknown');
    }
    if (domain == 'plate') {
      final p = parsePlate(r.text);
      if (p != null && p['layout'] == 'standard' && ((p['series'] as String).isEmpty || (p['number'] as String).length < 4)) {
        r.reasons.add('no series letters or a short number: a very old plate, or a character was dropped');
        weak.add(-1);
      }
      if (p != null && p['layout'] == 'standard' && (p['series'] as String).split('').any((c) => 'IOQ'.contains(c))) {
        // the grammar notes it: series letters skip I and O (and Q in many states); a read that
        // contains one is far more likely a misread D / Q / 0 than a real series
        r.reasons.add('series letter I/O/Q: not issued on HSRP plates, so it is probably a misread look-alike');
        weak.add(-1);
      }
    }
    if (weak.isNotEmpty && r.status == 'accept') r.status = 'review';
    return r;
  }

  /// The strict path has folded I->1 and O->0 (ENGINE -> ENG1NE), so match words on the unfolded form.
  static bool isClutter(String text) {
    final unfolded = text.replaceAll('1', 'I').replaceAll('0', 'O');
    if (vinCheckDigitOk(text)) return false;
    return labelWords.any((w) => text.contains(w) || unfolded.contains(w));
  }

  static bool _implausible(String domain, Reading r) {
    var n = 0;
    for (var i = 0; i < r.text.length; i++) {
      if (r.text[i] != '?') n++;
    }
    return n < minEvidence[domain]!;
  }

  /// Sub-windows of the line. Decoding the whole line fails when it carries extra text; a window
  /// that starts where a state code (plate) / any 17-character span (VIN) could start and ends on a
  /// digit (Bharat plates: any character) is tried instead. Windows are cut on the CTC alignment of
  /// the best path.
  (Reading, List<Logp>)? _bestWindow(String domain, List<Logp> lps, Grammar g) {
    final ref = lps[0];
    final text = greedy(ref).$1;
    final nUnk = '?'.allMatches(text).length;
    if (text.length < 6 || (text.contains('?') && nUnk > text.length ~/ 2)) return null;
    final List<(int, int, double)> spans;
    try {
      spans = ctcAlign(ref, encodeText(text));
    } on ArgumentError {
      return null;
    }
    final n = text.length;
    final wins = <(int, int)>[];
    if (domain == 'plate') {
      for (var i = 0; i < n - 6; i++) {
        final head = text.substring(i, math.min(i + 2, n));
        final bharat = text.substring(i + 2, math.min(i + 4, n)) == 'BH'; // 22 BH 1234 AA ends in letters
        if (!(stateCodes.contains(head) || head.contains('?') || bharat || nearState(head))) continue;
        for (var j = i + 7; j <= math.min(n, i + 11); j++) {
          if (!(i == 0 && j == n) && (bharat || _isDigit(text[j - 1]) || text[j - 1] == '?')) wins.add((i, j));
        }
      }
    } else {
      for (var i = 0; i < n - 16; i++) {
        if (!(i == 0 && i + 17 == n)) wins.add((i, i + 17));
      }
    }
    (Reading, List<Logp>)? best;
    final T = ref.t;
    for (final (i, j) in wins.take(40)) {
      final a = math.max(0, spans[i].$1 - 1), b = math.min(T, spans[j - 1].$2 + 2);
      final sub = <Logp>[
        for (final lp in lps)
          lp.slice(((a * lp.t) / T).toInt(), math.max(((a * lp.t) / T).toInt() + 2, ((b * lp.t) / T).toInt())),
      ];
      final r = sub.length == 1 ? readLine(sub[0], g) : fuseFrames(sub, g);
      if (!r.valid || _implausible(domain, r)) continue;
      if (best == null || _cmpPair((statusRank[r.status]!, r.confidence), (statusRank[best.$1.status]!, best.$1.confidence)) > 0) {
        best = (r, sub);
      }
    }
    return best;
  }

  static int _cmpPair((int, double) a, (int, double) b) => a.$1 != b.$1 ? a.$1.compareTo(b.$1) : a.$2.compareTo(b.$2);

  // ---------------------------------------------------------------------- one photo
  SceneResult read(RgbImage rgb, {Set<String> domains = const {'plate', 'vin'}, bool allowLegacyVin = false, int imageIndex = 0}) {
    final swAll = Stopwatch()..start();
    final res = SceneResult(size: (rgb.width, rgb.height));
    final obs = observe(rgb);
    final lines = obs.lines;
    res.timings.addAll({for (final e in obs.timings.entries) e.key: pyRound(e.value, 1)});
    res.lines = [
      for (final ln in lines)
        {'text': ln.text0, 'box': _boxList(ln.box.bbox), 'detector_score': pyRound(ln.box.score, 3)}
    ];
    var short = _stableSort(
      lines.where((ln) => ln.nDigit >= 2 || (ln.text0.length >= 6 && '?'.allMatches(ln.text0).length >= 2)),
      (a, b) {
        if (a.nAlnum != b.nAlnum) return b.nAlnum.compareTo(a.nAlnum); // key -n_alnum
        return a.box.center.$2.compareTo(b.box.center.$2);
      },
    ).take(maxLines).toList();
    short = _stableSort(short, (a, b) {
      final c = a.box.center.$2.compareTo(b.box.center.$2);
      return c != 0 ? c : a.box.center.$1.compareTo(b.box.center.$1);
    });
    final cands = <List<LineObs>>[for (final ln in short) [ln]];
    final pairs = <(LineObs, LineObs)>[];
    for (final a in short) {
      for (final b in short) {
        if (identical(a, b)) continue;
        if (isStacked(a.box, b.box) || isSideBySide(a.box, b.box)) pairs.add((a, b));
      }
    }
    double dist((LineObs, LineObs) ab) {
      final ca = ab.$1.box.center, cb = ab.$2.box.center;
      final dx = ca.$1 - cb.$1, dy = ca.$2 - cb.$2;
      return math.sqrt(dx * dx + dy * dy);
    }

    final sortedPairs = _stableSort(pairs, (x, y) => dist(x).compareTo(dist(y)));
    for (final p in sortedPairs.take(8)) {
      cands.add([p.$1, p.$2]);
    }
    final regions = domains.contains('plate') ? _regions(lines, short, rgb) : <List<LineObs>>[];

    final swDec = Stopwatch()..start();
    final scored = <FieldReading>[];
    for (final domain in domains) {
      for (final lns in cands) {
        var est = 0;
        for (final ln in lns) {
          est += math.max(ln.nAlnum, 1);
        }
        if ((domain == 'plate' && !(est >= 4 && est <= 16)) || (domain == 'vin' && est < 6)) continue;
        final dec = decode(domain, [join([for (final ln in lns) ln.lp0])], allowLegacy: allowLegacyVin);
        if (dec == null) continue;
        final (reading, note) = dec;
        scored.add(_makeField(domain, lns, reading, imageIndex, [if (note.isNotEmpty) note], views: 1));
      }
    }
    for (final rows in regions) {
      final dec = decode('plate', [join([for (final ln in rows) ln.lp0])], allowLegacy: false);
      if (dec != null) {
        final (reading, note) = dec;
        scored.add(_makeField('plate', rows, reading, imageIndex, [if (note.isNotEmpty) note, 'read as one plate region'], views: 1));
      }
    }
    // refine the most promising candidates with every photometric view
    final refined = <FieldReading>[];
    if (multiView) {
      for (final domain in domains) {
        final top = _stableSort(scored.where((f) => f.domain == domain), (a, b) => _cmpRank(b.rank(), a.rank())).take(refineTop);
        for (final f in top) {
          if (f.status == 'accept' && f.reading.confidence >= 0.97 && f.reading.reasons.isEmpty) continue;
          final lns = f.lines;
          final vws = _viewLogps(lns);
          final names = vws.keys.toList();
          if (names.length < 2) continue;
          final stack = [for (final n in names) join(vws[n]!)];
          final dec = decode(domain, stack, allowLegacy: allowLegacyVin);
          if (dec == null) continue;
          final (reading, note) = dec;
          refined.add(_makeField(domain, lns, reading, imageIndex, [if (note.isNotEmpty) note, 'fused ${names.length} photometric views'], views: names.length));
        }
      }
    }
    final pool = [...scored, ...refined];
    res.timings['decode'] = pyRound(swDec.elapsedMicroseconds / 1000.0, 1);
    res.candidates = [
      for (final f in _stableSort(pool, (a, b) => _cmpRank(b.rank(), a.rank())).take(12))
        {
          'domain': f.domain,
          'value': f.reading.text,
          'masked': f.reading.masked,
          'status': f.status,
          'confidence': pyRound(f.reading.confidence, 3),
          'views': f.views,
          'evidence': f.evidence,
        }
    ];
    final (plate, vin) = assignFields(pool);
    res.plate = plate;
    res.vin = vin;
    res.screenContext = screenContext([for (final ln in lines) ln.raw0], excluded: _ownedLines(lines, [plate, vin]));
    res.timings['total'] = pyRound(swAll.elapsedMicroseconds / 1000.0, 1);
    return res;
  }

  // ----------------------------------------------------------------------------- helpers
  /// Indices of detected lines that lie (>= 40% of their own area) inside a chosen plate / VIN box.
  static Set<int> _ownedLines(List<LineObs> lines, List<FieldReading?> fields) {
    final owned = <int>{};
    for (var i = 0; i < lines.length; i++) {
      final (x0, y0, x1, y1) = lines[i].box.bbox;
      final area = math.max(0, x1 - x0) * math.max(0, y1 - y0);
      for (final f in fields) {
        if (f == null) continue;
        final ix = math.max(0, math.min(x1, f.box.$3) - math.max(x0, f.box.$1));
        final iy = math.max(0, math.min(y1, f.box.$4) - math.max(y0, f.box.$2));
        if (area > 0 && ix * iy >= 0.4 * area) owned.add(i);
      }
    }
    return owned;
  }

  /// Plate-sized regions: connected groups of nearby text boxes, re-read as ONE deskewed crop (split
  /// into rows when it is plate-shaped). This rescues plates the detector fragmented - a row cut by
  /// a scratch, a state code split from the number - which no pairing of boxes can.
  List<List<LineObs>> _regions(List<LineObs> lines, List<LineObs> short, RgbImage rgb, {int maxRegions = 3}) {
    final n = lines.length;
    final parent = List<int>.generate(n, (i) => i);
    int find(int i) {
      while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    }

    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        if (isNear(lines[i].box, lines[j].box)) parent[find(i)] = find(j);
      }
    }
    final groups = <int, List<int>>{}; // insertion-ordered like the Python dict
    for (var i = 0; i < n; i++) {
      groups.putIfAbsent(find(i), () => []).add(i);
    }
    final seeds = Set<LineObs>.identity()..addAll(short);
    final out = <List<LineObs>>[];
    for (final idx in groups.values) {
      final members = [for (final i in idx) lines[i]];
      if (members.length < 2 || !members.any(seeds.contains)) continue;
      final pts = <double>[for (final m in members) ...m.box.quad];
      final r = minAreaRect(pts);
      final quad = r.corners;
      if (r.short < 12 || !(1.1 <= r.long / r.short && r.long / r.short <= 7.5)) continue;
      final crop = warpQuad(rgb, quad, pad: 0.05);
      final h = crop.height, w = crop.width;
      if (h < 12) continue;
      final crops = [for (final c in (looksTwoRow(h, w) ? splitRows(crop) : [crop])) normaliseAspect(c)];
      final recs = engine.recProbsMany(crops);
      final rows = <LineObs>[];
      final score = members.map((m) => m.box.score).reduce(math.min);
      for (var k = 0; k < crops.length; k++) {
        final lp = _logp(recs[k]);
        final a = k / crops.length, b = (k + 1) / crops.length;
        // quad[i] are (x, y) points: TL = 0, TR = 1, BR = 2, BL = 3
        double px(int pt, int c) => quad[2 * pt + c];
        final rq = <double>[
          for (final c in [0, 1]) px(0, c) + (px(3, c) - px(0, c)) * a,
          for (final c in [0, 1]) px(1, c) + (px(2, c) - px(1, c)) * a,
          for (final c in [0, 1]) px(1, c) + (px(2, c) - px(1, c)) * b,
          for (final c in [0, 1]) px(0, c) + (px(3, c) - px(0, c)) * b,
        ];
        rows.add(LineObs(TextBox(rq, score), crops[k], recs[k], lp, greedy(lp).$1));
      }
      out.add(rows);
    }
    final sorted = _stableSort(out, (a, b) {
      int digits(List<LineObs> r) => r.fold(0, (s, ln) => s + ln.nDigit);
      return (-digits(a)).compareTo(-digits(b));
    });
    return sorted.take(maxRegions).toList();
  }

  FieldReading _makeField(String domain, List<LineObs> lns, Reading reading, int image, List<String> notes, {required int views}) {
    final parsed = (reading.valid && reading.status != 'reject') ? (domain == 'plate' ? parsePlate(reading.text) : parseVin(reading.text)) : null;
    var evidence = 0;
    for (var i = 0; i < reading.text.length; i++) {
      if (reading.text[i] != '?' && !reading.inferred.contains(i)) evidence++;
    }
    final biggest = _firstMax(lns, (a, b) => (a.box.width * a.box.height).compareTo(b.box.width * b.box.height));
    return FieldReading(
      domain: domain,
      reading: reading,
      parsed: parsed,
      category: _categoryOf(domain, lns, parsed),
      box: _unionBbox([for (final ln in lns) ln.box]),
      quads: [for (final ln in lns) _roundQuad(ln.box.quad)],
      image: image,
      views: views,
      rows: lns.length,
      evidence: evidence,
      quality: biggest.crop.height >= 8 ? assessCrop(biggest.crop) : null,
      notes: notes,
      lines: List<LineObs>.of(lns),
      posterior: _posteriorOf[reading],
    );
  }

  /// Vehicle category of a PLATE from the biggest line crop it was read from (null for a VIN or no usable pixels).
  /// Port of scene._category_of.
  static Map<String, dynamic>? _categoryOf(String domain, List<LineObs> lns, Map<String, dynamic>? parsed) {
    if (domain != 'plate' || lns.isEmpty) return null;
    final biggest = _firstMax(lns, (a, b) => (a.box.width * a.box.height).compareTo(b.box.width * b.box.height));
    final crop = biggest.crop;
    if (crop.height < 4 || crop.width < 4) return null;
    return classifyPlate(crop, parsed);
  }

  /// numpy `round(a, 1)`: `rint(a * 10) / 10`.
  static List<double> _roundQuad(List<double> q) => [for (final v in q) roundHalfEven(v * 10) / 10];

  // ------------------------------------------------------------------- several photos
  /// Several photos of the same vehicle: read each, then fuse the readings of the same identifier on
  /// the posteriors (glare and smudges move between photos). Readings of the same domain that cannot
  /// be reconciled are reported in `conflicts` - never silently averaged.
  SceneResult readFrames(List<RgbImage> rgbs, {Set<String> domains = const {'plate', 'vin'}, bool allowLegacyVin = false}) {
    final singles = [for (var i = 0; i < rgbs.length; i++) read(rgbs[i], domains: domains, allowLegacyVin: allowLegacyVin, imageIndex: i)];
    if (singles.length == 1) return singles[0];
    final out = SceneResult(size: singles[0].size)
      ..lines = [for (final s in singles) ...s.lines]
      ..perImage = singles;
    out.timings = {
      for (final k in ['detect', 'recognise', 'decode', 'total']) k: pyRound(singles.fold(0.0, (s, r) => s + (r.timings[k] ?? 0)), 1)
    };
    out.screenContext = _firstMax([for (final s in singles) s.screenContext], (a, b) => (a['score'] as double).compareTo(b['score'] as double));
    for (final domain in domains) {
      final found = <(int, FieldReading)>[];
      for (var i = 0; i < singles.length; i++) {
        final f = domain == 'plate' ? singles[i].plate : singles[i].vin;
        if (f != null) found.add((i, f));
      }
      if (found.isEmpty) continue;
      // group by compatibility, keep the group with the most evidence
      final groups = <List<(int, FieldReading)>>[];
      for (final item in found) {
        var placed = false;
        for (final g in groups) {
          if (readsCompatible(g[0].$2.reading.masked, item.$2.reading.masked)) {
            g.add(item);
            placed = true;
            break;
          }
        }
        if (!placed) groups.add([item]);
      }
      double groupScore(List<(int, FieldReading)> g) => g.fold(0.0, (s, it) => s + it.$2.evidence * (1 + statusRank[it.$2.status]!));
      final sorted = _stableSort(groups, (a, b) => groupScore(b).compareTo(groupScore(a)));
      final main = sorted[0];
      for (final g in sorted.skip(1)) {
        for (final (i, f) in g) {
          out.conflicts.add({
            'domain': domain,
            'image': i,
            'masked': f.reading.masked,
            'agrees_with': [for (final m in main) m.$1],
            'main': main[0].$2.reading.masked,
          });
        }
      }
      var best = _firstMax([for (final m in main) m.$2], (a, b) => _cmpRank(a.rank(), b.rank()));
      if (main.length > 1) {
        final fused = _fuseGroup(domain, main, rgbs, allowLegacyVin);
        if (fused != null && _cmpRank(fused.rank(), best.rank()) >= 0) best = fused;
      }
      if (domain == 'plate') {
        out.plate = best;
      } else {
        out.vin = best;
      }
    }
    return out;
  }

  /// Re-decode one identifier from the best-view posteriors of every photo that read it.
  FieldReading? _fuseGroup(String domain, List<(int, FieldReading)> group, List<RgbImage> rgbs, bool allowLegacy) {
    final lps = <Logp>[];
    final wts = <double>[];
    for (final (i, f) in group) {
      final lns = <Logp>[];
      for (final q in f.quads) {
        final crop = normaliseAspect(warpQuad(rgbs[i], q));
        final p = engine.recProbsMany([crop])[0];
        lns.add(_logp(p));
      }
      lps.add(join(lns));
      wts.add(1.0 + 0.5 * statusRank[f.status]!);
    }
    if (lps.length < 2) return null;
    final g = domain == 'plate' ? plateGrammar : vinGrammar;
    final reading = fuseFrames(lps, g, weights: wts);
    final fusedPost = entropyFuse(lps, weights: wts);
    if (reading.status == 'reject' || (domain == 'vin' && !allowLegacy && parseVin(reading.text)?['layout'] != 'vin17')) return null;
    final (i, f) = _firstMax(group, (a, b) => _cmpRank(a.$2.rank(), b.$2.rank()));
    var evidence = 0;
    for (var k = 0; k < reading.text.length; k++) {
      if (reading.text[k] != '?' && !reading.inferred.contains(k)) evidence++;
    }
    final fusedParsed = reading.status != 'reject' ? (domain == 'plate' ? parsePlate(reading.text) : parseVin(reading.text)) : null;
    // the category is judged from the best-ranked photo's biggest line crop (the fused posterior has no pixels),
    // against the FUSED reading's text layout
    final category = f.lines.isNotEmpty ? _categoryOf(domain, f.lines, fusedParsed) : f.category;
    return FieldReading(
      domain: domain,
      reading: reading,
      parsed: fusedParsed,
      category: category,
      box: f.box,
      quads: f.quads,
      image: i,
      views: lps.length,
      rows: f.rows,
      evidence: evidence,
      quality: f.quality,
      notes: ['fused ${lps.length} photos: ${group.map((m) => m.$1).join(', ')}'],
      posterior: fusedPost,
    );
  }
}
