// PP-OCR (PaddleOCR v3 detector + recogniser) for the on-device reader, exposing the recogniser's
// PROBABILITIES (not just a string) projected onto the plate / VIN alphabet. Port of
// `ml/vinplate/ocr/ppocr.py`. The grammar decoder, the "?" for damaged characters and multi-view
// fusion all need those per-time-step probabilities.
//
// Layout of this file: the pure functions (`detPreprocess`, `boxesFromProbMap`, `recPreprocess`,
// `VocabProjection`) have no native dependency and are unit-tested on the PC; only `PpEngine.detect`
// and `PpEngine.recProbsMany` call ONNX Runtime and can only be exercised on a device.

import 'dart:convert' show LineSplitter;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../engine/image_ops.dart' show PilFilter, resizePil, roundHalfEven;
import '../engine/ort.dart';
import '../engine/types.dart';
import 'geometry.dart';

// ---- detector config (RapidOCR config.yaml, as in the Python reference) ----------------------
const detLimitSide = 736;
const detMaxSide = 1920;
const dbThresh = 0.3;
const dbBoxThresh = 0.5;
const dbUnclip = 1.6;
const dbMinSize = 3;
const recHeight = 48;
const recMaxWidth = 1600;

const _detMean = [0.485, 0.456, 0.406];
const _detStd = [0.229, 0.224, 0.225];

/// The decoder's alphabet: class 0 = CTC blank, 1..36 = '0'-'9','A'-'Z', 37 = '?' (unreadable).
const vocabAlphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const vocabSize = 38;
const vocabBlank = 0;
const vocabUnk = 37;

// ===================================================================== recogniser output
/// Raw PP-OCR softmax for one crop: [t] time steps x [classes] (6625 for the shipped model).
class RecOut {
  const RecOut(this.probs, this.t, this.classes);
  final Float32List probs;
  final int t;
  final int classes;
}

// ================================================================ vocabulary projection
/// Projects PP-OCR classes onto the decoder alphabet. Letters fold to upper case; blank and the
/// space class go to CTC blank; a lowercase l / bar is spread over L, 1, I; EVERY other symbol (CJK,
/// punctuation, ...) goes to '?'. A glyph damaged enough to look like a Chinese character is exactly
/// the case to abstain on.
///
/// Fast path: nearly all of the 6.6k classes map to '?', so only the few classes that map to a real
/// character or blank are gathered and '?' is taken as the remainder, 1 - (mass on those).
class VocabProjection {
  /// [chars]: the full class list: index 0 = blank (""), then the charset lines, then " ".
  factory VocabProjection(List<String> chars) {
    final cls = <int>[], voc = <int>[];
    final wt = <double>[];
    void add(int k, int v, double w) {
      cls.add(k);
      voc.add(v);
      wt.add(w);
    }

    int idx(String c) => vocabAlphabet.indexOf(c) + 1;
    for (var k = 0; k < chars.length; k++) {
      final c = chars[k];
      if (k == 0 || c == ' ') {
        add(k, vocabBlank, 1.0);
      } else if (c == 'l' || c == '|') {
        add(k, idx('L'), 0.34);
        add(k, idx('1'), 0.33);
        add(k, idx('I'), 0.33);
      } else if (c.length == 1 && c != '?' && RegExp(r'^[0-9A-Za-z]$').hasMatch(c)) {
        add(k, idx(c.toUpperCase()), 1.0);
      } // everything else: '?', taken as the remainder
    }
    return VocabProjection._(chars.length, Int32List.fromList(cls), Int32List.fromList(voc), Float64List.fromList(wt));
  }

  VocabProjection._(this.classes, this._cls, this._voc, this._wt);

  final int classes;
  final Int32List _cls, _voc;
  final Float64List _wt;

  /// Number of classes that do NOT map to '?'.
  int get keptClasses => _cls.toSet().length;

  /// Natural-log probabilities [t][vocabSize], row-major. [temperature] > 1 flattens.
  Float64List toLogp(Float32List probs, int t, {double temperature = 1.0}) {
    final out = Float64List(t * vocabSize);
    final p = Float64List(vocabSize);
    for (var s = 0; s < t; s++) {
      p.fillRange(0, vocabSize, 0.0);
      final base = s * classes;
      var kept = 0.0; // float32-ish sum of the mass on kept classes
      var last = -1;
      for (var i = 0; i < _cls.length; i++) {
        final k = _cls[i];
        final pr = probs[base + k];
        p[_voc[i]] += pr * _wt[i];
        if (k != last) {
          kept += pr; // a class with several targets (l, |) is counted once
          last = k;
        }
      }
      p[vocabUnk] = math.max(0.0, 1.0 - kept);
      var total = 0.0;
      for (var v = 0; v < vocabSize; v++) {
        if (p[v] < 1e-9) p[v] = 1e-9; // np.clip(p, 1e-9, None)
        total += p[v];
      }
      final o = s * vocabSize;
      for (var v = 0; v < vocabSize; v++) {
        out[o + v] = math.log(p[v] / total);
      }
      if (temperature != 1.0) {
        var mx = -double.infinity;
        for (var v = 0; v < vocabSize; v++) {
          out[o + v] /= temperature;
          if (out[o + v] > mx) mx = out[o + v];
        }
        var e = 0.0;
        for (var v = 0; v < vocabSize; v++) {
          out[o + v] -= mx;
          e += math.exp(out[o + v]);
        }
        final lse = math.log(e);
        for (var v = 0; v < vocabSize; v++) {
          out[o + v] -= lse;
        }
      }
    }
    return out;
  }
}

// ============================================================================ pre-processing
/// Detector input: scale so the short side is at least [detLimitSide] (and the long side at most
/// [maxDetSide]), round both sides to multiples of 32, BGR order, ImageNet mean / std, NCHW float32.
({Float32List data, int rw, int rh}) detPreprocess(RgbImage img, {int maxDetSide = detMaxSide}) {
  final w = img.width, h = img.height;
  var ratio = 1.0;
  if (math.min(h, w) < detLimitSide) ratio = detLimitSide / math.min(h, w);
  if (math.max(h, w) * ratio > maxDetSide) ratio = maxDetSide / math.max(h, w);
  final rh = math.max(32, roundHalfEven((h * ratio).truncate() / 32) * 32);
  final rw = math.max(32, roundHalfEven((w * ratio).truncate() / 32) * 32);
  final im = resizePil(img, rw, rh, PilFilter.bilinear);
  final n = rw * rh;
  final data = Float32List(3 * n);
  final px = im.rgb;
  for (var i = 0; i < n; i++) {
    // channel c of the BGR image is rgb byte (2 - c)
    data[i] = (px[3 * i + 2] / 255.0 - _detMean[0]) / _detStd[0];
    data[n + i] = (px[3 * i + 1] / 255.0 - _detMean[1]) / _detStd[1];
    data[2 * n + i] = (px[3 * i] / 255.0 - _detMean[2]) / _detStd[2];
  }
  return (data: data, rw: rw, rh: rh);
}

/// Recogniser input for one crop: height 48, width ceil(48 * w / h) in [8, 1600], BGR, (x/255-.5)/.5,
/// CHW float32. Crops of the same width can be batched without padding.
({Float32List data, int w}) recPreprocess(RgbImage crop) {
  final nw = math.min(recMaxWidth, math.max(8, (recHeight * crop.width / math.max(crop.height, 1)).ceil()));
  final im = resizePil(crop, nw, recHeight, PilFilter.bilinear);
  final plane = recHeight * nw;
  final data = Float32List(3 * plane);
  final px = im.rgb;
  for (var i = 0; i < plane; i++) {
    data[i] = (px[3 * i + 2] / 255.0 - 0.5) / 0.5;
    data[plane + i] = (px[3 * i + 1] / 255.0 - 0.5) / 0.5;
    data[2 * plane + i] = (px[3 * i] / 255.0 - 0.5) / 0.5;
  }
  return (data: data, w: nw);
}

// ======================================================================= DB post-process
/// DBNet probability map [rw] x [rh] -> rotated text boxes in the original [w] x [h] image.
/// Threshold 0.3, 2x2 dilation (looks up / left), 8-connected components of the dilated map, box
/// score = mean probability over the component's thresholded pixels (>= 0.5), minimum-area
/// rectangle of the component, unclip 1.6 (grow every side by d = long*short*1.6 / (2(long+short))),
/// scale to image coordinates and clip. Sorted by (centre y, centre x).
List<TextBox> boxesFromProbMap(Float32List prob, int rw, int rh, int w, int h) {
  final n = rw * rh;
  final seg = Uint8List(n);
  for (var i = 0; i < n; i++) {
    seg[i] = prob[i] > dbThresh ? 1 : 0;
  }
  final dil = Uint8List.fromList(seg);
  for (var y = 0; y < rh; y++) {
    for (var x = 0; x < rw; x++) {
      if (dil[y * rw + x] == 1) continue;
      final i = y * rw + x;
      if ((y > 0 && seg[i - rw] == 1) || (x > 0 && seg[i - 1] == 1) || (y > 0 && x > 0 && seg[i - rw - 1] == 1)) dil[i] = 1;
    }
  }
  final seen = Uint8List(n);
  final queue = Int32List(n); // every component is a contiguous run of this buffer
  var tail = 0;
  final out = <TextBox>[];
  for (var start = 0; start < n; start++) {
    if (dil[start] == 0 || seen[start] == 1) continue;
    final s0 = tail;
    queue[tail++] = start;
    seen[start] = 1;
    for (var head = s0; head < tail; head++) {
      final p = queue[head];
      final y = p ~/ rw, x = p - y * rw;
      for (var dy = -1; dy <= 1; dy++) {
        final yy = y + dy;
        if (yy < 0 || yy >= rh) continue;
        for (var dx = -1; dx <= 1; dx++) {
          final xx = x + dx;
          if (xx < 0 || xx >= rw) continue;
          final q = yy * rw + xx;
          if (dil[q] == 1 && seen[q] == 0) {
            seen[q] = 1;
            queue[tail++] = q;
          }
        }
      }
    }
    final count = tail - s0;
    if (count < 8) continue;
    var minY = rh, maxY = -1;
    var scoreSum = 0.0, scoreN = 0;
    for (var i = s0; i < tail; i++) {
      final y = queue[i] ~/ rw;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (seg[queue[i]] == 1) {
        scoreSum += prob[queue[i]];
        scoreN++;
      }
    }
    final score = scoreN == 0 ? 0.0 : scoreSum / scoreN;
    if (score < dbBoxThresh) continue;
    // The hull of a blob is the hull of its per-row extreme pixels.
    final rows = maxY - minY + 1;
    final lo = Int32List(rows)..fillRange(0, rows, 1 << 30);
    final hi = Int32List(rows)..fillRange(0, rows, -1);
    for (var i = s0; i < tail; i++) {
      final y = queue[i] ~/ rw, x = queue[i] - y * rw;
      final r = y - minY;
      if (x < lo[r]) lo[r] = x;
      if (x > hi[r]) hi[r] = x;
    }
    final pts = <double>[];
    for (var r = 0; r < rows; r++) {
      if (hi[r] < 0) continue;
      pts..add(lo[r].toDouble())..add((r + minY).toDouble());
      if (hi[r] != lo[r]) pts..add(hi[r].toDouble())..add((r + minY).toDouble());
    }
    final rect = minAreaRect(pts);
    if (rect.short < dbMinSize) continue;
    final c = rect.corners;
    final d = rect.long * rect.short * dbUnclip / (2 * (rect.long + rect.short));
    final cx = (c[0] + c[2] + c[4] + c[6]) / 4, cy = (c[1] + c[3] + c[5] + c[7]) / 4;
    final ux = (c[2] - c[0]) / math.max(rect.long, 1e-6), uy = (c[3] - c[1]) / math.max(rect.long, 1e-6);
    final vx = (c[6] - c[0]) / math.max(rect.short, 1e-6), vy = (c[7] - c[1]) / math.max(rect.short, 1e-6);
    final l = rect.long / 2 + d, sh = rect.short / 2 + d;
    if (math.min(2 * l, 2 * sh) < dbMinSize + 2) continue;
    final q = <double>[
      cx - ux * l - vx * sh, cy - uy * l - vy * sh,
      cx + ux * l - vx * sh, cy + uy * l - vy * sh,
      cx + ux * l + vx * sh, cy + uy * l + vy * sh,
      cx - ux * l + vx * sh, cy - uy * l + vy * sh,
    ];
    for (var i = 0; i < 4; i++) {
      q[2 * i] = (q[2 * i] * (w / rw)).clamp(0, w - 1).toDouble();
      q[2 * i + 1] = (q[2 * i + 1] * (h / rh)).clamp(0, h - 1).toDouble();
    }
    final tb = TextBox(q, score);
    if (tb.width <= 3 || tb.height <= 3) continue;
    out.add(tb);
  }
  out.sort((a, b) {
    final c = a.center.$2.compareTo(b.center.$2);
    return c != 0 ? c : a.center.$1.compareTo(b.center.$1);
  });
  return out;
}

/// Best-path decode of raw PP-OCR probabilities with the full class list [chars] (index 0 = blank).
String greedyDecode(RecOut r, List<String> chars) {
  final sb = StringBuffer();
  var prev = -1;
  for (var s = 0; s < r.t; s++) {
    final base = s * r.classes;
    var best = 0;
    var bv = r.probs[base];
    for (var c = 1; c < r.classes; c++) {
      if (r.probs[base + c] > bv) {
        bv = r.probs[base + c];
        best = c;
      }
    }
    if (best != 0 && best != prev && best < chars.length) sb.write(chars[best]);
    prev = best;
  }
  return sb.toString();
}

// ========================================================================= the engine
class PpEngine {
  /// Loads both models from the app's assets.
  ///
  /// [nnapi] is accepted so existing call sites compile but it has NO EFFECT: the PP-OCR detector and
  /// recogniser have dynamic input shapes and XNNPACK / NNAPI crash natively on them on Android, so
  /// both are always loaded with the plain CPU provider (see lib/id_reader.dart).
  static Future<PpEngine> load({bool nnapi = false}) async {
    final det = (await rootBundle.load('assets/models/ocr_det.onnx')).buffer.asUint8List();
    final rec = (await rootBundle.load('assets/models/ocr_rec.onnx')).buffer.asUint8List();
    final charset = await rootBundle.loadString('assets/models/ocr_charset.txt');
    return PpEngine.fromBytes(det, rec, charset, nnapi: nnapi);
  }

  /// For isolates and tests: the model files and charset text already in memory. [nnapi] has no
  /// effect (CPU provider only, 4 threads) - see [load].
  PpEngine.fromBytes(Uint8List det, Uint8List rec, String charset, {bool nnapi = false, int threads = 4, this.maxDetSide = detMaxSide})
      : chars = ['', ...const LineSplitter().convert(charset), ' '],
        _det = Model.load('det', det, accelerator: Accelerator.cpu, threads: threads),
        _rec = Model.load('rec', rec, accelerator: Accelerator.cpu, threads: threads) {
    _proj = VocabProjection(chars);
  }

  final List<String> chars;
  final int maxDetSide;
  final Model _det, _rec;
  late final VocabProjection _proj;

  /// Text lines found in [rgb], as rotated boxes (see [boxesFromProbMap]).
  List<TextBox> detect(RgbImage rgb) {
    final p = detPreprocess(rgb, maxDetSide: maxDetSide);
    final pred = _det.run(_det.inputNames.first, p.data, [1, 3, p.rh, p.rw]).values.first.$1;
    return boxesFromProbMap(pred, p.rw, p.rh, rgb.width, rgb.height);
  }

  /// Raw recogniser probabilities for upright crops. Crops are grouped by their normalised width so
  /// no padding (which would change the probabilities) is ever added to a crop.
  List<RecOut> recProbsMany(List<RgbImage> crops) {
    final prepared = [for (final c in crops) recPreprocess(c)];
    final out = List<RecOut?>.filled(crops.length, null);
    final groups = <int, List<int>>{};
    for (var i = 0; i < prepared.length; i++) {
      groups.putIfAbsent(prepared[i].w, () => []).add(i);
    }
    for (final e in groups.entries) {
      final nw = e.key, idx = e.value;
      final plane = 3 * recHeight * nw;
      final batch = Float32List(idx.length * plane);
      for (var k = 0; k < idx.length; k++) {
        batch.setRange(k * plane, (k + 1) * plane, prepared[idx[k]].data);
      }
      final res = _rec.run(_rec.inputNames.first, batch, [idx.length, 3, recHeight, nw]).values.first;
      final t = res.$2[1], classes = res.$2[2];
      for (var k = 0; k < idx.length; k++) {
        out[idx[k]] = RecOut(Float32List.sublistView(res.$1, k * t * classes, (k + 1) * t * classes), t, classes);
      }
    }
    return [for (final o in out) o!];
  }

  /// Probabilities projected onto the decoder alphabet, natural log, row-major [t][vocabSize].
  Float64List toLogp(RecOut r, {double temperature = 1.0}) => _proj.toLogp(r.probs, r.t, temperature: temperature);

  /// RapidOCR-style best-path text (argmax per step, repeats and blanks dropped), for diagnostics.
  String greedyText(RecOut r) => greedyDecode(r, chars);

  void dispose() {
    _det.release();
    _rec.release();
  }
}
