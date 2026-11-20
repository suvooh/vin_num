// Ports of the Python reference checks (ml/vinplate/authenticity). Pure Dart, unit-tested.
// All thresholds are PROVISIONAL and mirror the Python provisional values.
import 'dart:math' as math;
import 'dart:typed_data';

// ---------------------------------------------------------------- image helper
class Rgb {
  final int w, h;
  final Uint8List data; // interleaved RGB
  Rgb(this.w, this.h, this.data);
  int at(int x, int y, int c) => data[(y * w + x) * 3 + c];
}

double _lin(int v) => math.pow(v / 255.0, 2.2).toDouble();
double lum(Rgb im, int x, int y) =>
    _lin(im.at(x, y, 0)) * 0.2126 + _lin(im.at(x, y, 1)) * 0.7152 + _lin(im.at(x, y, 2)) * 0.0722;

double _median(List<double> v) {
  if (v.isEmpty) return 0;
  final s = [...v]..sort();
  return s[s.length ~/ 2];
}

// ------------------------------------------------------------ retroreflection
class Box {
  final int x0, y0, x1, y1;
  const Box(this.x0, this.y0, this.x1, this.y1);
  bool contains(int x, int y) => x >= x0 && x < x1 && y >= y0 && y < y1;
}

class RetroResult {
  final String verdict; // retroreflective | not_retroreflective | inconclusive
  final double? ratio, gainPlate, gainSurround;
  final List<String> reasons;
  RetroResult(this.verdict, {this.ratio, this.gainPlate, this.gainSurround, this.reasons = const []});
}

/// Small whole-pixel shift of [mov] that best matches [ref] (SAD on a 4x box-downsampled luma).
(int, int) alignShift(Rgb ref, Rgb mov, {int search = 5}) {
  const f = 4;
  List<double> small(Rgb im) {
    final w = im.w ~/ f, h = im.h ~/ f;
    final o = List<double>.filled(w * h, 0);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var s = 0.0;
        for (var j = 0; j < f; j++) {
          for (var i = 0; i < f; i++) {
            s += math.sqrt(lum(im, x * f + i, y * f + j));
          }
        }
        o[y * w + x] = s / (f * f);
      }
    }
    return o;
  }

  final w = ref.w ~/ f, h = ref.h ~/ f;
  final a = small(ref), b = small(mov);
  var best = double.infinity;
  var bx = 0, by = 0;
  for (var dy = -search; dy <= search; dy++) {
    for (var dx = -search; dx <= search; dx++) {
      var sad = 0.0;
      var n = 0;
      for (var y = search; y < h - search; y += 2) {
        for (var x = search; x < w - search; x += 2) {
          sad += (a[y * w + x] - b[(y + dy) * w + (x + dx)]).abs();
          n++;
        }
      }
      sad /= n;
      if (sad < best) {
        best = sad;
        bx = dx;
        by = dy;
      }
    }
  }
  return (bx * f, by * f); // content in mov sits at +d relative to ref
}

/// Flash-OFF vs flash-ON frames. HSRP sheeting is retroreflective: it brightens far more under
/// flash than its surround. A screen emits its own light; a print reflects diffusely.
RetroResult retroreflection(Rgb off, Rgb on, Box plate,
    {bool hsrpExpected = true, double minRatio = 2.5, double maxRatio = 1.4, bool align = true}) {
  if (!hsrpExpected) {
    return RetroResult('inconclusive', reasons: ['plate not expected to be retroreflective (non-HSRP)']);
  }
  var dx = 0, dy = 0;
  if (align) (dx, dy) = alignShift(off, on);
  final ringPx = math.max(6, (0.5 * math.sqrt((plate.x1 - plate.x0) * (plate.y1 - plate.y0))).round());
  final outer = Box(plate.x0 - ringPx, plate.y0 - ringPx, plate.x1 + ringPx, plate.y1 + ringPx);
  final inner = Box(plate.x0 - ringPx ~/ 4, plate.y0 - ringPx ~/ 4, plate.x1 + ringPx ~/ 4, plate.y1 + ringPx ~/ 4);
  final offP = <double>[], onP = <double>[], offR = <double>[], onR = <double>[];
  for (var y = math.max(0, outer.y0); y < math.min(off.h, outer.y1); y += 2) {
    for (var x = math.max(0, outer.x0); x < math.min(off.w, outer.x1); x += 2) {
      final ox = x + dx, oy = y + dy;
      if (ox < 0 || oy < 0 || ox >= on.w || oy >= on.h) continue;
      final lo = lum(off, x, y), ln = lum(on, ox, oy);
      if (plate.contains(x, y)) {
        offP.add(lo);
        onP.add(ln);
      } else if (!inner.contains(x, y)) {
        offR.add(lo);
        onR.add(ln);
      }
    }
  }
  if (offP.length < 100 || offR.length < 100) {
    return RetroResult('inconclusive', reasons: ['plate or surround region too small to measure']);
  }
  const eps = 1e-3;
  final op = _median(offP), np = _median(onP), or = _median(offR), nr = _median(onR);
  final gp = (np + eps) / (op + eps), gr = (nr + eps) / (or + eps);
  List<String> why(String s) => [s];
  if (math.max(gp, gr) < 1.03) {
    return RetroResult('inconclusive', gainPlate: gp, gainSurround: gr, reasons: why('flash/torch did not brighten the scene (did it fire?)'));
  }
  if (op > 0.85) {
    return RetroResult('inconclusive', gainPlate: gp, gainSurround: gr, reasons: why('plate already saturated without flash (strong ambient light)'));
  }
  if (op < 0.01 && or < 0.01) {
    return RetroResult('inconclusive', gainPlate: gp, gainSurround: gr, reasons: why('scene too dark in the flash-off frame'));
  }
  final headroom = (1 + eps) / (op + eps);
  if (headroom / math.max(gr, 1.0) < minRatio * 1.15) {
    return RetroResult('inconclusive', gainPlate: gp, gainSurround: gr, reasons: why(
        'flash too strong for the exposure to measure: the surround brightened ${gr.toStringAsFixed(1)}x and the plate '
        '(${op.toStringAsFixed(2)} linear before flash) could only show ${headroom.toStringAsFixed(1)}x before the sensor clips. '
        'Retake from further away or in more ambient light (plate not near-black, flash not overwhelming).'));
  }
  final ratio = gp / gr;
  if (ratio >= minRatio) return RetroResult('retroreflective', ratio: ratio, gainPlate: gp, gainSurround: gr);
  if (ratio <= maxRatio) {
    return RetroResult('not_retroreflective', ratio: ratio, gainPlate: gp, gainSurround: gr, reasons: [
      'plate brightened ${gp.toStringAsFixed(2)}x under flash vs ${gr.toStringAsFixed(2)}x for its surround '
          '(a screen emits its own light; a print reflects like its surroundings)'
    ]);
  }
  return RetroResult('inconclusive', ratio: ratio, gainPlate: gp, gainSurround: gr, reasons: why('ratio in the grey zone'));
}

// ---------------------------------------------------------------- provenance
class Finding {
  final String code, severity, detail; // hard | soft | info
  Finding(this.code, this.severity, this.detail);
}

const _aiTools = [
  'stable diffusion', 'midjourney', 'dall-e', 'dall·e', 'firefly', 'imagen', 'gemini', 'flux', 'comfyui',
  'automatic1111', 'novelai', 'invokeai', 'leonardo.ai', 'ideogram', 'sdxl', 'openai', 'dreamstudio', 'nano banana'
];
const _editors = [
  'photoshop', 'gimp', 'lightroom', 'snapseed', 'pixlr', 'paint.net', 'canva', 'picsart', 'affinity', 'photopea',
  'facetune', 'meitu', 'capture one', 'luminar', 'fotor', 'pixelmator'
];
const _dstHard = {
  'trainedalgorithmicmedia', 'compositewithtrainedalgorithmicmedia', 'algorithmicmedia', 'compositesynthetic',
  'datadrivenmedia', 'screencapture', 'print'
};

/// Metadata scan over the raw file bytes (EXIF Software, XMP DigitalSourceType, AI-tool markers,
/// C2PA presence). Absence of a marker means NOTHING; C2PA is reported "present, NOT verified"
/// because verification needs the c2pa library (server side).
List<Finding> scanProvenance(Uint8List bytes) {
  final out = <Finding>[];
  final text = String.fromCharCodes(bytes.length > 400000 ? bytes.sublist(0, 400000) : bytes).toLowerCase();
  if (text.contains('digitalsourcetype')) {
    for (final m in RegExp(r'digitalsourcetype/([a-z]+)').allMatches(text)) {
      final v = m.group(1)!;
      if (_dstHard.contains(v)) out.add(Finding('xmp_$v', 'hard', 'XMP DigitalSourceType = $v'));
    }
  }
  // Tool names are matched only inside the XMP packet and the header region (where EXIF/comment
  // strings live), on word boundaries: short tokens ("flux") must not match random JPEG bytes.
  final x0 = text.indexOf('<x:xmpmeta');
  final x1 = x0 < 0 ? -1 : text.indexOf('</x:xmpmeta>', x0);
  final region = (x0 >= 0 && x1 > x0 ? text.substring(x0, x1) : '') + text.substring(0, math.min(text.length, 20000));
  String? aiHit;
  for (final t in _aiTools) {
    if (RegExp('(^|[^a-z0-9])${RegExp.escape(t)}([^a-z0-9]|\$)').hasMatch(region)) {
      aiHit = t;
      break;
    }
  }
  if (aiHit != null) {
    out.add(Finding('ai_marker', 'hard', 'metadata names an AI image tool ($aiHit)'));
  } else if (text.contains('steps: ') && text.contains('sampler: ') && text.contains('cfg scale')) {
    out.add(Finding('ai_marker', 'hard', 'Stable-Diffusion generation parameters embedded'));
  }
  if (out.every((f) => f.code != 'ai_marker')) {
    for (final e in _editors) {
      if (RegExp('(^|[^a-z0-9])${RegExp.escape(e)}').hasMatch(region)) {
        out.add(Finding('editor_marker', 'soft', 'metadata mentions an image editor ($e)'));
        break;
      }
    }
  }
  if (text.contains('c2pa') && text.contains('jumb')) {
    out.add(Finding('c2pa_unverified', 'info', 'C2PA manifest present but NOT verified on device'));
  }
  if (!text.contains('exif')) {
    out.add(Finding('no_exif', 'info', 'no EXIF (stripped by a messenger/screenshot, or not a camera file)'));
  }
  return out;
}

// -------------------------------------------------------------------- fusion
class Decision {
  final String status; // REJECT | REVIEW | NO FLAGS
  final double paScore;
  final List<String> errors; // user-facing errors, one per failed check
  final List<String> notRun;
  final List<String> positive;
  Decision(this.status, this.paScore, this.errors, this.notRun, this.positive);
}

Decision decide({
  required List<Finding> provenance,
  double? recaptureScore,
  RetroResult? retro,
  required bool flashCaptured,
  required bool fromLiveCamera,
  double? tamperScore,
  bool located = true,
}) {
  final errors = <String>[], notRun = <String>[], positive = <String>[];
  var reject = false, review = false;
  for (final f in provenance.where((f) => f.severity == 'hard')) {
    reject = true;
    errors.add('AI / SCREEN-CAPTURE DECLARED: ${f.detail}');
  }
  for (final f in provenance.where((f) => f.severity == 'soft')) {
    review = true;
    errors.add('EDITED? ${f.detail}');
  }
  if (!located) {
    notRun.add('plate/VIN not located by OCR: liveness and edit checks used the guide box, not the real region');
  }
  // The text-edit model is EXPERIMENTAL (held-out AUC 0.79 on simulated edits, 38% detection at 5%
  // false positives): its score is displayed but deliberately never changes the verdict.
  notRun.add('text-edit (tamper) check: experimental, score shown but NOT used in the decision'
      '${tamperScore == null ? ' (no plate/VIN region located)' : ''}');
  var pa = 0.0;
  final parts = <String>[];
  if (recaptureScore == null) {
    notRun.add('screen/print model');
  } else if (recaptureScore >= 0.9) {
    pa += 2;
    parts.add('screen/print model ${recaptureScore.toStringAsFixed(2)} (+2)');
  } else if (recaptureScore >= 0.5) {
    pa += 1;
    parts.add('screen/print model ${recaptureScore.toStringAsFixed(2)} (+1)');
  }
  if (!fromLiveCamera) {
    notRun.add('liveness (flash retroreflection): image was not captured live');
  } else if (!flashCaptured || retro == null) {
    notRun.add('liveness (flash retroreflection)');
  } else if (retro.verdict == 'not_retroreflective') {
    pa += 2;
    parts.add('plate did not retroreflect under flash (+2)');
  } else if (retro.verdict == 'retroreflective') {
    positive.add('plate retroreflected under flash (consistent with a real HSRP plate)');
  } else {
    notRun.add('liveness inconclusive: ${retro.reasons.join('; ')}');
  }
  if (pa >= 3) {
    reject = true;
    errors.add('LIVENESS / SCREEN-CAPTURE FAILED (score ${pa.toStringAsFixed(1)}): ${parts.join('; ')}');
  } else if (pa >= 1) {
    review = true;
    errors.add('POSSIBLE SCREEN/PRINT RE-CAPTURE (score ${pa.toStringAsFixed(1)}): ${parts.join('; ')}');
  }
  // "No flags" with nothing affirmative is NOT a pass: it means the checks that could catch a fake
  // either did not run or were inconclusive. Say so.
  final status = reject ? 'REJECT' : (review ? 'REVIEW' : (positive.isNotEmpty ? 'NO FLAGS' : 'CANNOT VERIFY'));
  if (status == 'CANNOT VERIFY') {
    errors.add('Nothing affirmative could be established: no liveness evidence, so this image is UNVERIFIED '
        '(a photo of a screen or print could look exactly like this to the checks that did run).');
  }
  return Decision(status, pa, errors, notRun, positive);
}

// ------------------------------------------------------------------- patches
/// 4 textured, non-overlapping native-resolution 96x96 patches from [roi], NCHW float32 in 0..1.
Float32List? pickPatches(Rgb im, Box roi, {int k = 4, int size = 96}) {
  final x0 = math.max(0, roi.x0), y0 = math.max(0, roi.y0);
  final x1 = math.min(im.w, roi.x1), y1 = math.min(im.h, roi.y1);
  if (x1 - x0 < size || y1 - y0 < size) return null;
  final cands = <(double, int, int)>[];
  for (var y = y0; y <= y1 - size; y += size ~/ 2) {
    for (var x = x0; x <= x1 - size; x += size ~/ 2) {
      var s = 0.0, s2 = 0.0;
      var n = 0;
      for (var j = 0; j < size; j += 3) {
        for (var i = 0; i < size; i += 3) {
          final g = im.at(x + i, y + j, 1).toDouble();
          s += g;
          s2 += g * g;
          n++;
        }
      }
      cands.add((s2 / n - (s / n) * (s / n), x, y));
    }
  }
  cands.sort((a, b) => b.$1.compareTo(a.$1));
  final chosen = <(int, int)>[];
  for (final c in cands) {
    if (chosen.every((p) => (c.$2 - p.$1).abs() >= size || (c.$3 - p.$2).abs() >= size)) chosen.add((c.$2, c.$3));
    if (chosen.length == k) break;
  }
  while (chosen.length < k) {
    chosen.add(chosen.isEmpty ? (x0, y0) : chosen[chosen.length % chosen.length]);
  }
  final out = Float32List(k * 3 * size * size);
  var o = 0;
  for (final p in chosen) {
    for (var c = 0; c < 3; c++) {
      for (var j = 0; j < size; j++) {
        for (var i = 0; i < size; i++) {
          out[o++] = im.at(p.$1 + i, p.$2 + j, c) / 255.0;
        }
      }
    }
  }
  return out;
}

double sigmoid(double x) => 1 / (1 + math.exp(-x));

/// Up to [k] native 96px patches spread across [box] (horizontally, centred vertically), clamped
/// inside the image: what the tamper model inspects. NCHW float32 in 0..1, or null if the image
/// is smaller than one patch.
Float32List? patchesAround(Rgb im, Box box, {int k = 4, int size = 96}) {
  if (im.w < size || im.h < size) return null;
  final cy = (box.y0 + box.y1) ~/ 2;
  final span = math.max(1, box.x1 - box.x0);
  final out = Float32List(k * 3 * size * size);
  var o = 0;
  for (var n = 0; n < k; n++) {
    final cx = box.x0 + (span * (n + 0.5) / k).round();
    final x = (cx - size ~/ 2).clamp(0, im.w - size);
    final y = (cy - size ~/ 2).clamp(0, im.h - size);
    for (var c = 0; c < 3; c++) {
      for (var j = 0; j < size; j++) {
        for (var i = 0; i < size; i++) {
          out[o++] = im.at(x + i, y + j, c) / 255.0;
        }
      }
    }
  }
  return out;
}
