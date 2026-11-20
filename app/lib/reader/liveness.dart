// On-device authenticity / liveness DECISION, ported from ml/vinplate/authenticity/bundle.py
// (`assess_bundle`'s decision logic) and fusion.py (`decide`). It works on ALREADY-COMPUTED inputs
// (provenance findings, PadNet scores, the flash result, planarity results), so the expensive
// parts run wherever the caller likes (an isolate) and this stays a pure, testable function.
//
// Two questions, never confused:
//   status    - were red flags raised?            no_flags | review | reject
//   liveness  - is there POSITIVE physical evidence of a real scene?   confirmed | suspected_spoof | not_established
//
// `no_flags` is NOT liveness. `confirmed` needs exactly ONE physical cue and no presentation-attack
// evidence against it:
//   * the flash retroreflection test passed on an HSRP plate, UNIFORMLY across the plate (a bright
//     spot on glass is reported as `specular_hotspot` and counts against).
// Two things that used to count as proof no longer do, because a REAL monitor photo defeated them:
//   * parallax is only listed as SUPPORTING evidence (a screen held in front of a real office
//     background also shows depth);
//   * a LOW learned-recapture (PadNet) score is never positive evidence: it scored a photo of a
//     monitor 0.003 ("genuine"). PadNet can only count AGAINST a bundle.
// Screen / interface text in the scene (`screenContext`, from the OCR) adds to the
// presentation-attack score and makes the bundle a `suspected_spoof` from 1.0 upwards.
//
// Differences from the Python (all on the safe side or forced by the platform):
//  * pixel forensics (ELA / noise / moire / banding) are NOT available on device: 'forensics' is
//    listed in not_run with that reason, never silently a pass;
//  * no C2PA verification, no attestation: assurance stays 'none';
//  * the flash result is computed by checks.dart `retroreflection` (SAD alignment, box rings),
//    not by the Python phase-correlation / mask-dilation implementation; the uniformity /
//    hot-spot rule on top of it is the same arithmetic as active.gain_uniformity, with the same
//    thresholds (see liveness_test.dart for the cross-check against Python on generated frames).
// All thresholds are PROVISIONAL and have only met toy physics and simulated attacks.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../checks.dart';
import '../engine/types.dart';
import 'enhance.dart' show percentiles;
import 'planarity.dart';

const _roles = ['primary', 'flash_on', 'angle', 'other'];

/// Mirrors fusion.REVIEW_AT / REJECT_AT, bundle.SCREEN_SPOOF_AT and bundle's min planarity baseline.
const reviewAt = 1.0, rejectAt = 3.0, screenSpoofAt = 1.0, minBaselinePx = 1.5;

/// Mirrors active.MAX_GAIN_SPREAD / HOTSPOT_P90_OVER_P50 / HOTSPOT_MAX_MEDIAN_GAIN / GAIN_MIN_PIXELS
/// and retroreflection()'s min_ratio / max_ratio. All PROVISIONAL.
const maxGainSpread = 2.5, hotspotP90OverP50 = 3.0, hotspotMaxMedianGain = 2.0, gainMinPixels = 50;
const retroMinRatio = 2.5, retroMaxRatio = 1.4;

/// Hard provenance findings that say the file is not a clean capture but do NOT claim a
/// re-photographed / generated image. Every other hard finding is a declared screen / print / AI
/// origin, i.e. evidence the scene was not live. (Dart's scanProvenance never emits these three;
/// kept so a richer scanner can.)
const _hardNotASpoofClaim = {'undecodable', 'c2pa_unreadable', 'c2pa_invalid'};

class LivenessFrame {
  const LivenessFrame({required this.role, this.name = '', this.provenance, this.recapture});

  /// primary | flash_on | angle | other
  final String role;
  final String name;

  /// Findings from `scanProvenance(fileBytes)`; null when the frame has no original file bytes.
  final List<Finding>? provenance;

  /// PadNet score in 0..1 for this frame, or null if the model did not run on it.
  final double? recapture;
}

class PlanarityOutcome {
  const PlanarityOutcome(this.name, this.result);
  final String name;
  final PlanarityResult result;
}

class LivenessDecision {
  LivenessDecision({
    required this.status,
    required this.liveness,
    this.assurance = 'none',
    this.reasons = const [],
    this.notRun = const [],
    this.positive = const [],
    this.paScore = 0,
    this.perFrame = const [],
    this.evidence = const {},
  });

  final String status; // no_flags | review | reject
  final String liveness; // confirmed | suspected_spoof | not_established
  final String assurance;
  final List<String> reasons, notRun, positive;
  final double paScore;
  final List<Map<String, dynamic>> perFrame;
  final Map<String, dynamic> evidence;

  /// Same keys the app's `Authenticity.fromJson` reads (and the backend's BundleDecision.as_dict).
  Map<String, dynamic> toJson() => {
        'status': status,
        'liveness': liveness,
        'assurance': assurance,
        'reasons': reasons,
        'not_run': notRun,
        'positive': positive,
        'pa_score': paScore,
        'per_frame': perFrame,
        'evidence': evidence,
      };
}

// ------------------------------------------------------------------ read consistency
/// Could [a] and [b] be two reads of the same identifier? '?' matches any single character and may
/// also stand for nothing (the decoder inserts '?' at suspected dropped characters), but a known
/// character can never be added, dropped or changed. Port of bundle._compatible.
bool readsCompatible(String a, String b) {
  final n = a.length, m = b.length;
  final ok = [for (var i = 0; i <= n; i++) List<bool>.filled(m + 1, false)];
  ok[0][0] = true;
  for (var i = 0; i <= n; i++) {
    for (var j = 0; j <= m; j++) {
      if (i == 0 && j == 0) continue;
      var v = false;
      if (i > 0 && j > 0 && (a[i - 1] == b[j - 1] || a[i - 1] == '?' || b[j - 1] == '?')) v = ok[i - 1][j - 1];
      if (!v && i > 0 && a[i - 1] == '?') v = ok[i - 1][j];
      if (!v && j > 0 && b[j - 1] == '?') v = ok[i][j - 1];
      ok[i][j] = v;
    }
  }
  return ok[n][m];
}

/// Human-readable conflicts between reads of the same plate taken from different frames
/// (port of bundle.read_conflicts).
List<String> readConflicts(List<String?> readings, [List<String>? labels]) {
  final clean = <(int, String)>[
    for (var i = 0; i < readings.length; i++)
      if (readings[i] != null && readings[i]!.trim().isNotEmpty) (i, readings[i]!.toUpperCase().split(RegExp(r'\s+')).join()),
  ];
  final out = <String>[];
  for (var x = 0; x < clean.length; x++) {
    for (var y = x + 1; y < clean.length; y++) {
      final (i, a) = clean[x];
      final (j, b) = clean[y];
      if (!readsCompatible(a, b)) {
        final la = labels != null && i < labels.length ? labels[i] : 'read$i';
        final lb = labels != null && j < labels.length ? labels[j] : 'read$j';
        out.add("plate text differs between frames: '$a' ($la) vs '$b' ($lb)");
      }
    }
  }
  return out;
}

// ----------------------------------------------------------------- flash test wrapper
/// Plate mask for the flash test: the ROI eroded by [erode] on every side (bundle._plate_mask).
Box erodedPlateBox((int, int, int, int) roi, {double erode = 0.10}) {
  final (x0, y0, x1, y1) = roi;
  final dx = pyRound((x1 - x0) * erode), dy = pyRound((y1 - y0) * erode);
  return Box(x0 + dx, y0 + dy, x1 - dx, y1 - dy);
}

/// [mov] at [ref]'s size, or null when their aspect ratios differ by more than 3% (portrait vs
/// landscape: comparing those would measure nothing). Bilinear resize. Port of bundle._match_size.
RgbImage? matchSize(RgbImage ref, RgbImage mov) {
  if (mov.width == ref.width && mov.height == ref.height) return mov;
  if (((ref.width / ref.height) / (mov.width / mov.height) - 1.0).abs() > 0.03) return null;
  final w = ref.width, h = ref.height;
  final out = Uint8List(w * h * 3);
  final rx = w > 1 ? (mov.width - 1) / (w - 1) : 0.0, ry = h > 1 ? (mov.height - 1) / (h - 1) : 0.0;
  for (var y = 0; y < h; y++) {
    final fy = y * ry;
    final y0 = math.min(mov.height - 1, fy.floor()), y1 = math.min(mov.height - 1, y0 + 1);
    final wy = fy - y0;
    for (var x = 0; x < w; x++) {
      final fx = x * rx;
      final x0 = math.min(mov.width - 1, fx.floor()), x1 = math.min(mov.width - 1, x0 + 1);
      final wx = fx - x0;
      for (var c = 0; c < 3; c++) {
        double px(int xx, int yy) => mov.rgb[(yy * mov.width + xx) * 3 + c].toDouble();
        final v = (px(x0, y0) * (1 - wx) + px(x1, y0) * wx) * (1 - wy) + (px(x0, y1) * (1 - wx) + px(x1, y1) * wx) * wy;
        out[(y * w + x) * 3 + c] = v.round().clamp(0, 255);
      }
    }
  }
  return RgbImage(w, h, out);
}

/// The flash test's verdict (port of active.RetroResult): checks.dart's `RetroResult` has no room for
/// the uniformity result, so the bundle works with this one.
class FlashTest {
  const FlashTest(this.verdict, {this.ratio, this.gainPlate, this.gainSurround, this.reasons = const [], this.gainSpread, this.hotspot = false});

  /// retroreflective | not_retroreflective | specular_hotspot | inconclusive
  final String verdict;
  final double? ratio, gainPlate, gainSurround;
  final List<String> reasons;

  /// p90/p10 of the per-pixel flash gain over the plate background (1.0 = perfectly uniform)
  final double? gainSpread;

  /// the gain is concentrated in a small bright spot (glare on glass), not plate-wide
  final bool hotspot;

  static FlashTest fromRetro(RetroResult r) =>
      FlashTest(r.verdict, ratio: r.ratio, gainPlate: r.gainPlate, gainSurround: r.gainSurround, reasons: r.reasons);

  Map<String, dynamic> toJson() => {
        'verdict': verdict,
        'ratio': ratio,
        'gain_plate': gainPlate,
        'gain_surround': gainSurround,
        'reasons': reasons,
        'gain_spread': gainSpread,
        'hotspot': hotspot,
      };
}

double? _r3(double? v) => v == null ? null : _r(v, 3);

/// How evenly the flash brightened the plate (port of active.gain_uniformity). Retroreflective
/// sheeting returns the light straight back, so the WHOLE white background gains about the same
/// factor; a flash reflected off glass lights a small spot. Over the plate pixels that are bright
/// without flash (the white background, not the dark characters): G = (on + eps) / (off + eps);
/// returns (p90/p10, hotspot) with hotspot = p90/p50 >= 3 and median G < 2; (null, false) with too
/// few pixels. [dx], [dy]: where the content of [off] sits in [on] (from `alignShift`).
(double?, bool) flashGainUniformity(Rgb off, Rgb on, Box plate, {int dx = 0, int dy = 0, double eps = 1e-3}) {
  final lo = <double>[], ln = <double>[];
  for (var y = math.max(0, plate.y0); y < math.min(off.h, plate.y1); y++) {
    for (var x = math.max(0, plate.x0); x < math.min(off.w, plate.x1); x++) {
      final ox = x + dx, oy = y + dy;
      if (ox < 0 || oy < 0 || ox >= on.w || oy >= on.h) continue;
      lo.add(lum(off, x, y));
      ln.add(lum(on, ox, oy));
    }
  }
  if (lo.isEmpty) return (null, false);
  final median = percentiles(lo, [50])[0];
  final g = <double>[
    for (var i = 0; i < lo.length; i++)
      if (lo[i] >= 0.5 * median) (ln[i] + eps) / (lo[i] + eps),
  ];
  if (g.length < gainMinPixels) return (null, false);
  final p = percentiles(g, [10, 50, 90]);
  return (p[2] / p[0], p[2] / p[1] >= hotspotP90OverP50 && p[1] < hotspotMaxMedianGain);
}

/// Whole-pixel shift (dx, dy) that lines [on] up with [off] ("content of [off] at p sits at p + d in
/// [on]"), for the per-pixel gain map. checks.dart's `alignShift` compares luminance directly over the
/// whole frame: a flash that brightens the scene biases it towards darker content, and the plate itself
/// (the one thing that changes completely under flash) pulls it around - on a toy scene with NO motion it
/// returned shifts of 4..20 px, which smears the gain map. This one
///  * compares log-luminance with the mean removed (a multiplicative exposure / flash change becomes an
///    offset; the Python's phase correlation is gain-invariant for the same reason),
///  * leaves the plate (grown by one block) out of the comparison,
///  * scores a shift by the mean of the smallest half of the block differences after re-centring them on their
///    median (robust to a region that changed for another reason, e.g. glare beside the plate),
///  * works on a window around the plate, on 4x4 blocks, +-[search] blocks, and starts from "no motion":
///    it moves only for a clearly better match.
/// Returns (0, 0) when the window is too small to search.
(int, int) alignForGain(Rgb off, Rgb on, Box plate, {int search = 5}) {
  const f = 4;
  final margin = math.max(plate.x1 - plate.x0, plate.y1 - plate.y0);
  final wx0 = math.max(0, plate.x0 - margin) ~/ f * f, wy0 = math.max(0, plate.y0 - margin) ~/ f * f;
  final wx1 = math.min(math.min(off.w, on.w), plate.x1 + margin), wy1 = math.min(math.min(off.h, on.h), plate.y1 + margin);
  final bw = (wx1 - wx0) ~/ f, bh = (wy1 - wy0) ~/ f;
  if (bw < 2 * search + 4 || bh < 2 * search + 4) return (0, 0);
  // blocks that overlap the plate, grown by one block
  final masked = List<bool>.filled(bw * bh, false);
  for (var y = 0; y < bh; y++) {
    for (var x = 0; x < bw; x++) {
      final x0 = wx0 + x * f, y0 = wy0 + y * f;
      masked[y * bw + x] = x0 + 2 * f > plate.x0 && x0 - f < plate.x1 && y0 + 2 * f > plate.y0 && y0 - f < plate.y1;
    }
  }
  List<double> blocks(Rgb im) {
    final o = List<double>.filled(bw * bh, 0);
    var sum = 0.0, n = 0;
    for (var y = 0; y < bh; y++) {
      for (var x = 0; x < bw; x++) {
        var m = 0.0;
        for (var j = 0; j < f; j++) {
          for (var i = 0; i < f; i++) {
            m += lum(im, wx0 + x * f + i, wy0 + y * f + j);
          }
        }
        final v = math.log(m / (f * f) + 1e-3);
        o[y * bw + x] = v;
        if (!masked[y * bw + x]) {
          sum += v;
          n++;
        }
      }
    }
    if (n == 0) return o;
    final mean = sum / n;
    for (var i = 0; i < o.length; i++) {
      o[i] -= mean;
    }
    return o;
  }

  final a = blocks(off), b = blocks(on);
  // The score of a shift: the signed differences are first re-centred on their MEDIAN (a region that
  // changed for another reason - the flash glaring off glass beside the plate, a lit patch of bodywork -
  // would otherwise leave a global offset behind), then the score is the mean of the SMALLEST HALF of the
  // absolute deviations. Neither can be pulled by a region covering less than half of the window.
  // (Histograms, no sorting.)
  const nsb = 512, nsLo = -4.0, nsW = 8.0 / nsb; // signed differences in [-4, 4)
  const nab = 256, naW = 2.0 / nab; // absolute deviations in [0, 2)
  final sc = List<int>.filled(nsb, 0);
  final ac = List<int>.filled(nab, 0);
  final asum = List<double>.filled(nab, 0);
  double scoreAt(int dx, int dy) {
    sc.fillRange(0, nsb, 0);
    var n = 0;
    for (var y = search; y < bh - search; y++) {
      for (var x = search; x < bw - search; x++) {
        final ia = y * bw + x, ib = (y + dy) * bw + (x + dx);
        if (masked[ia] || masked[ib]) continue;
        sc[math.max(0, math.min(nsb - 1, ((a[ia] - b[ib] - nsLo) / nsW).floor()))]++;
        n++;
      }
    }
    if (n < 20) return double.infinity;
    var acc = 0, mk = 0;
    while (mk < nsb - 1 && acc + sc[mk] < n / 2) {
      acc += sc[mk++];
    }
    final med = nsLo + (mk + 0.5) * nsW;
    ac.fillRange(0, nab, 0);
    asum.fillRange(0, nab, 0.0);
    for (var y = search; y < bh - search; y++) {
      for (var x = search; x < bw - search; x++) {
        final ia = y * bw + x, ib = (y + dy) * bw + (x + dx);
        if (masked[ia] || masked[ib]) continue;
        final d = (a[ia] - b[ib] - med).abs();
        final k = math.min(nab - 1, (d / naW).floor());
        ac[k]++;
        asum[k] += d;
      }
    }
    var left = n / 2, total = 0.0;
    for (var k = 0; k < nab && left > 0; k++) {
      if (ac[k] == 0) continue;
      final take = math.min(left, ac[k].toDouble());
      total += asum[k] * (take / ac[k]);
      left -= take;
    }
    return total / (n / 2);
  }

  // start from "no motion" and move only for a clearly better match
  var best = scoreAt(0, 0);
  var bx = 0, by = 0;
  for (var dy = -search; dy <= search; dy++) {
    for (var dx = -search; dx <= search; dx++) {
      if (dx == 0 && dy == 0) continue;
      final score = scoreAt(dx, dy);
      if (score < best * 0.98 - 1e-9) {
        best = score;
        bx = dx;
        by = dy;
      }
    }
  }
  return (bx * f, by * f);
}

/// checks.dart's `retroreflection` plus the uniformity / hot-spot rule of active.retroreflection:
///  * guards (non-HSRP, too small, no flash, saturated, too dark, no headroom) -> unchanged;
///  * a hot-spot -> `specular_hotspot` (glare on a glossy surface), whatever the ratio;
///  * ratio >= 2.5 -> `retroreflective` only when the gain is uniform (p90/p10 <= 2.5), otherwise
///    `inconclusive` with a reason: it is not accepted as proof;
///  * ratio <= 1.4 -> `not_retroreflective`; in between -> `inconclusive` (grey zone).
FlashTest judgeFlash(Rgb off, Rgb on, Box plate, {required bool hsrpExpected, bool align = true}) {
  final base = retroreflection(off, on, plate, hsrpExpected: hsrpExpected, align: align);
  final baseRatio = base.ratio;
  if (baseRatio == null) return FlashTest.fromRetro(base); // a guard fired: nothing was measured
  var dx = 0, dy = 0;
  if (align) (dx, dy) = alignForGain(off, on, plate);
  final (spread, hotspot) = flashGainUniformity(off, on, plate, dx: dx, dy: dy);
  final ratio = _r3(baseRatio)!; // Python compares the rounded ratio
  FlashTest make(String verdict, List<String> reasons) => FlashTest(verdict,
      ratio: ratio, gainPlate: _r3(base.gainPlate), gainSurround: _r3(base.gainSurround), reasons: reasons, gainSpread: _r3(spread), hotspot: hotspot);
  if (hotspot) {
    // A bright spot on an otherwise unchanged plate is a reflection of the flash on a glossy surface (the
    // glass of a screen, a laminated print), not the plate-wide return of retroreflective sheeting.
    return make('specular_hotspot', [
      'the flash lit a small hot-spot on the plate while the rest barely changed: that is glare on a '
          'glossy surface (a screen or print), not the plate-wide return of retroreflective sheeting'
    ]);
  }
  if (ratio >= retroMinRatio) {
    if (spread == null || spread <= maxGainSpread) return make('retroreflective', const []);
    return make('inconclusive', [
      'the flash gain is not uniform across the plate (p90/p10 = ${spread.toStringAsFixed(1)} > $maxGainSpread): '
          'retroreflective sheeting brightens evenly, so this is not accepted as proof'
    ]);
  }
  if (ratio <= retroMaxRatio) return make('not_retroreflective', base.reasons);
  return make('inconclusive', const ['ratio in the grey zone between retroreflective and not']);
}

/// Flash-off vs flash-on for one bundle: returns (result, reasonItWasNotRun). Wraps [judgeFlash]
/// with the bundle's mask rule and size matching.
(FlashTest?, String?) runFlashTest(RgbImage primary, RgbImage flashOn, (int, int, int, int)? roi, {required bool hsrpExpected}) {
  if (roi == null) return (null, 'no plate region (ROI) for the primary frame, so no plate mask can be built');
  final on = matchSize(primary, flashOn);
  if (on == null) return (null, 'flash-on frame has a different aspect ratio than the primary frame');
  Rgb conv(RgbImage i) => Rgb(i.width, i.height, i.rgb);
  return (judgeFlash(conv(primary), conv(on), erodedPlateBox(roi), hsrpExpected: hsrpExpected), null);
}

// --------------------------------------------------------------------------- decision
String _pyList(List<String> v) => '[${v.map((s) => "'$s'").join(', ')}]';
double _r(double v, int p) {
  final m = math.pow(10, p).toDouble();
  return (v * m).roundToDouble() / m;
}

/// Python's `format(v, 'g')`: 6 significant digits, no trailing zeros (1.0 -> '1', 1.5 -> '1.5').
String _g(double v) {
  final t = v.toStringAsPrecision(6);
  if (!t.contains('.') || t.contains('e')) return t;
  return t.replaceFirst(RegExp(r'0+$'), '').replaceFirst(RegExp(r'\.$'), '');
}

const _order = {'no_flags': 0, 'review': 1, 'reject': 2};
String _bump(String s, String to) => _order[to]! > _order[s]! ? to : s;

/// The on-device authenticity decision. See the library comment for inputs and semantics.
///
/// [retro]: the flash test result for the first `flash_on` frame vs the primary (null = not run;
/// then [retroSkipReason] says why). [planarity]: one outcome per angle frame that was compared;
/// [skippedAngleFrames]: angle frames not compared (aspect ratio differs). [readings]: OCR reads of
/// the plate (may contain '?'), one per frame or any list, for the consistency check.
/// [screenContext]: the scene reader's `SceneResult.screenContext` (interface text visible in the
/// photos: keys score / url / ui_words / text_lines / reasons), or null.
LivenessDecision assessLiveness({
  required List<LivenessFrame> frames,
  FlashTest? retro,
  String? retroSkipReason,
  List<PlanarityOutcome> planarity = const [],
  List<String> skippedAngleFrames = const [],
  bool hsrpExpected = false,
  bool requireLiveness = false,
  List<String?>? readings,
  Map<String, dynamic>? screenContext,
}) {
  if (frames.isEmpty) throw ArgumentError('assessLiveness needs at least one frame');
  for (final f in frames) {
    if (!_roles.contains(f.role)) throw ArgumentError('unknown frame role ${f.role}; expected one of $_roles');
  }
  final names = [for (var i = 0; i < frames.length; i++) frames[i].name.isEmpty ? 'frame$i' : frames[i].name];
  final pIdx = math.max(0, frames.indexWhere((f) => f.role == 'primary'));
  final notes = <String>[];
  final notRunWhy = <String, String>{};

  // ---- per-frame passive results --------------------------------------------------
  // provenance: the same finding in several frames is listed once and tagged with every frame
  final seen = <String, (Finding, List<String>)>{};
  for (var i = 0; i < frames.length; i++) {
    for (final f in frames[i].provenance ?? const <Finding>[]) {
      final key = '${f.code}\u0000${f.severity}\u0000${f.detail}';
      seen.putIfAbsent(key, () => (f, <String>[])).$2.add(names[i]);
    }
  }
  final merged = [for (final e in seen.values) Finding(e.$1.code, e.$1.severity, '[${e.$2.join(', ')}] ${e.$1.detail}')];
  final hard = merged.where((f) => f.severity == 'hard').toList();
  final soft = merged.where((f) => f.severity == 'soft').toList();

  final recapture = <String, double>{};
  for (var i = 0; i < frames.length; i++) {
    final s = frames[i].recapture;
    if (s != null) recapture[names[i]] = s;
  }
  double? modelRecapture;
  if (recapture.isEmpty) {
    notRunWhy['recapture'] = 'no recapture model loaded';
  } else {
    final worst = recapture.entries.reduce((a, b) => b.value > a.value ? b : a);
    modelRecapture = worst.value;
    if (worst.value >= 0.5) notes.add("highest learned recapture score is in frame '${worst.key}' (${worst.value.toStringAsFixed(2)})");
  }
  notRunWhy['forensics'] = 'pixel forensics not available on device';

  // ---- flash: retroreflection -----------------------------------------------------
  final flashIdx = [for (var i = 0; i < frames.length; i++) if (frames[i].role == 'flash_on') i];
  FlashTest? activeRetro;
  if (flashIdx.isNotEmpty) {
    if (flashIdx.length > 1) notes.add("${flashIdx.length} flash-on frames supplied; only '${names[flashIdx.first]}' was used");
    if (retro == null) {
      notRunWhy['retroreflection'] = retroSkipReason ?? 'the flash test was not computed';
    } else {
      activeRetro = retro;
      if (retro.verdict == 'inconclusive') {
        notRunWhy['retroreflection'] = retro.reasons.isEmpty ? 'inconclusive' : retro.reasons.join('; ');
      }
    }
  } else {
    notRunWhy['retroreflection'] = 'no flash-on frame in the bundle';
  }

  // ---- angles: parallax / planarity -----------------------------------------------
  final angleIdx = [for (var i = 0; i < frames.length; i++) if (frames[i].role == 'angle') i];
  final conclusive = [for (final p in planarity) if (p.result.verdict == 'planar' || p.result.verdict == 'non_planar') p];
  final nonPlanar = [for (final p in conclusive) if (p.result.verdict == 'non_planar' && (p.result.baselinePx ?? 0.0) >= minBaselinePx) p];
  PlanarityResult? activePlan;
  if (nonPlanar.isNotEmpty) {
    activePlan = nonPlanar.first.result;
  } else if (conclusive.isNotEmpty && conclusive.every((p) => p.result.verdict == 'planar')) {
    activePlan = conclusive.first.result; // ONE weak cue however many angles agree
  } else if (conclusive.isNotEmpty) {
    activePlan = const PlanarityResult('inconclusive', reasons: ['angle frames gave conflicting or ambiguous results']);
  } else if (planarity.isNotEmpty) {
    activePlan = planarity.first.result;
  }
  if (activePlan == null || activePlan.verdict == 'inconclusive') {
    if (angleIdx.isEmpty) {
      notRunWhy['planarity'] = 'no angle frames in the bundle';
    } else if (skippedAngleFrames.isNotEmpty && planarity.isEmpty) {
      notRunWhy['planarity'] = 'angle frames ${_pyList(skippedAngleFrames)} differ in aspect ratio from the primary';
    } else {
      final why = [for (final p in planarity) if (p.result.reasons.isNotEmpty) p];
      notRunWhy['planarity'] = why.isNotEmpty ? why.first.result.reasons.join('; ') : 'inconclusive';
    }
  }
  if (skippedAngleFrames.isNotEmpty && planarity.isNotEmpty) {
    notes.add('angle frames ${_pyList(skippedAngleFrames)} skipped: aspect ratio differs from the primary');
  }

  // ---- consistency of the reads across frames -------------------------------------
  var conflicts = <String>[];
  if (readings != null && readings.isNotEmpty) {
    conflicts = readConflicts(readings, readings.length == frames.length ? names : null);
  }

  // ---- fuse (port of fusion.decide) -----------------------------------------------
  final reasons = <String>[];
  var reject = false, review = false;
  for (final f in hard) {
    reject = true;
    reasons.add('${f.code}: ${f.detail}');
  }
  for (final f in soft) {
    review = true;
    reasons.add('${f.code}: ${f.detail}');
  }
  var pa = 0.0;
  final parts = <String>[];
  if (modelRecapture != null) {
    final w = modelRecapture >= 0.9 ? 2.0 : (modelRecapture >= 0.5 ? 1.0 : 0.0);
    if (w > 0) {
      pa += w;
      parts.add('learned recapture score ${modelRecapture.toStringAsFixed(2)} (+${w.toStringAsFixed(1)})');
    }
  }
  final retroV = activeRetro?.verdict;
  final positive = <String>[];
  if (retroV == 'not_retroreflective') {
    pa += 2.0;
    parts.add('plate did not retroreflect under flash (+2.0)');
  } else if (retroV == 'specular_hotspot') {
    pa += 1.0;
    parts.add('plate showed a specular hot-spot under flash, not a retroreflective response (+1.0)');
  } else if (retroV == 'retroreflective') {
    positive.add('retroreflection_passed');
  }
  final ctxScore = (screenContext?['score'] as num?)?.toDouble() ?? 0.0;
  if (ctxScore > 0.0) {
    pa += ctxScore;
    final ctxReasons = [for (final r in (screenContext?['reasons'] as List?) ?? const []) '$r'];
    parts.add('screen / interface context in the scene (+${_g(ctxScore)}): ${ctxReasons.join('; ')}');
  }
  if (activePlan?.verdict == 'planar') {
    pa += 0.5;
    parts.add('scene motion fits a single plane (+0.5)');
  }
  if (pa >= rejectAt) {
    reject = true;
    reasons.add('presentation attack suspected, score ${pa.toStringAsFixed(1)} >= ${rejectAt.toStringAsFixed(1)}: ${parts.join('; ')}');
  } else if (pa >= reviewAt) {
    review = true;
    reasons.add('possible presentation attack, score ${pa.toStringAsFixed(1)}: ${parts.join('; ')}');
  } else if (parts.isNotEmpty) {
    reasons.add('weak presentation-attack hints only: ${parts.join('; ')}');
  }
  reasons.addAll(notes);
  var status = reject ? 'reject' : (review ? 'review' : 'no_flags');
  if (conflicts.isNotEmpty) {
    status = _bump(status, 'review');
    reasons.addAll(conflicts);
  }

  final notRun = <String>[
    if (modelRecapture == null) 'recapture',
    'tamper', // no trained text-edit / AI-image detectors exist on or off device
    'ai_generated',
    if (activeRetro == null || activeRetro.verdict == 'inconclusive') 'retroreflection',
    if (activePlan == null || activePlan.verdict == 'inconclusive') 'planarity',
    'forensics',
  ];

  // ---- liveness: positive physical evidence only ----------------------------------
  final basis = <String>[];
  if (retroV == 'retroreflective' && hsrpExpected) basis.add('retroreflection_passed');
  if (nonPlanar.isNotEmpty) {
    // supporting evidence ONLY: a screen held in front of a real background also shows depth
    positive.add('parallax_non_planar');
  }
  final spoofClaim = hard.any((f) => !_hardNotASpoofClaim.contains(f.code));
  final String liveness;
  if (pa >= reviewAt || retroV == 'not_retroreflective' || retroV == 'specular_hotspot' || ctxScore >= screenSpoofAt || spoofClaim) {
    liveness = 'suspected_spoof';
  } else if (basis.isNotEmpty && status != 'reject') {
    liveness = 'confirmed';
  } else {
    liveness = 'not_established';
  }

  if (liveness == 'not_established') {
    final missing = <String>[
      if (notRunWhy.containsKey('retroreflection')) 'retroreflection not run (${notRunWhy['retroreflection']})',
      if (notRunWhy.containsKey('planarity')) 'parallax not run (${notRunWhy['planarity']})',
    ];
    if (status == 'no_flags') {
      reasons.add('no flags were raised, but there is NO positive evidence of a live scene: that is not liveness. ${missing.join('; ')}');
    }
  }
  if (nonPlanar.isNotEmpty && liveness != 'suspected_spoof') {
    reasons.add('the angle frames show depth, but depth alone does not establish liveness (a screen held in front of a real '
        'background also shows depth); only the flash retroreflection test can confirm it');
  }
  if (recapture.isNotEmpty && recapture.values.reduce(math.max) < 0.5) {
    reasons.add('the learned screen / print model scored this bundle clean, but a clean score is never used as proof of a live '
        'scene: it was trained on simulated attacks only and has scored a real monitor photo as genuine');
  }
  if (requireLiveness && liveness != 'confirmed') {
    status = _bump(status, 'review');
    reasons.add('liveness required but ${liveness.replaceAll('_', ' ')}');
  }

  return LivenessDecision(
    status: status,
    liveness: liveness,
    assurance: 'none',
    reasons: reasons.toSet().toList(),
    notRun: notRun,
    positive: positive.toSet().toList(),
    paScore: _r(pa, 2),
    perFrame: [
      for (var i = 0; i < frames.length; i++)
        {
          'index': i,
          'name': names[i],
          'role': frames[i].role,
          'recapture': frames[i].recapture == null ? null : _r(frames[i].recapture!, 3),
          'provenance': {
            'hard': [for (final f in frames[i].provenance ?? const <Finding>[]) if (f.severity == 'hard') f.code],
            'soft': [for (final f in frames[i].provenance ?? const <Finding>[]) if (f.severity == 'soft') f.code],
          },
        }
    ],
    evidence: {
      'liveness_basis': basis,
      'recapture_by_frame': {for (final e in recapture.entries) e.key: _r(e.value, 3)},
      'planarity_by_frame': {for (final p in planarity) p.name: p.result.toJson()},
      'read_conflicts': conflicts,
      'not_run_reasons': notRunWhy,
      'primary_frame': names[pIdx],
      'hsrp_expected': hsrpExpected,
      'screen_context': screenContext,
      'calibrated': false,
      'on_device': true,
      'active': {
        if (activeRetro != null) 'retroreflection': activeRetro.toJson(),
        if (activePlan != null) 'planarity': activePlan.toJson(),
      },
    },
  );
}
