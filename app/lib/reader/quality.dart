// Capture-quality gate: cheap checks that say "retake" (and what to change) before or alongside a
// reading. Port of `ml/vinplate/quality/gate.py`. The thresholds are PROVISIONAL (chosen from first
// principles and eyeballed on synthetic crops); they have not been calibrated on real captures, so
// treat the verdict as guidance, not as a hard gate. Pure Dart, isolate-safe.

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/image_ops.dart' show GrayImage, PilFilter, resizePilGray;
import '../engine/types.dart';
import 'enhance.dart' show componentSizes4, gaussianFilter, laplace, percentiles;

const Map<String, double> qualityThresholds = {
  'min_height_px': 24, // glyph rows below this cannot be read reliably
  'min_sharpness': 4.0, // re-blur ratio on the 48-px-high normalised crop
  'max_glare_frac': 0.06, // saturated pixels
  'max_glare_blob': 0.025, // largest connected saturated blob, as a fraction of the crop area
  'min_mean': 40.0,
  'max_mean': 215.0,
  'min_contrast': 0.18, // (p95 - p5) / 255
  'max_clipped_dark': 0.25,
};

class QualityReport {
  const QualityReport({required this.metrics, required this.issues, required this.guidance});
  final Map<String, double> metrics;
  final List<String> issues, guidance;

  bool get ok => issues.isEmpty;

  Map<String, dynamic> toJson() => {'ok': ok, 'metrics': metrics, 'issues': issues, 'guidance': guidance, 'calibrated': false};
}

double _round(double v, int digits) {
  final f = math.pow(10, digits).toDouble();
  return (v * f).roundToDouble() / f;
}

double _variance(Float64List a) {
  var mean = 0.0;
  for (final v in a) {
    mean += v;
  }
  mean /= a.length;
  var s = 0.0;
  for (final v in a) {
    s += (v - mean) * (v - mean);
  }
  return s / a.length;
}

/// [crop]: an upright crop of ONE plate / VIN region (the output of the text detector's warp).
QualityReport assessCrop(RgbImage crop, {Map<String, double>? thresholds}) {
  final t = {...qualityThresholds, ...?thresholds};
  final w = crop.width, h = crop.height;
  // gate.py computes this on the uint8 array: float64 arithmetic, THEN a cast to float32 (unlike
  // enhance.py, which converts to float32 first). The cast is what assigning into a Float32List does.
  final lum = Float32List(w * h);
  for (var i = 0; i < lum.length; i++) {
    lum[i] = crop.rgb[3 * i] * 0.299 + crop.rgb[3 * i + 1] * 0.587 + crop.rgb[3 * i + 2] * 0.114;
  }
  // astype(uint8) truncates; a 48-px-high normalised crop makes sharpness independent of crop size
  final u8 = Uint8List(w * h);
  for (var i = 0; i < u8.length; i++) {
    final v = lum[i];
    u8[i] = v <= 0 ? 0 : (v >= 255 ? 255 : v.toInt());
  }
  final nw = math.max(8, (w * (48.0 / h)).toInt());
  final gnU8 = resizePilGray(GrayImage(w, h, u8), nw, 48, PilFilter.bilinear).data;
  final n = gnU8.length;
  final gn = Float64List(n);
  for (var i = 0; i < n; i++) {
    gn[i] = gnU8[i].toDouble();
  }

  // No-reference sharpness by re-blur: a sharp image loses most of its Laplacian energy when blurred
  // a little more; an already-blurry one barely changes.
  final sharp = _variance(laplace(gn, nw, 48)) / (_variance(laplace(gaussianFilter(gn, nw, 48, 1.2), nw, 48)) + 1e-6);
  final sat = Uint8List(n);
  var satCount = 0, dark = 0;
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    if (gn[i] >= 250) {
      sat[i] = 1;
      satCount++;
    }
    if (gn[i] <= 5) dark++;
    sum += gn[i];
  }
  final glareFrac = satCount / n;
  final sizes = componentSizes4(sat, nw, 48);
  final blob = sizes.isEmpty ? 0.0 : sizes.reduce(math.max) / n;
  final p = percentiles(gn, [5, 95]);
  final contrast = (p[1] - p[0]) / 255;
  final mean = sum / n;
  final darkFrac = dark / n;
  final m = <String, double>{
    'height_px': h.toDouble(),
    'sharpness': _round(sharp, 2),
    'glare_frac': _round(glareFrac, 4),
    'glare_blob': _round(blob, 4),
    'mean': _round(mean, 1),
    'contrast': _round(contrast, 3),
    'dark_clipped': _round(darkFrac, 4),
  };

  final issues = <String>[], tips = <String>[];
  if (h < t['min_height_px']!) {
    issues.add('too small');
    tips.add('Move closer so the plate/number fills more of the frame.');
  }
  if (sharp < t['min_sharpness']!) {
    issues.add('blurred');
    tips.add('Hold the phone steady and tap to focus.');
  }
  if (glareFrac > t['max_glare_frac']! || blob > t['max_glare_blob']!) {
    issues.add('glare');
    tips.add('Tilt the phone a little or step aside to move the reflection off the number.');
  }
  if (m['mean']! < t['min_mean']! || m['dark_clipped']! > t['max_clipped_dark']!) {
    issues.add('too dark');
    tips.add('Add light, or turn the torch on and move to avoid reflections.');
  }
  if (m['mean']! > t['max_mean']!) {
    issues.add('overexposed');
    tips.add('Reduce the light on the plate or avoid direct flash.');
  }
  if (contrast < t['min_contrast']! && !issues.contains('overexposed') && !issues.contains('too dark')) {
    issues.add('low contrast');
    tips.add('Clean the surface if possible and retake from a different angle.');
  }
  return QualityReport(metrics: m, issues: issues, guidance: tips);
}
