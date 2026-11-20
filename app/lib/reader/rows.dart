// Split a rectified plate into its printed rows. Port of `ml/vinplate/geometry/rows.py`. Pure Dart.

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/image_ops.dart' show cropRgb;
import '../engine/types.dart';
import 'enhance.dart' show reflectIndex;

/// Width / height of a rectified plate: 1-row ~4.2, 2-row ~1.6-2.2.
const twoRowMaxAspect = 2.6;

bool looksTwoRow(int h, int w) => w / h < twoRowMaxAspect;

/// Cut a 2-row plate at the quietest horizontal band between the rows (smallest vertical gradient
/// energy in the middle 30%), with [overlap] of the plate height shared by both crops so a character
/// that straddles the cut is not lost. Returns [top, bottom].
List<RgbImage> splitRows(RgbImage crop, {double overlap = 0.12}) {
  final w = crop.width, h = crop.height;
  // luma in float64 (uint8 * python float), stored as float32 like `.astype(np.float32)`
  final g = Float32List(w * h);
  for (var i = 0; i < w * h; i++) {
    g[i] = crop.rgb[3 * i] * 0.299 + crop.rgb[3 * i + 1] * 0.587 + crop.rgb[3 * i + 2] * 0.114;
  }
  // mean absolute vertical difference per row boundary, then a uniform (box) filter of the given size
  final diffs = Float64List(math.max(0, h - 1));
  for (var y = 0; y < h - 1; y++) {
    var s = 0.0;
    for (var x = 0; x < w; x++) {
      s += (g[(y + 1) * w + x] - g[y * w + x]).abs();
    }
    diffs[y] = s / w;
  }
  final size = math.max(3, h ~/ 20);
  final lo = (0.35 * h).toInt(), hi = (0.65 * h).toInt();
  var cut = h ~/ 2;
  if (hi > lo && diffs.isNotEmpty) {
    double? best;
    for (var i = lo; i < hi && i < diffs.length; i++) {
      var s = 0.0;
      for (var j = 0; j < size; j++) {
        s += diffs[reflectIndex(i - size ~/ 2 + j, diffs.length)];
      }
      final e = s / size;
      if (best == null || e < best) {
        best = e;
        cut = i;
      }
    }
  }
  final m = (overlap * h).toInt();
  return [cropRgb(crop, 0, 0, w, math.min(h, cut + m)), cropRgb(crop, 0, math.max(0, cut - m), w, h)];
}
