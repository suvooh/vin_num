// Text-box geometry for the on-device plate / VIN reader: the minimum-area rectangle of a blob and
// the upright perspective crop of a quad. Port of `ml/vinplate/ocr/ppocr.py` (min_area_rect,
// _homography_coeffs, warp_quad). Pure Dart, no Flutter imports: safe inside an isolate.
//
// Sampling: the crop is resampled the way Pillow's `Image.transform(PERSPECTIVE, BICUBIC)` does it
// (output pixel centre -> homography -> 4x4 cubic convolution with a = -1, edge clamped, result
// truncated to 8 bits). That was checked against Pillow on random data, so Dart and Python crops
// agree to within 1 grey level.

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/image_ops.dart' show rot90, roundHalfEven;
import '../engine/types.dart';

/// A detected text line. [quad] is TL, TR, BR, BL as x0,y0,x1,y1,x2,y2,x3,y3 (image pixels); the
/// long side is the width, so text reads left to right along TL -> TR.
class TextBox {
  const TextBox(this.quad, this.score); // quad must hold 8 numbers (TL, TR, BR, BL)
  final List<double> quad;
  final double score;

  double get width => math.sqrt(math.pow(quad[2] - quad[0], 2) + math.pow(quad[3] - quad[1], 2));
  double get height => math.sqrt(math.pow(quad[6] - quad[0], 2) + math.pow(quad[7] - quad[1], 2));
  (double, double) get center => ((quad[0] + quad[2] + quad[4] + quad[6]) / 4, (quad[1] + quad[3] + quad[5] + quad[7]) / 4);

  /// Direction of the text, degrees (0 = horizontal, positive = clockwise on screen).
  double get angle => math.atan2(quad[3] - quad[1], quad[2] - quad[0]) * 180 / math.pi;

  (int, int, int, int) get bbox {
    final xs = [quad[0], quad[2], quad[4], quad[6]], ys = [quad[1], quad[3], quad[5], quad[7]];
    return (xs.reduce(math.min).floor(), ys.reduce(math.min).floor(), xs.reduce(math.max).ceil(), ys.reduce(math.max).ceil());
  }
}

double _rint(double v) => roundHalfEven(v).toDouble();

double _cross(double ox, double oy, double ax, double ay, double bx, double by) => (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);

/// Convex hull (Andrew's monotone chain, collinear points dropped) of points sorted by (x, y).
List<(double, double)> _hull(List<(double, double)> p) {
  final lower = <(double, double)>[];
  for (final q in p) {
    while (lower.length >= 2 && _cross(lower[lower.length - 2].$1, lower[lower.length - 2].$2, lower.last.$1, lower.last.$2, q.$1, q.$2) <= 0) {
      lower.removeLast();
    }
    lower.add(q);
  }
  final upper = <(double, double)>[];
  for (final q in p.reversed) {
    while (upper.length >= 2 && _cross(upper[upper.length - 2].$1, upper[upper.length - 2].$2, upper.last.$1, upper.last.$2, q.$1, q.$2) <= 0) {
      upper.removeLast();
    }
    upper.add(q);
  }
  lower.removeLast();
  upper.removeLast();
  return [...lower, ...upper];
}

/// Minimum-area bounding rectangle of [points] (x0,y0,x1,y1,...). Returns the corners as TL, TR,
/// BR, BL (8 numbers) with the LONG side along x, plus the long and short side lengths.
/// Same algorithm as the Python reference: rotate to each hull-edge direction (mod 90 degrees,
/// rounded to 6 decimals) and keep the smallest axis-aligned box.
({List<double> corners, double long, double short}) minAreaRect(List<double> points) {
  final raw = <(double, double)>[for (var i = 0; i + 1 < points.length; i += 2) (points[i], points[i + 1])];
  raw.sort((a, b) => a.$1 != b.$1 ? a.$1.compareTo(b.$1) : a.$2.compareTo(b.$2));
  final pts = <(double, double)>[];
  for (final q in raw) {
    if (pts.isEmpty || pts.last.$1 != q.$1 || pts.last.$2 != q.$2) pts.add(q);
  }
  if (pts.isEmpty) return (corners: List<double>.filled(8, 0), long: 0, short: 0);
  var hull = pts;
  if (pts.length >= 3) {
    final h = _hull(pts);
    if (h.length >= 3) hull = h; // all collinear: keep the points, as Qhull's failure does in Python
  }
  final angles = <double>{};
  for (var i = 0; i < hull.length; i++) {
    final a = hull[i], b = hull[(i + 1) % hull.length];
    final ang = math.atan2(b.$2 - a.$2, b.$1 - a.$1) % (math.pi / 2);
    angles.add(_rint(ang * 1e6) / 1e6);
  }
  final sorted = (angles.isEmpty ? [0.0] : angles.toList())..sort();
  double bestArea = double.infinity, bc = 1, bs = 0, mn0 = 0, mn1 = 0, mx0 = 0, mx1 = 0;
  for (final a in sorted) {
    final c = math.cos(a), s = math.sin(a);
    var a0 = double.infinity, a1 = double.infinity, b0 = -double.infinity, b1 = -double.infinity;
    for (final p in hull) {
      final x = c * p.$1 + s * p.$2, y = -s * p.$1 + c * p.$2;
      if (x < a0) a0 = x;
      if (x > b0) b0 = x;
      if (y < a1) a1 = y;
      if (y > b1) b1 = y;
    }
    final area = (b0 - a0) * (b1 - a1);
    if (area < bestArea) {
      bestArea = area;
      bc = c;
      bs = s;
      mn0 = a0;
      mn1 = a1;
      mx0 = b0;
      mx1 = b1;
    }
  }
  var ux = bc, uy = bs, vx = -bs, vy = bc; // rows of the rotation: the rotated x / y axes in image space
  var long = mx0 - mn0, short = mx1 - mn1;
  final c0 = (mn0 + mx0) / 2, c1 = (mn1 + mx1) / 2;
  if (short > long) {
    final tx = ux, ty = uy;
    ux = vx;
    uy = vy;
    vx = tx;
    vy = ty;
    final t = long;
    long = short;
    short = t;
  }
  if (ux < 0) {
    ux = -ux;
    uy = -uy;
  }
  if (vy < 0) {
    vx = -vx;
    vy = -vy;
  }
  // centre back in image coordinates: centre @ R
  final cx = c0 * bc + c1 * -bs, cy = c0 * bs + c1 * bc;
  final hl = long / 2, hs = short / 2;
  final corners = <double>[
    cx - ux * hl - vx * hs, cy - uy * hl - vy * hs, // TL
    cx + ux * hl - vx * hs, cy + uy * hl - vy * hs, // TR
    cx + ux * hl + vx * hs, cy + uy * hl + vy * hs, // BR
    cx - ux * hl + vx * hs, cy - uy * hl + vy * hs, // BL
  ];
  return (corners: corners, long: long, short: short);
}

/// Pillow PERSPECTIVE coefficients (a..h) mapping OUTPUT points [dst] to INPUT points [src]
/// (each 8 numbers: 4 x,y pairs): x_in = (a x + b y + c) / (g x + h y + 1), y_in = (d x + e y + f) / (g x + h y + 1).
List<double> homographyCoeffs(List<double> dst, List<double> src) {
  final a = List.generate(8, (_) => List<double>.filled(9, 0)); // augmented 8x8 system
  for (var k = 0; k < 4; k++) {
    final x = dst[2 * k], y = dst[2 * k + 1], u = src[2 * k], v = src[2 * k + 1];
    a[2 * k] = [x, y, 1, 0, 0, 0, -u * x, -u * y, u];
    a[2 * k + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v];
  }
  for (var col = 0; col < 8; col++) {
    var piv = col;
    for (var r = col + 1; r < 8; r++) {
      if (a[r][col].abs() > a[piv][col].abs()) piv = r;
    }
    if (a[piv][col].abs() < 1e-12) throw const FormatException('degenerate quad');
    final tmp = a[col];
    a[col] = a[piv];
    a[piv] = tmp;
    for (var r = col + 1; r < 8; r++) {
      final f = a[r][col] / a[col][col];
      for (var c = col; c < 9; c++) {
        a[r][c] -= f * a[col][c];
      }
    }
  }
  final out = List<double>.filled(8, 0);
  for (var r = 7; r >= 0; r--) {
    var s = a[r][8];
    for (var c = r + 1; c < 8; c++) {
      s -= a[r][c] * out[c];
    }
    out[r] = s / a[r][r];
  }
  return out;
}

/// Upright crop of [quad] (TL,TR,BR,BL; 8 numbers) by perspective transform. [pad] grows the quad
/// by that fraction of its height on every side first. The quad is clamped to the image. Vertical
/// text (taller than 1.5x its width) is rotated upright (counter-clockwise, like np.rot90).
RgbImage warpQuad(RgbImage img, List<double> quad, {double pad = 0}) {
  var q = List<double>.of(quad);
  double dist(int i, int j) => math.sqrt(math.pow(q[2 * i] - q[2 * j], 2) + math.pow(q[2 * i + 1] - q[2 * j + 1], 2));
  if (pad != 0) {
    final cx = (q[0] + q[2] + q[4] + q[6]) / 4, cy = (q[1] + q[3] + q[5] + q[7]) / 4;
    final w = dist(1, 0), h = dist(3, 0);
    final ux = (q[2] - q[0]) / math.max(w, 1e-6), uy = (q[3] - q[1]) / math.max(w, 1e-6);
    final vx = (q[6] - q[0]) / math.max(h, 1e-6), vy = (q[7] - q[1]) / math.max(h, 1e-6);
    final l = w / 2 + pad * h, s = h / 2 + pad * h;
    q = [
      cx - ux * l - vx * s, cy - uy * l - vy * s,
      cx + ux * l - vx * s, cy + uy * l - vy * s,
      cx + ux * l + vx * s, cy + uy * l + vy * s,
      cx - ux * l + vx * s, cy - uy * l + vy * s,
    ];
  }
  final iw = img.width, ih = img.height;
  for (var i = 0; i < 4; i++) {
    q[2 * i] = q[2 * i].clamp(0, iw - 1).toDouble();
    q[2 * i + 1] = q[2 * i + 1].clamp(0, ih - 1).toDouble();
  }
  final w = math.max(2, roundHalfEven(math.max(dist(1, 0), dist(2, 3))));
  final h = math.max(2, roundHalfEven(math.max(dist(3, 0), dist(2, 1))));
  final co = homographyCoeffs([0, 0, w.toDouble(), 0, w.toDouble(), h.toDouble(), 0, h.toDouble()], q);
  var out = _transformBicubic(img, w, h, co);
  if (h / w >= 1.5) out = rot90(out);
  return out;
}

/// Pillow `Image.transform((w, h), PERSPECTIVE, coeffs, BICUBIC)`: out-of-image samples stay black.
RgbImage _transformBicubic(RgbImage img, int w, int h, List<double> c) {
  final iw = img.width, ih = img.height, src = img.rgb;
  final out = Uint8List(w * h * 3);
  final rows = List<double>.filled(4, 0);
  final xs = Int32List(4), ys = Int32List(4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final xo = x + 0.5, yo = y + 0.5;
      final den = c[6] * xo + c[7] * yo + 1;
      var sx = (c[0] * xo + c[1] * yo + c[2]) / den, sy = (c[3] * xo + c[4] * yo + c[5]) / den;
      if (!(sx >= 0 && sx < iw && sy >= 0 && sy < ih)) continue;
      sx -= 0.5;
      sy -= 0.5;
      final xf = sx.floor(), yf = sy.floor();
      final dx = sx - xf, dy = sy - yf;
      for (var k = 0; k < 4; k++) {
        xs[k] = (xf - 1 + k).clamp(0, iw - 1).toInt();
        ys[k] = (yf - 1 + k).clamp(0, ih - 1).toInt();
      }
      for (var ch = 0; ch < 3; ch++) {
        for (var r = 0; r < 4; r++) {
          final base = ys[r] * iw;
          rows[r] = _cubic(src[(base + xs[0]) * 3 + ch].toDouble(), src[(base + xs[1]) * 3 + ch].toDouble(), src[(base + xs[2]) * 3 + ch].toDouble(),
              src[(base + xs[3]) * 3 + ch].toDouble(), dx);
        }
        final v = _cubic(rows[0], rows[1], rows[2], rows[3], dy);
        out[(y * w + x) * 3 + ch] = v <= 0 ? 0 : (v >= 255 ? 255 : v.toInt());
      }
    }
  }
  return RgbImage(w, h, out);
}

/// Pillow's BICUBIC macro (Geometry.c): cubic convolution with a = -1.
double _cubic(double v1, double v2, double v3, double v4, double d) {
  final p2 = -v1 + v3, p3 = 2 * (v1 - v2) + v3 - v4, p4 = -v1 + v2 - v3 + v4;
  return v2 + d * (p2 + d * (p3 + d * p4));
}
