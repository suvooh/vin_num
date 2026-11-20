/// What KIND of vehicle a plate belongs to (private, commercial, EV, rental, ...): from the plate COLOURS in the
/// photo and the TEXT PATTERN. Dart port of `ml/vinplate/category.py` (the reference: its docstring spells out
/// every formula, and the tables below are copied from it verbatim). Pure Dart, deterministic, no image library;
/// checked against golden outputs of the Python file (`test/reader/category_test.dart`).
///
///     classifyPlate(crop, parsed) -> {key, label, confidence, bg:{name,rgb}, ink:{name,rgb}|null, reasons,
///                                     alternatives:[{key,label}], ev, bh}
///
/// (The Python docstring names this port `app/lib/reader/category.dart`; the file is `plate_category.dart`.)
///
/// ABOUT THE RULES: they come from two infographics supplied by the user (colour -> category) and are RECALLED,
/// NOT VERIFIED against the official notifications (MoRTH / CMVR rules and state practice vary). They are data:
/// change the tables and nothing else. A category is the reader's best explanation of what it SAW - it is shown
/// as a suggestion with its reasons and alternatives, never as a fact.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/types.dart';

// ===================================================================================== RULE TABLE
const Map<String, String> categoryLabels = {
  'private': 'Private vehicle',
  'commercial': 'Commercial / transport vehicle',
  'rental': 'Rental / self-drive vehicle',
  'ev_private': 'Electric vehicle (private)',
  'ev_commercial': 'Electric vehicle (commercial)',
  'temporary': 'Temporary registration',
  'trade': 'Trade certificate / dealer vehicle',
  'diplomatic': 'Diplomatic / consular vehicle',
  'military': 'Military / defence vehicle',
  'unknown': 'Type not determined',
};

/// (background colour, ink colour or null = any, category key, confidence factor). FIRST match wins.
const List<(String, String?, String, double)> colourRules = [
  ('blue', null, 'diplomatic', 0.7),
  ('green', 'yellow', 'ev_commercial', 1.0),
  ('green', null, 'ev_private', 1.0),
  ('yellow', 'red', 'temporary', 1.0),
  ('yellow', null, 'commercial', 1.0),
  ('black', 'yellow', 'rental', 1.0),
  ('grey', 'yellow', 'rental', 1.0),
  ('red', null, 'trade', 1.0),
  ('white', null, 'private', 1.0),
];

/// A text layout that decides the category by itself, with the backgrounds it is normally printed on.
const Map<String, (String, Set<String>)> layoutRules = {
  'diplomatic': ('diplomatic', {'blue'}),
  'military': ('military', {'black', 'grey'}),
  'dealer': ('trade', {'red'}),
};
const String bharatBg = 'white';

/// Where "green" ends and "blue" begins on the hue wheel. MEASURED: the real Indian EV plate is a teal (174-175 deg on a clean
/// photo of a green-plate car, 184-190 deg and washed out in a phone photo of it on a monitor); a real diplomatic blue is about
/// 210-225 deg. The first version used 170 and named that EV plate blue -> diplomatic. (Mirror of category.py GREEN_BLUE_HUE.)
const double greenBlueHue = 195.0;

/// TEXT-PATTERN GUARD: a diplomatic plate carries a CD / CC / UN code, so these text layouts can never be diplomatic, whatever
/// colour was measured. (Mirror of category.py NOT_DIPLOMATIC_LAYOUTS.)
const Set<String> notDiplomaticLayouts = {'standard', 'bharat', 'military', 'dealer'};

/// Categories one colour step away, tried when the confidence is below [altBelow] (at most 2, in this order).
const Map<String, List<String>> categoryAlternatives = {
  'private': ['commercial'],
  'commercial': ['private', 'temporary'],
  'temporary': ['commercial', 'trade'],
  'rental': ['diplomatic'],
  'diplomatic': ['rental', 'ev_private'],
  'ev_private': ['ev_commercial', 'diplomatic'],
  'ev_commercial': ['ev_private', 'diplomatic'],
  'trade': ['commercial', 'temporary'],
  'military': ['rental'],
};
const Map<String, List<String>> _unknownAlternatives = {
  'black': ['military', 'rental'],
  'grey': ['military', 'rental'],
};
const double altBelow = 0.75;

// Exact reason strings (the goldens use these verbatim).
String _rBg(String c) => 'plate background looks $c';
String _rInk(String c) => 'text looks $c';
const String _rOneColour = 'text colour could not be separated from the background';
const String _rBh = 'format NN BH NNNN AA = Bharat series';
const String _rBhColour = 'Bharat-series plates are white with black text';
const Map<String, String> _rLayout = {
  'diplomatic': 'format NNN CD NN = diplomatic mission code',
  'military': 'format NNL NNNNNN L = military',
  'dealer': 'format DLR NN NNNN = dealer',
};
const String _rAgree = 'colour agrees with the plate format';
String _rDisagree(String colour, String layout) => 'plate colour ($colour) does not match the $layout format';
const String _rBlue = 'blue plates with ordinary text are unusual';
const String _rDipFormat = 'diplomatic plates carry a CD / CC / UN code; this text is an ordinary registration, so the blue colour is probably a colour cast';
const String _rNoRule = 'no rule matches this colour pair';

// ======================================================================================= mechanism
const double _minSeparation = 40.0;
const double _sepFull = 120.0;

/// RGB (0..255) -> (hue degrees, saturation, value), exactly as in the Python reference.
(double, double, double) _hsv(double r, double g, double b) {
  r = r / 255.0;
  g = g / 255.0;
  b = b / 255.0;
  final mx = math.max(r, math.max(g, b));
  final mn = math.min(r, math.min(g, b));
  final d = mx - mn;
  final v = mx;
  final s = mx == 0 ? 0.0 : d / mx;
  double h;
  if (d == 0) {
    h = 0.0;
  } else if (mx == r) {
    h = 60.0 * ((g - b) / d);
  } else if (mx == g) {
    h = 60.0 * ((b - r) / d + 2.0);
  } else {
    h = 60.0 * ((r - g) / d + 4.0);
  }
  if (h < 0) h += 360.0;
  if (h >= 360.0) h -= 360.0;
  return (h, s, v);
}

double _m(double value, double boundary, double width) => (value - boundary).abs() / width;

double _hd(double h, double b) {
  final d = (h - b).abs();
  return math.min(d, 360.0 - d);
}

double _conf(List<double> margins) {
  var lo = margins[0];
  for (final m in margins) {
    if (m < lo) lo = m;
  }
  return 0.5 + 0.5 * math.min(1.0, math.max(0.0, lo));
}

/// A float RGB (0..255) -> (colour name, confidence). See `ml/vinplate/category.py` for the exact rules.
(String, double) nameColour(double r, double g, double b) {
  final (h, s, v) = _hsv(r, g, b);
  if (v < 0.28) return ('black', _conf([_m(v, 0.28, 0.15)]));
  final base = [_m(s, 0.22, 0.15), _m(v, 0.28, 0.15)];
  if (s < 0.22) {
    if (v >= 0.62) return ('white', _conf([...base, _m(v, 0.62, 0.15)]));
    if (v >= 0.40) return ('grey', _conf([...base, _m(v, 0.62, 0.15), _m(v, 0.40, 0.15)]));
    return ('black', _conf([...base, _m(v, 0.40, 0.15)]));
  }
  if (h >= 14 && h < 70 && s < 0.30 && v >= 0.62) {
    return ('white', _conf([...base, _m(s, 0.30, 0.15), _m(v, 0.62, 0.15)]));
  }
  if (h < 14 || h >= 340) return ('red', _conf([...base, _hd(h, 14) / 20.0, _hd(h, 340) / 20.0]));
  if (h < 70) {
    final double cream;
    if (s < 0.30) {
      cream = _m(v, 0.62, 0.15); // v < 0.62 here
    } else if (v >= 0.62) {
      cream = _m(s, 0.30, 0.15);
    } else {
      cream = math.max(_m(s, 0.30, 0.15), _m(v, 0.62, 0.15));
    }
    return ('yellow', _conf([...base, _hd(h, 14) / 20.0, _hd(h, 70) / 20.0, cream]));
  }
  if (h < greenBlueHue) return ('green', _conf([...base, _hd(h, 70) / 20.0, _hd(h, greenBlueHue) / 20.0]));
  if (h < 270) return ('blue', _conf([...base, _hd(h, greenBlueHue) / 20.0, _hd(h, 270) / 20.0]));
  return ('other', _conf([...base, _hd(h, 270) / 20.0, _hd(h, 340) / 20.0]));
}

/// Per-channel median of [vals] (Python `np.median`: the mean of the two middle values when the count is even).
double _median(Float64List vals) {
  final s = Float64List.fromList(vals)..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2.0;
}

/// Where the plate colours were found: (background mean, ink mean or null, distance between the final centres).
class TwoMeans {
  const TwoMeans(this.bg, this.ink, this.dist);
  final List<double> bg;
  final List<double>? ink;
  final double dist;
}

/// Deterministic 2-means over a strided sample of the crop (see the Python docstring for the exact steps).
TwoMeans twoMeans(RgbImage crop) {
  final h = crop.height, w = crop.width;
  final step = math.max(1, math.sqrt(h * w / 4096).floor());
  final rows = (h + step - 1) ~/ step, cols = (w + step - 1) ~/ step;
  final n = rows * cols;
  final xs = Float64List(n), ys = Float64List(n), zs = Float64List(n);
  final data = crop.rgb;
  var k = 0;
  for (var y = 0; y < h; y += step) {
    for (var x = 0; x < w; x += step) {
      final i = (y * w + x) * 3;
      xs[k] = data[i].toDouble();
      ys[k] = data[i + 1].toDouble();
      zs[k] = data[i + 2].toDouble();
      k++;
    }
  }
  var bg0 = _median(xs), bg1 = _median(ys), bg2 = _median(zs);
  // ink0 = the sampled pixel farthest from the median (first maximum)
  var far = 0;
  var farD = -1.0;
  for (var i = 0; i < n; i++) {
    final a = xs[i] - bg0, b = ys[i] - bg1, c = zs[i] - bg2;
    final d = a * a + b * b + c * c;
    if (d > farD) {
      farD = d;
      far = i;
    }
  }
  var ink0 = xs[far], ink1 = ys[far], ink2 = zs[far];
  final lab = Uint8List(n);
  for (var it = 0; it < 8; it++) {
    var sb0 = 0.0, sb1 = 0.0, sb2 = 0.0, si0 = 0.0, si1 = 0.0, si2 = 0.0;
    var nb = 0, ni = 0;
    for (var i = 0; i < n; i++) {
      final x = xs[i], y = ys[i], z = zs[i];
      final a = x - bg0, b = y - bg1, c = z - bg2;
      final dBg = a * a + b * b + c * c;
      final a2 = x - ink0, b2 = y - ink1, c2 = z - ink2;
      final dInk = a2 * a2 + b2 * b2 + c2 * c2;
      if (dInk < dBg) {
        // strictly nearer to the ink centre; ties -> bg
        lab[i] = 1;
        si0 += x;
        si1 += y;
        si2 += z;
        ni++;
      } else {
        lab[i] = 0;
        sb0 += x;
        sb1 += y;
        sb2 += z;
        nb++;
      }
    }
    if (nb > 0) {
      bg0 = sb0 / nb;
      bg1 = sb1 / nb;
      bg2 = sb2 / nb;
    }
    if (ni > 0) {
      ink0 = si0 / ni;
      ink1 = si1 / ni;
      ink2 = si2 / ni;
    }
  }
  var nInk = 0;
  for (var i = 0; i < n; i++) {
    nInk += lab[i];
  }
  if (nInk > n - nInk) {
    final t0 = bg0, t1 = bg1, t2 = bg2;
    bg0 = ink0;
    bg1 = ink1;
    bg2 = ink2;
    ink0 = t0;
    ink1 = t1;
    ink2 = t2;
  }
  final da = bg0 - ink0, db = bg1 - ink1, dc = bg2 - ink2;
  final dist = math.sqrt(da * da + db * db + dc * dc);
  if (dist < _minSeparation) {
    var m0 = 0.0, m1 = 0.0, m2 = 0.0;
    for (var i = 0; i < n; i++) {
      m0 += xs[i];
      m1 += ys[i];
      m2 += zs[i];
    }
    return TwoMeans([m0 / n, m1 / n, m2 / n], null, dist);
  }
  return TwoMeans([bg0, bg1, bg2], [ink0, ink1, ink2], dist);
}

List<int> _rgbInts(List<double> c) => [for (final v in c) math.min(255, math.max(0, (v + 0.5).floor()))];

/// Python `round(x, 3)` for the confidences: exact decimal rounding of the binary value.
double _round3(double x) => double.parse(x.toStringAsFixed(3));

Map<String, dynamic> _unknown(List<String> reasons, {Map<String, dynamic>? bg, Map<String, dynamic>? ink, double conf = 0.0}) => {
      'key': 'unknown',
      'label': categoryLabels['unknown'],
      'confidence': _round3(conf),
      'bg': bg,
      'ink': ink,
      'reasons': reasons,
      'alternatives': <Map<String, dynamic>>[],
      'ev': false,
      'bh': false,
    };

List<Map<String, dynamic>> _alts(List<String> keys) => [
      for (final k in keys.take(2)) {'key': k, 'label': categoryLabels[k]}
    ];

(String, double)? _colourRule(String bg, String? ink) {
  for (final (rbg, rink, key, factor) in colourRules) {
    if (rbg == bg && (rink == null || rink == ink)) return (key, factor);
  }
  return null;
}

/// The category of a plate from its crop and the parsed text layout ([parsePlate] result or null).
/// A missing / empty crop gives `unknown` ("no rule matches"), never a guess.
Map<String, dynamic> classifyPlate(RgbImage? crop, Map<String, dynamic>? parsed) {
  if (crop == null || crop.width < 1 || crop.height < 1) return _unknown([_rNoRule]);
  final tm = twoMeans(crop);
  final bgC = tm.bg, inkC = tm.ink, dist = tm.dist;
  final (bgName, bgConf) = nameColour(bgC[0], bgC[1], bgC[2]);
  final String? inkName;
  final double? inkConf;
  if (inkC != null) {
    final (n, c) = nameColour(inkC[0], inkC[1], inkC[2]);
    inkName = n;
    inkConf = c;
  } else {
    inkName = null;
    inkConf = null;
  }
  final bg = <String, dynamic>{'name': bgName, 'rgb': _rgbInts(bgC)};
  final Map<String, dynamic>? ink = inkC != null ? {'name': inkName, 'rgb': _rgbInts(inkC)} : null;
  final layout = parsed?['layout'] as String?;
  final base = math.min(bgConf, inkC != null ? inkConf! : 0.7) * math.min(1.0, dist / _sepFull);
  final reasons = <String>[_rBg(bgName), inkC != null ? _rInk(inkName!) : _rOneColour];

  final bh = layout == 'bharat';
  final colour = _colourRule(bgName, inkName);
  final lr = layout == null ? null : layoutRules[layout];
  if (lr != null) {
    final (key, expectedBg) = lr;
    reasons.add(_rLayout[layout]!);
    final double conf;
    if (expectedBg.contains(bgName)) {
      conf = math.min(1.0, base + 0.1);
      reasons.add(_rAgree);
    } else {
      conf = base * 0.6;
      reasons.add(_rDisagree(bgName, layout!));
    }
    final altKeys = <String>[if (colour != null && colour.$1 != key) colour.$1];
    for (final k in categoryAlternatives[key] ?? const <String>[]) {
      if (!altKeys.contains(k) && k != key) altKeys.add(k);
    }
    return {
      'key': key,
      'label': categoryLabels[key],
      'confidence': _round3(conf),
      'bg': bg,
      'ink': ink,
      'reasons': reasons,
      'alternatives': conf < altBelow ? _alts(altKeys) : <Map<String, dynamic>>[],
      'ev': false,
      'bh': false,
    };
  }

  if (colour == null) {
    final keys = _unknownAlternatives[bgName] ?? const ['private', 'commercial'];
    final out = _unknown([...reasons, _rNoRule], bg: bg, ink: ink, conf: base * 0.6);
    out['alternatives'] = _alts(keys);
    out['bh'] = bh;
    if (bh) (out['reasons'] as List<String>).insert(2, _rBh);
    return out;
  }

  final (key, factor) = colour;
  if (key == 'diplomatic' && layout != null && notDiplomaticLayouts.contains(layout)) {
    // text-pattern guard: the likelier readings of a blue-looking plate with a normal number
    final alt = inkName == 'yellow' ? const ['ev_commercial', 'ev_private'] : const ['ev_private', 'private'];
    final out = _unknown([...reasons, _rDipFormat], bg: bg, ink: ink, conf: base * 0.5);
    out['alternatives'] = _alts(alt);
    out['bh'] = bh;
    if (bh) (out['reasons'] as List<String>).insert(2, _rBh);
    return out;
  }
  var conf = base * factor;
  if (bgName == 'blue') reasons.add(_rBlue);
  if (bh) {
    reasons.add(_rBh);
    if (bgName == bharatBg) {
      conf = math.min(1.0, conf + 0.1);
      reasons.add(_rAgree);
    } else {
      conf = conf * 0.6;
      reasons.add(_rBhColour);
    }
  }
  // an ordinary registration rules out a diplomatic plate (notDiplomaticLayouts), so it is never offered as an alternative
  final near = [
    for (final k in categoryAlternatives[key] ?? const <String>[])
      if (!(k == 'diplomatic' && layout != null && notDiplomaticLayouts.contains(layout))) k
  ];
  return {
    'key': key,
    'label': categoryLabels[key],
    'confidence': _round3(conf),
    'bg': bg,
    'ink': ink,
    'reasons': reasons,
    'alternatives': conf < altBelow ? _alts(near) : <Map<String, dynamic>>[],
    'ev': key.startsWith('ev_'),
    'bh': bh,
  };
}
