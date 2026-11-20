/// Template grammar for Indian plates and VINs: an NFA simulated incrementally, plus parsing and
/// soft priors. Port of `ml/vinplate/grammar/{base,india_plate,vin}.py` - same states, same
/// wildcard semantics for '?', same scores. The decoder asks "can this prefix still become a
/// plate?" at every CTC step for every beam hypothesis, so `step` is cached.
library;

import 'dart:math' as math;

import 'vocab.dart';

// ======================================================================================= tokens
abstract class Token {
  String get name;
}

/// Between [lo] and [hi] characters, each drawn from [chars].
class Cls implements Token {
  const Cls(this.name, this.chars, [this.lo = 1, this.hi = 1]);
  @override
  final String name;
  final String chars;
  final int lo, hi;
}

/// Exactly one of [options] ("" makes it optional).
class Choice implements Token {
  const Choice(this.name, this.options);
  @override
  final String name;
  final List<String> options;
}

typedef ScoreFn = double Function(Map<String, String> parts);

class Template {
  Template(this.name, this.tokens, {this.logPrior = 0.0, ScoreFn? score}) : score = score ?? _zero;
  final String name;
  final List<Token> tokens;
  final double logPrior;
  final ScoreFn score;
  static double _zero(Map<String, String> _) => 0.0;
}

// ================================================================================ NFA state
class _El {
  const _El(this.t, this.i, this.p);
  final int t, i;
  final Object p; // int count for a Cls, String consumed prefix for a Choice
  String get key => '$t,$i,$p';
}

/// An interned set of NFA elements. Equal sets are the same object, so identity is equality.
class GState {
  GState._(this.id, this._elements);
  final int id;
  final List<_El> _elements;
}

Object _p0(Token tok) => tok is Cls ? 0 : '';

bool _canFinish(Token tok, Object p) => tok is Cls ? (p as int) >= tok.lo : (tok as Choice).options.contains(p as String);

Object? _consume(Token tok, Object p, String ch) {
  if (tok is Cls) {
    final n = p as int;
    return (tok.chars.contains(ch) && n < tok.hi) ? n + 1 : null;
  }
  final q = (p as String) + ch;
  return (tok as Choice).options.any((o) => o.startsWith(q)) ? q : null;
}

class Grammar {
  Grammar(this.name, this.templates) {
    _start = _make([for (var t = 0; t < templates.length; t++) _El(t, 0, _p0(templates[t].tokens[0]))]);
  }

  final String name;
  final List<Template> templates;
  late final GState _start;
  final Map<String, GState> _intern = {};
  final Map<int, GState?> _cache = {};

  GState _make(Iterable<_El> els) {
    final byKey = <String, _El>{for (final e in els) e.key: e};
    final keys = byKey.keys.toList()..sort();
    final canon = keys.join('|');
    return _intern.putIfAbsent(canon, () => GState._(_intern.length, [for (final k in keys) byKey[k]!]));
  }

  GState start() => _start;

  /// Elements reachable from (i, p) by finishing tokens that may end here (the optional ones).
  List<(int, Object)> _targets(int t, int i, Object p) {
    final toks = templates[t].tokens;
    final out = <(int, Object)>[(i, p)];
    while (_canFinish(toks[i], p) && i + 1 < toks.length) {
      i += 1;
      p = _p0(toks[i]);
      out.add((i, p));
    }
    return out;
  }

  /// Advance by one character. '?' (unreadable) is a wildcard: it advances every way any legal
  /// character could, so a prefix containing it stays alive as long as SOME completion is legal.
  /// Returns null when no template can consume [ch].
  GState? step(GState state, String ch) {
    final key = state.id * 128 + ch.codeUnitAt(0);
    if (_cache.containsKey(key)) return _cache[key];
    GState? res;
    if (ch == unk) {
      final union = <String, _El>{};
      for (var k = 0; k < alphabet.length; k++) {
        final n = step(state, alphabet[k]);
        if (n != null) {
          for (final e in n._elements) {
            union[e.key] = e;
          }
        }
      }
      res = union.isEmpty ? null : _make(union.values);
    } else {
      final out = <_El>[];
      for (final e in state._elements) {
        final toks = templates[e.t].tokens;
        for (final (j, q) in _targets(e.t, e.i, e.p)) {
          final np = _consume(toks[j], q, ch);
          if (np != null) out.add(_El(e.t, j, np));
        }
      }
      res = out.isEmpty ? null : _make(out);
    }
    _cache[key] = res;
    return res;
  }

  bool accepting(GState state) {
    for (final e in state._elements) {
      final toks = templates[e.t].tokens;
      if (!_canFinish(toks[e.i], e.p)) continue;
      var rest = true;
      for (var k = e.i + 1; k < toks.length; k++) {
        if (!_canFinish(toks[k], _p0(toks[k]))) {
          rest = false;
          break;
        }
      }
      if (rest) return true;
    }
    return false;
  }

  bool isValid(String text) {
    GState? s = start();
    for (var i = 0; i < text.length; i++) {
      s = step(s!, text[i]);
      if (s == null) return false;
    }
    return accepting(s!);
  }

  // ------------------------------------------------------------------ segmentation + prior
  List<(Template, Map<String, String>)> segmentations(String text) => [
        for (final tpl in templates)
          for (final parts in _segment(tpl.tokens, text, 0, 0)) (tpl, parts),
      ];

  /// Best segmentation by `template prior + template score`; the first of equals wins (same
  /// iteration order as the Python generator).
  (Template, Map<String, String>, double)? bestParse(String text) {
    (Template, Map<String, String>, double)? best;
    for (final (tpl, parts) in segmentations(text)) {
      final lp = tpl.logPrior + tpl.score(parts);
      if (best == null || lp > best.$3) best = (tpl, parts, lp);
    }
    return best;
  }

  /// Soft prior of a complete string; null if the grammar rejects it.
  double? logPrior(String text) => bestParse(text)?.$3;
}

List<Map<String, String>> _segment(List<Token> tokens, String text, int i, int pos) {
  if (i == tokens.length) return pos == text.length ? [<String, String>{}] : const [];
  final tok = tokens[i];
  final out = <Map<String, String>>[];
  if (tok is Cls) {
    var n = 0;
    while (pos + n < text.length && n < tok.hi && (tok.chars.contains(text[pos + n]) || text[pos + n] == unk)) {
      n++;
    }
    for (var take = n; take >= tok.lo; take--) {
      for (final rest in _segment(tokens, text, i + 1, pos + take)) {
        out.add({tok.name: text.substring(pos, pos + take), ...rest});
      }
    }
  } else {
    for (final opt in (tok as Choice).options) {
      final end = pos + opt.length;
      if (end > text.length) continue;
      final seg = text.substring(pos, end);
      var ok = true;
      for (var k = 0; k < opt.length; k++) {
        if (seg[k] != opt[k] && seg[k] != unk) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      for (final rest in _segment(tokens, text, i + 1, end)) {
        out.add({tok.name: seg, ...rest});
      }
    }
  }
  return out;
}

// ============================================================================ Indian plates
const String _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const String _digits = '0123456789';

const Map<String, String> stateNames = {
  'AN': 'Andaman & Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
  'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
  'DD': 'Daman & Diu / Dadra & Nagar Haveli', 'DL': 'Delhi', 'GA': 'Goa',
  'GJ': 'Gujarat', 'HP': 'Himachal Pradesh', 'HR': 'Haryana', 'JH': 'Jharkhand',
  'JK': 'Jammu & Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala', 'LA': 'Ladakh',
  'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur',
  'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha',
  'PB': 'Punjab', 'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim',
  'TG': 'Telangana', 'TN': 'Tamil Nadu', 'TR': 'Tripura', 'UK': 'Uttarakhand',
  'UP': 'Uttar Pradesh', 'WB': 'West Bengal',
  // retired / legacy codes still seen on older plates
  'CT': 'Chhattisgarh (old)', 'DN': 'Dadra & Nagar Haveli (old)', 'OR': 'Odisha (old)',
  'TS': 'Telangana (old)', 'UA': 'Uttarakhand (old)',
};

const Set<String> legacyStateCodes = {'CT', 'DN', 'OR', 'TS', 'UA'};

/// Every state / UT code accepted by the grammar (current and retired). Kept equal to the keys
/// of [stateNames] (checked in the tests).
const Set<String> stateCodes = {
  'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA',
  'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TG', 'TN', 'TR', 'UK', 'UP', 'WB',
  'CT', 'DN', 'OR', 'TS', 'UA',
};

/// `STATE_CODES = tuple(sorted(STATE_NAMES))` in Python (the Choice options, in this order).
final List<String> _stateCodesSorted = (stateNames.keys.toList()..sort());

const String _seriesOdd = 'IOQ';

int? _intOrNull(String x) => RegExp(r'^[0-9]+$').hasMatch(x) ? int.parse(x) : null;

double _standardScore(Map<String, String> p) {
  var s = 0.0;
  final n = p['number']!.length;
  // HSRP-era numbers are zero-padded to 4 digits; short ones exist only on pre-HSRP plates, and a
  // washed-out digit is indistinguishable from an absent one in CTC output, so shortness is expensive.
  s += const {4: 0.0, 3: -4.0, 2: -5.0, 1: -6.0}[n]!;
  s += const {2: 0.0, 1: -0.3}[p['rto']!.length]!;
  s += const {0: -1.5, 1: -0.2, 2: 0.0}[p['series']!.length] ?? 0.0;
  if (p['series']!.length == 3) {
    // legitimate in Delhi (DL 1C AB 1234); elsewhere almost always an inserted / misread character
    s += (p['state'] == 'DL' && p['rto']!.length == 1) ? -0.4 : -2.5;
  }
  final rto = _intOrNull(p['rto']!);
  if (rto == 0) s -= 6.0; // (no RTO table is loaded: RTO_ALLOWED is None in the reference)
  var odd = 0;
  for (final c in p['series']!.split('')) {
    if (_seriesOdd.contains(c)) odd++;
  }
  s -= 2.0 * odd;
  if (legacyStateCodes.contains(p['state'])) s -= 0.7;
  return s;
}

double _bharatScore(Map<String, String> p) {
  var s = 0.0;
  final year = _intOrNull(p['year']!);
  if (year != null && year < 21) s -= 4.0; // BH series started in 2021
  var odd = 0;
  for (final c in p['suffix']!.split('')) {
    if (_seriesOdd.contains(c)) odd++;
  }
  s -= 2.0 * odd;
  return s;
}

Grammar _buildPlate() => Grammar('india_plate', [
      Template('standard', [
        Choice('state', _stateCodesSorted),
        const Cls('rto', _digits, 1, 2),
        const Cls('series', _letters, 0, 3),
        const Cls('number', _digits, 1, 4),
      ], logPrior: 0.0, score: _standardScore),
      Template('bharat', [
        const Cls('year', _digits, 2, 2),
        const Choice('bh', ['BH']),
        const Cls('number', _digits, 4, 4),
        const Cls('suffix', _letters, 1, 2),
      ], logPrior: -0.5, score: _bharatScore),
      Template('diplomatic', [
        const Cls('code', _digits, 1, 3),
        const Choice('mission', ['CD', 'CC', 'UN']),
        const Cls('number', _digits, 1, 4),
      ], logPrior: -3.0),
      Template('military', [
        const Cls('year', _digits, 2, 2),
        const Cls('base', _letters, 1, 1),
        const Cls('serial', _digits, 6, 6),
        const Cls('klass', _letters, 1, 1),
      ], logPrior: -4.0),
      // Dealer / trade-certificate mark (DLR 01 2345), as shown on a user-supplied infographic. ILLUSTRATIVE and
      // low-prior; mirrors ml/vinplate/grammar/india_plate.py exactly: 'DLR' is required, then 1-2 digits and
      // 3-4 digits (longest first group wins: DLR12345 -> rto 12 + number 345).
      Template('dealer', [
        const Choice('series', ['DLR']),
        const Cls('rto', _digits, 1, 2),
        const Cls('number', _digits, 3, 4),
      ], logPrior: -2.0),
    ]);

final Grammar plateGrammar = _buildPlate();

double _round(double x, int d) {
  final m = math.pow(10, d).toDouble();
  return (x * m).roundToDouble() / m;
}

/// Structured read of a complete plate string, or null if it is not a legal layout.
Map<String, dynamic>? parsePlate(String text) {
  final best = plateGrammar.bestParse(text);
  if (best == null) return null;
  final (tpl, p, prior) = best;
  final out = <String, dynamic>{'layout': tpl.name, 'plate': text, 'log_prior': _round(prior, 3)};
  if (tpl.name == 'standard') {
    final st = p['state']!;
    out['state_code'] = st;
    out['state_name'] = stateNames[st]; // null while the state code still holds a '?'
    out['rto'] = p['rto'];
    out['series'] = p['series'];
    out['number'] = p['number'];
    out['legacy_state_code'] = legacyStateCodes.contains(st);
    final String head;
    final List<String> tail;
    if (st == 'DL' && p['series']!.isNotEmpty && p['rto']!.length == 1) {
      // Delhi prints the class letter against the RTO: DL 1C AB 1234
      head = '$st ${p['rto']}${p['series']![0]}';
      tail = [p['series']!.substring(1), p['number']!];
    } else {
      head = '$st ${p['rto']}';
      tail = [p['series']!, p['number']!];
    }
    out['display'] = [head, ...tail.where((t) => t.isNotEmpty)].join(' ');
  } else if (tpl.name == 'bharat') {
    out['year'] = p['year'];
    out['number'] = p['number'];
    out['suffix'] = p['suffix'];
    out['display'] = '${p['year']} BH ${p['number']} ${p['suffix']}';
  } else if (tpl.name == 'diplomatic') {
    out['code'] = p['code'];
    out['mission'] = p['mission'];
    out['number'] = p['number'];
    out['display'] = '${p['code']} ${p['mission']} ${p['number']}';
  } else if (tpl.name == 'dealer') {
    out['series'] = p['series'];
    out['rto'] = p['rto'];
    out['number'] = p['number'];
    out['display'] = '${p['series']} ${p['rto']} ${p['number']}';
  } else {
    out['display'] = '${p['year']}${p['base']} ${p['serial']}${p['klass']}';
  }
  return out;
}

// =================================================================================== VIN
const String vinChars = 'ABCDEFGHJKLMNPRSTUVWXYZ0123456789'; // no I, O, Q
const String _alnum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
const List<String> _vinLabels = ['', 'VIN', 'VINNO', 'CHASSIS', 'CHASSISNO', 'CHASISNO', 'CHASISSNO'];

const double checkDigitBonus = 1.0;
const double indiaWmiBonus = 0.8;
const double knownWmiBonus = 0.4;

/// WMI -> (maker, verified against NHTSA vPIC in the reference project).
const Map<String, (String, bool)> knownWmi = {
  'MAJ': ('Ford India', true), 'MD2': ('Bajaj Auto', true), 'ME3': ('Royal Enfield', true),
  'MA1': ('Mahindra', false), 'MA3': ('Maruti Suzuki', false), 'MAK': ('Honda Cars India', false),
  'MAL': ('Hyundai India', false), 'MAT': ('Tata Motors', false), 'MBH': ('Maruti Suzuki', false),
  'MBJ': ('Toyota Kirloskar', false), 'MBL': ('Hero MotoCorp', false),
  'ME4': ('Honda Motorcycle & Scooter India', false), 'MD6': ('TVS Motor', false),
};

const String _yearCodes = 'ABCDEFGHJKLMNPRSTVWXY123456789'; // 30 codes, 1980/2010 = A
const Map<String, int> _vinValues = {
  '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
  'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
  'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
  'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
};
const List<int> _vinWeights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];

/// Test hook: pins "today's year" for the model-year check (the reference reads the clock).
int? vinNowYearOverride;

/// ISO 3779 / FMVSS 115 check digit. False for anything that is not a clean 17-char VIN.
bool vinCheckDigitOk(String vin) {
  if (vin.length != 17) return false;
  var total = 0;
  for (var i = 0; i < 17; i++) {
    final v = _vinValues[vin[i]];
    if (v == null) return false;
    total += v * _vinWeights[i];
  }
  final r = total % 11;
  return vin[8] == (r == 10 ? 'X' : '$r');
}

/// Model years consistent with position 10 (30-year cycle), oldest first. [nowYear] is the
/// calendar year (default: the clock, or [vinNowYearOverride]); model years run one ahead.
List<int> vinYearCandidates(String vin, {int? nowYear}) {
  if (vin.length < 10 || !_yearCodes.contains(vin[9])) return [];
  final c = vin[9];
  final List<int> years;
  if (RegExp(r'^[0-9]$').hasMatch(c)) {
    years = [2000 + int.parse(c)]; // '1'..'9' -> 2001..2009
  } else {
    final i = _yearCodes.indexOf(c);
    years = [1980 + i, 2010 + i];
  }
  final now = (nowYear ?? vinNowYearOverride ?? DateTime.now().year) + 1;
  return [for (final y in years) if (y >= 1981 && y <= now) y];
}

double _strictScore(Map<String, String> p) {
  final v = p['vin']!;
  var s = 0.0;
  // An unreadable ('?') character carries no evidence either way: skip the checks that need it.
  if (v[9] != unk) {
    if (!_yearCodes.contains(v[9]) || vinYearCandidates(v).isEmpty) s -= 2.5;
  }
  if (v[0] == 'M' && 'ABCDE'.contains(v[1])) s += indiaWmiBonus;
  if (knownWmi.containsKey(v.substring(0, 3))) s += knownWmiBonus;
  if (v.substring(11).split('').every((c) => RegExp(r'^[0-9]$').hasMatch(c) || c == unk)) s += 0.4;
  if (!v.contains(unk) && vinCheckDigitOk(v)) s += checkDigitBonus;
  return s;
}

Grammar _buildVin() => Grammar('vin', [
      Template('vin17', [const Choice('label', _vinLabels), const Cls('vin', vinChars, 17, 17)], logPrior: 0.0, score: _strictScore),
      // Legacy / non-conformant chassis numbers (and a 17-char read still containing I/O/Q): allowed,
      // but a strict ISO reading wins whenever it is remotely competitive.
      Template('chassis_legacy', [const Choice('label', _vinLabels), const Cls('chassis', _alnum, 6, 17)], logPrior: -6.0),
    ]);

final Grammar vinGrammar = _buildVin();

Map<String, dynamic>? parseVin(String text) {
  final best = vinGrammar.bestParse(text);
  if (best == null) return null;
  final (tpl, p, prior) = best;
  final out = <String, dynamic>{
    'layout': tpl.name,
    'log_prior': _round(prior, 3),
    'label': (p['label'] ?? '').isEmpty ? null : p['label'],
  };
  if (tpl.name == 'vin17') {
    final v = p['vin']!;
    out['vin'] = v;
    out['wmi'] = v.substring(0, 3);
    out['vds'] = v.substring(3, 9);
    out['vis'] = v.substring(9);
    out['india_wmi'] = v[0] == 'M' && 'ABCDE'.contains(v[1]);
    out['known_maker'] = knownWmi[v.substring(0, 3)]?.$1;
    out['check_digit_ok'] = v.contains(unk) ? null : vinCheckDigitOk(v);
    out['year_candidates'] = vinYearCandidates(v);
  } else {
    out['vin'] = p['chassis'];
    out['short_chassis'] = p['chassis']!.length < 17;
  }
  return out;
}
