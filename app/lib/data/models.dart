/// Typed view of the backend's `/api/extract` response (see backend/app/service.py `field_out`).
/// Parsing is tolerant: a missing optional key never throws, a missing REQUIRED shape does.
library;

class Alternative {
  const Alternative(this.text, this.p);
  final String text;
  final double p;
}

/// One ranked candidate for a character the photo could not settle.
class SlotOption {
  const SlotOption(this.char, this.p);
  final String char;
  final double p; // share among the characters the grammar allows in this slot (sums to 1 over all options)
}

/// A character position that is unknown or weak, with the characters that are LEGAL there ranked by
/// the recogniser's own evidence. [suggested] is set only when one candidate clearly dominates; it is
/// shown as a suggestion, never as a reading.
class UnknownSlot {
  const UnknownSlot(this.position, this.options, this.suggested);
  final int position; // index into FieldResult.value
  final List<SlotOption> options;
  final String? suggested;

  factory UnknownSlot.fromJson(Map<String, dynamic> j) => UnknownSlot(
        (j['position'] as num).toInt(),
        [for (final o in (j['options'] as List? ?? const [])) SlotOption((o as Map)['char'] as String, (o['p'] as num).toDouble())],
        j['suggested'] as String?,
      );
}

/// Another vehicle type the evidence also allows (shown in the type picker as "possible").
class PlateAlt {
  const PlateAlt(this.key, this.p);
  final String key;
  final double p;
}

/// The vehicle category the reader worked out from the plate's colours and text pattern. The key is one of
/// private | commercial | rental | ev_private | ev_commercial | temporary | trade | diplomatic | military |
/// unknown. [confidence] below ~0.6 is shown as "Probably ..." with a prompt to check the plate.
class PlateCategory {
  const PlateCategory({
    required this.key,
    this.label = '',
    this.confidence = 0,
    this.reasons = const [],
    this.alternatives = const [],
    this.ev = false,
    this.bh = false,
    this.colourName,
  });

  final String key;
  final String label;
  final double confidence;
  final List<String> reasons;
  final List<PlateAlt> alternatives;
  final bool ev; // electric vehicle (green plate or an EV-specific format)
  final bool bh; // Bharat series (22 BH 1234 AA)
  final String? colourName; // the plate background the reader measured, e.g. "white"

  bool get isUnknown => key == 'unknown';

  factory PlateCategory.fromJson(Map<String, dynamic> j) {
    final bg = j['bg'];
    return PlateCategory(
      key: j['key'] as String? ?? 'unknown',
      label: j['label'] as String? ?? '',
      confidence: (j['confidence'] as num?)?.toDouble() ?? 0,
      reasons: [for (final r in (j['reasons'] as List? ?? const [])) '$r'],
      alternatives: [
        for (final a in (j['alternatives'] as List? ?? const []))
          if (a is Map && a['key'] is String) PlateAlt(a['key'] as String, ((a['p'] ?? a['confidence']) as num?)?.toDouble() ?? 0)
      ],
      ev: j['ev'] as bool? ?? false,
      bh: j['bh'] as bool? ?? false,
      colourName: bg is Map ? bg['name'] as String? : null,
    );
  }

  Map<String, dynamic> toJson() => {'key': key, 'label': label, 'confidence': confidence};
}

class FieldResult {
  const FieldResult({
    required this.domain,
    required this.status,
    required this.masked,
    required this.confidence,
    this.value,
    this.display,
    this.charConfidence = const [],
    this.inferred = const [],
    this.alternatives = const [],
    this.reasons = const [],
    this.guidance = const [],
    this.image = 0,
    this.box,
    this.checkDigitOk,
    this.stateName,
    this.maker,
    this.views = 1,
    this.rows = 1,
    this.layout,
    this.unknownSlots = const [],
    this.category,
  });

  final String domain; // plate | vin
  final String status; // accept | review (reject is never returned as a field)
  final String? value;
  final String masked;
  final String? display;
  final double confidence;
  final List<double> charConfidence;
  final List<int> inferred;
  final List<Alternative> alternatives;
  final List<String> reasons;
  final List<String> guidance;
  final int image;
  final List<int>? box;
  final bool? checkDigitOk;
  final String? stateName;
  final String? maker;
  final int views;
  final int rows;
  final String? layout;
  final List<UnknownSlot> unknownSlots;
  final PlateCategory? category; // plates only

  /// Positions (in [shown]) where one candidate clearly dominates the legal characters for that slot.
  /// Only valid while [shown] and [value] have the same length (no gap '?' was inserted into [masked]).
  Map<int, String> get inlineHints {
    final v = value ?? masked;
    if (masked.length != v.length) return const {};
    return {
      for (final s in unknownSlots)
        if (s.suggested != null && s.position < masked.length && masked[s.position] == '?') s.position: s.suggested!
    };
  }

  bool get isPlate => domain == 'plate';
  bool get isAccepted => status == 'accept';
  bool get needsReview => !isAccepted;
  int get unknownCount => '?'.allMatches(masked).length;

  /// What to DISPLAY: the clean value only when accepted; otherwise the masked text, so a
  /// character the pixels did not support is shown as unknown, never as a guess.
  String get shown => isAccepted ? (display ?? value ?? masked) : masked;

  factory FieldResult.fromJson(Map<String, dynamic> j) {
    final parsed = (j['parsed'] as Map?)?.cast<String, dynamic>() ?? const {};
    final quality = (j['quality'] as Map?)?.cast<String, dynamic>() ?? const {};
    return FieldResult(
      domain: j['domain'] as String? ?? 'plate',
      status: j['status'] as String? ?? 'review',
      value: j['value'] as String?,
      masked: j['masked'] as String? ?? (j['value'] as String? ?? ''),
      display: j['display'] as String?,
      confidence: (j['confidence'] as num?)?.toDouble() ?? 0,
      charConfidence: [for (final c in (j['char_confidence'] as List? ?? const [])) (c as num).toDouble()],
      inferred: [for (final i in (j['inferred'] as List? ?? const [])) (i as num).toInt()],
      alternatives: [
        for (final a in (j['alternatives'] as List? ?? const []))
          Alternative((a as Map)['text'] as String, (a['p'] as num).toDouble())
      ],
      reasons: [for (final r in (j['reasons'] as List? ?? const [])) '$r'],
      guidance: [for (final r in (quality['guidance'] as List? ?? const [])) '$r'],
      image: (j['image'] as num?)?.toInt() ?? 0,
      box: (j['box'] as List?)?.map((e) => (e as num).toInt()).toList(),
      checkDigitOk: j['check_digit_ok'] as bool?,
      stateName: parsed['state_name'] as String?,
      maker: parsed['known_maker'] as String?,
      views: (j['views'] as num?)?.toInt() ?? 1,
      rows: (j['rows'] as num?)?.toInt() ?? 1,
      layout: parsed['layout'] as String?,
      unknownSlots: [for (final u in (j['unknown_candidates'] as List? ?? const [])) UnknownSlot.fromJson((u as Map).cast<String, dynamic>())],
      category: j['category'] is Map ? PlateCategory.fromJson((j['category'] as Map).cast<String, dynamic>()) : null,
    );
  }
}

class Authenticity {
  const Authenticity({
    required this.status,
    required this.liveness,
    this.assurance = 'none',
    this.reasons = const [],
    this.notRun = const [],
    this.positive = const [],
    this.paScore = 0,
    this.notRunReasons = const {},
  });

  final String status; // no_flags | review | reject
  final String liveness; // confirmed | suspected_spoof | not_established
  final String assurance;
  final List<String> reasons;
  final List<String> notRun;
  final List<String> positive;
  final double paScore;
  final Map<String, String> notRunReasons;

  bool get isSpoofSuspected => liveness == 'suspected_spoof' || status == 'reject';
  bool get isConfirmed => liveness == 'confirmed' && status != 'reject';

  factory Authenticity.fromJson(Map<String, dynamic> j) {
    final ev = (j['evidence'] as Map?)?.cast<String, dynamic>() ?? const {};
    final why = (ev['not_run_reasons'] as Map?)?.map((k, v) => MapEntry('$k', '$v')) ?? <String, String>{};
    return Authenticity(
      status: j['status'] as String? ?? 'review',
      liveness: j['liveness'] as String? ?? 'not_established',
      assurance: j['assurance'] as String? ?? 'none',
      reasons: [for (final r in (j['reasons'] as List? ?? const [])) '$r'],
      notRun: [for (final r in (j['not_run'] as List? ?? const [])) '$r'],
      positive: [for (final r in (j['positive'] as List? ?? const [])) '$r'],
      paScore: (j['pa_score'] as num?)?.toDouble() ?? 0,
      notRunReasons: why,
    );
  }
}

/// What the reader noticed about the SCENE (not the plate): web-page text, many lines of UI text...
/// A score of 0 means nothing screen-like was seen.
class ScreenContext {
  const ScreenContext({this.score = 0, this.reasons = const []});
  final double score;
  final List<String> reasons;

  factory ScreenContext.fromJson(Map<String, dynamic> j) =>
      ScreenContext(score: (j['score'] as num?)?.toDouble() ?? 0, reasons: [for (final r in (j['reasons'] as List? ?? const [])) '$r']);
}

class PhotoInfo {
  const PhotoInfo(this.index, this.name, this.role, this.width, this.height);
  final int index;
  final String name;
  final String role;
  final int width, height;
}

class ScanResult {
  const ScanResult({
    required this.captureId,
    required this.processedAt,
    required this.auth,
    this.plate,
    this.vin,
    this.conflicts = const [],
    this.photos = const [],
    this.timingsMs = const {},
    this.screenContext = const ScreenContext(),
    this.raw,
  });

  final String captureId;
  final String processedAt;
  final FieldResult? plate;
  final FieldResult? vin;
  final Authenticity auth;
  final List<String> conflicts;
  final List<PhotoInfo> photos;
  final Map<String, double> timingsMs;
  final ScreenContext screenContext;
  final Map<String, dynamic>? raw; // the response as received (history stores and re-parses it)

  bool get foundAnything => plate != null || vin != null;

  factory ScanResult.fromJson(Map<String, dynamic> j) {
    final fields = (j['fields'] as Map?)?.cast<String, dynamic>();
    if (fields == null) throw const FormatException('response has no "fields"');
    FieldResult? f(String k) => fields[k] == null ? null : FieldResult.fromJson((fields[k] as Map).cast<String, dynamic>());
    return ScanResult(
      captureId: j['capture_id'] as String? ?? '',
      processedAt: j['processed_at'] as String? ?? '',
      plate: f('plate'),
      vin: f('vin'),
      auth: Authenticity.fromJson(((j['authenticity'] as Map?) ?? const {}).cast<String, dynamic>()),
      conflicts: [
        for (final c in (j['conflicts'] as List? ?? const []))
          'photo ${((c as Map)['image'] as num) + 1} read ${c['masked']} (disagrees with ${c['main']})'
      ],
      photos: [
        for (final p in (j['images'] as List? ?? const []))
          PhotoInfo((p as Map)['index'] as int, p['name'] as String? ?? '', p['role'] as String? ?? 'auto',
              ((p['size'] as List?)?[0] as num?)?.toInt() ?? 0, ((p['size'] as List?)?[1] as num?)?.toInt() ?? 0)
      ],
      timingsMs: ((j['timings_ms'] as Map?) ?? const {}).map((k, v) => MapEntry('$k', (v as num).toDouble())),
      screenContext: ScreenContext.fromJson(((j['screen_context'] as Map?) ?? const {}).cast<String, dynamic>()),
      raw: j,
    );
  }
}
