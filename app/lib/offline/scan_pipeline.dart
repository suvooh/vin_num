// The on-device counterpart of backend/app/service.py `analyse`: photos with roles in, the JSON the
// app's `ScanResult.fromJson` reads out. It is pure logic over small PORTS (a scene reader, a
// recapture scorer, a photo decoder), so it is unit-tested here without native ONNX Runtime; the
// real models are plugged in by offline_engine.dart.
//
// Roles (same as the backend): plate (the first is the primary / reference frame), plate_flash
// (flash ON, same pose), angle (same plate from another position), vin, auto.
//
// Robustness rule: every CHECK (recapture model, flash test, parallax, provenance, the decision
// itself) is isolated. A failing check is recorded as "not run" with its reason - it never aborts
// the scan and never becomes a pass. The READING is the product: it is always returned.
import '../checks.dart' show Finding, scanProvenance;
import '../engine/types.dart';
import '../reader/liveness.dart';
import '../reader/planarity.dart';
import 'engine_host.dart' show EnginePhoto;
import 'padnet.dart';
import 'photo_decode.dart';

typedef Box4 = (int, int, int, int);
typedef Rank = (int, int, double);

// ================================================================================== ports
/// What the pipeline needs of one reading (the scene reader's `FieldReading`).
class ReadField {
  const ReadField({
    required this.domain,
    required this.status,
    required this.rank,
    required this.image,
    required this.box,
    required this.text,
    required this.masked,
    required this.json,
  });
  final String domain, status, text, masked;
  final Rank rank;
  final int image; // index within the group of photos it was read from
  final Box4 box;
  final Map<String, dynamic> json; // FieldReading.toJson()
}

/// `SceneResult.screenContext` when nothing suggests a screen (the Python's NO_SCREEN_CONTEXT).
const Map<String, dynamic> noScreenContext = {'score': 0.0, 'url': false, 'ui_words': <String>[], 'text_lines': 0, 'reasons': <String>[]};

/// What the pipeline needs of a scene result (`SceneResult`).
class ReadScene {
  const ReadScene({
    this.plate,
    this.vin,
    this.lines = const [],
    this.conflicts = const [],
    this.perImage = const [],
    this.timings = const {},
    this.screenContext = noScreenContext,
  });
  final ReadField? plate, vin;
  final List<Map<String, dynamic>> lines, conflicts;
  final List<ReadScene> perImage; // one per photo when several were fused; empty for a single photo
  final Map<String, double> timings;

  /// interface / browser text visible in the photo(s): score 0 (nothing) .. 3 (certainly a screen)
  final Map<String, dynamic> screenContext;
}

abstract interface class SceneReaderPort {
  ReadScene readFrames(List<RgbImage> rgbs, {required Set<String> domains, bool allowLegacyVin = false});
}

// ============================================================================== helpers
const knownRoles = {'auto', 'plate', 'plate_flash', 'angle', 'vin'};

int compareRank(Rank a, Rank b) {
  if (a.$1 != b.$1) return a.$1.compareTo(b.$1);
  if (a.$2 != b.$2) return a.$2.compareTo(b.$2);
  return a.$3.compareTo(b.$3);
}

const Rank _noRank = (-1, 0, 0.0);

/// The public shape of one reading (backend `field_out`). `value` only when there is something to
/// act on (accept / review); `masked` is what to DISPLAY unless the status is accept.
Map<String, dynamic> fieldOut(ReadField f, int imageIndex) {
  final d = Map<String, dynamic>.from(f.json);
  d['image'] = imageIndex;
  final actionable = f.status == 'accept' || f.status == 'review';
  d['value'] = actionable ? d['value'] : null;
  final parsed = (d['parsed'] as Map?) ?? const {};
  final display = parsed['display'];
  d['display'] = (display is String && display.isNotEmpty) ? display : d['value'];
  d['needs_review'] = f.status != 'accept';
  if (f.domain == 'vin') {
    d['check_digit_ok'] = parsed['check_digit_ok'];
    d['read_length'] = f.text.length;
  }
  return d;
}

double _ctxScore(Map<String, dynamic> c) => (c['score'] as num?)?.toDouble() ?? 0.0;

String _msg(Object e) => e.toString().replaceFirst(RegExp(r'^(Exception|Error|StateError|Bad state): '), '');

// ============================================================================ the pipeline
class ScanPipeline {
  ScanPipeline({required this.reader, this.pad, DecodeFn? decode, this.padUnavailableReason}) : decode = decode ?? decodePhoto;

  final SceneReaderPort reader;
  final PadScorer? pad;
  final DecodeFn decode;

  /// Why [pad] is null (shown as the reason the recapture check did not run).
  final String? padUnavailableReason;

  Map<String, dynamic> run(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) {
    final total = Stopwatch()..start();
    if (photos.isEmpty) throw ArgumentError('scan needs at least one photo');

    // ---- decode (a bad photo is a FormatException: the worker reports it as such) ---------
    final decoded = [for (final p in photos) decode(p.bytes)];
    final rgbs = [for (final d in decoded) d.rgb];
    final roles = [for (final p in photos) knownRoles.contains(p.role) ? p.role : 'auto'];
    List<int> ids(String role) => [for (var i = 0; i < roles.length; i++) if (roles[i] == role) i];
    final gPlate = ids('plate'), gFlash = ids('plate_flash'), gAngle = ids('angle'), gVin = ids('vin'), gAuto = ids('auto');

    // ---- readings: plate+angle photos are read for a plate, vin photos for a VIN (legacy chassis
    //      numbers allowed), unknown photos for both ----------------------------------------
    final plateIds = [...gPlate, ...gAngle];
    final plateRes = plateIds.isEmpty ? null : reader.readFrames([for (final i in plateIds) rgbs[i]], domains: const {'plate'});
    final vinRes = gVin.isEmpty ? null : reader.readFrames([for (final i in gVin) rgbs[i]], domains: const {'vin'}, allowLegacyVin: true);
    final autoRes = gAuto.isEmpty ? null : reader.readFrames([for (final i in gAuto) rgbs[i]], domains: const {'plate', 'vin'});

    (ReadField?, int) locate(ReadField? f, List<int> idsOf) => (f, f != null && f.image >= 0 && f.image < idsOf.length ? idsOf[f.image] : -1);
    (ReadField?, int) best(List<(ReadField?, int)> c) {
      var b = c.first; // on a tie the FIRST wins, like Python's max()
      for (final x in c.skip(1)) {
        if (compareRank(x.$1?.rank ?? _noRank, b.$1?.rank ?? _noRank) > 0) b = x;
      }
      return b;
    }

    // the scene's screen context: the highest-scoring group (the first on a tie, like Python's max())
    var screenCtx = noScreenContext;
    for (final r in [plateRes, vinRes, autoRes]) {
      if (r != null && _ctxScore(r.screenContext) > _ctxScore(screenCtx)) screenCtx = r.screenContext;
    }

    final (plate, plateImg) = best([locate(plateRes?.plate, plateIds), locate(autoRes?.plate, gAuto)]);
    final (vin, vinImg) = best([locate(vinRes?.vin, gVin), locate(autoRes?.vin, gAuto)]);

    // conflicts between photos of one identifier; mapped to GLOBAL photo indexes
    final conflicts = <Map<String, dynamic>>[];
    for (final (res, idsOf) in [(plateRes, plateIds), (vinRes, gVin), (autoRes, gAuto)]) {
      if (res == null) continue;
      int g(Object? local) => local is int && local >= 0 && local < idsOf.length ? idsOf[local] : -1;
      for (final c in res.conflicts) {
        conflicts.add({
          ...c,
          'image': g(c['image']),
          if (c['agrees_with'] is List) 'agrees_with': [for (final a in c['agrees_with'] as List) g(a)],
        });
      }
    }

    // per-photo results: the plate region of the primary frame feeds the flash test
    final plateBoxes = <int, Box4>{};
    final lines = <Map<String, dynamic>>[];
    for (final (res, idsOf) in [(plateRes, plateIds), (vinRes, gVin), (autoRes, gAuto)]) {
      if (res == null) continue;
      final singles = res.perImage.isEmpty ? [res] : res.perImage;
      for (var k = 0; k < singles.length; k++) {
        final box = singles[k].plate?.box;
        if (box != null && k < idsOf.length) plateBoxes[idsOf[k]] = box;
        for (final ln in singles[k].lines) {
          lines.add({'image': k < idsOf.length ? idsOf[k] : -1, ...ln});
        }
      }
    }
    double sum(String k) => [for (final r in [plateRes, vinRes, autoRes]) if (r != null) r.timings[k] ?? 0.0].fold(0.0, (a, b) => a + b);

    // ---- authenticity ------------------------------------------------------------------
    final tAuth = Stopwatch()..start();
    final primaryIdx = gPlate.isEmpty ? null : gPlate.first;
    final primaryRoi = primaryIdx == null ? null : plateBoxes[primaryIdx];
    // the primary comes first: it is the reference for the flash and parallax comparisons
    final order = [?primaryIdx, for (var i = 0; i < photos.length; i++) if (i != primaryIdx) i];
    String bundleRole(int i) => switch (roles[i]) { 'plate' => i == primaryIdx ? 'primary' : 'other', 'plate_flash' => 'flash_on', 'angle' => 'angle', _ => 'other' };
    String nameOf(int i) => '$i:${photos[i].name}';

    final padWatch = Stopwatch();
    final padNotes = <String>[];
    final frames = <LivenessFrame>[];
    for (final i in order) {
      List<Finding>? prov;
      try {
        prov = scanProvenance(photos[i].bytes);
      } catch (_) {
        prov = null; // an unreadable header is not a finding
      }
      double? score;
      final scorer = pad;
      if (scorer != null) {
        padWatch.start();
        try {
          final roi = plateBoxes[i] ?? (roles[i] == 'plate_flash' ? primaryRoi : null);
          final (s, note) = scorer.score(rgbs[i], roi);
          score = s;
          if (note != null) padNotes.add('[${nameOf(i)}] $note');
        } catch (e) {
          padNotes.add('[${nameOf(i)}] the screen / print model failed: ${_msg(e)}');
        }
        padWatch.stop();
      }
      frames.add(LivenessFrame(role: bundleRole(i), name: nameOf(i), provenance: prov, recapture: score));
    }

    // flash: the first flash-on frame against the primary
    FlashTest? retro;
    String? retroWhy;
    if (gFlash.isNotEmpty) {
      if (primaryIdx == null) {
        retroWhy = 'a flash photo needs a normal plate photo to compare against';
      } else {
        try {
          (retro, retroWhy) = runFlashTest(rgbs[primaryIdx], rgbs[gFlash.first], primaryRoi, hsrpExpected: hsrp);
        } catch (e) {
          retro = null;
          retroWhy = 'the flash test failed: ${_msg(e)}';
        }
      }
    }

    // parallax: the primary against each angle frame
    final plan = <PlanarityOutcome>[];
    final skipped = <String>[];
    for (final i in gAngle) {
      if (primaryIdx == null) {
        plan.add(PlanarityOutcome(nameOf(i), const PlanarityResult('inconclusive', reasons: ['an angle photo needs a normal plate photo to compare against'])));
        continue;
      }
      try {
        final mov = matchSize(rgbs[primaryIdx], rgbs[i]);
        if (mov == null) {
          skipped.add(nameOf(i));
        } else {
          plan.add(PlanarityOutcome(nameOf(i), planarity(rgbs[primaryIdx], mov)));
        }
      } catch (e) {
        plan.add(PlanarityOutcome(nameOf(i), PlanarityResult('inconclusive', reasons: ['the parallax check failed: ${_msg(e)}'])));
      }
    }

    // consistency of the plate read across the plate photos
    final plateReads = <String?>[
      if (plateRes != null) for (final s in (plateRes.perImage.isEmpty ? [plateRes] : plateRes.perImage)) s.plate?.masked,
    ];

    LivenessDecision decision;
    try {
      decision = assessLiveness(
        frames: frames,
        retro: retro,
        retroSkipReason: retroWhy,
        planarity: plan,
        skippedAngleFrames: skipped,
        hsrpExpected: hsrp,
        requireLiveness: requireLiveness,
        readings: plateReads.length > 1 ? plateReads : null,
        screenContext: screenCtx,
      );
    } catch (e) {
      // the decision itself failed: say so, and do not pretend anything was checked
      decision = LivenessDecision(
        status: 'review',
        liveness: 'not_established',
        reasons: ['the authenticity assessment failed: ${_msg(e)}'],
        notRun: const ['recapture', 'retroreflection', 'planarity', 'forensics'],
        evidence: {'not_run_reasons': <String, String>{}, 'on_device': true, 'calibrated': false},
      );
    }
    // say WHY the recapture check did not run, when it did not
    if (decision.notRun.contains('recapture')) {
      final nr = decision.evidence['not_run_reasons'];
      if (nr is Map<String, String>) {
        nr['recapture'] = pad == null ? (padUnavailableReason ?? 'no recapture model loaded') : (padNotes.isNotEmpty ? padNotes.join('; ') : 'inconclusive');
      }
    }
    if (pad != null) {
      for (final n in padNotes) {
        if (!decision.reasons.contains(n)) decision.reasons.add(n);
      }
    }
    tAuth.stop();

    return {
      'fields': {
        'plate': plate == null ? null : fieldOut(plate, plateImg),
        'vin': vin == null ? null : fieldOut(vin, vinImg),
      },
      'authenticity': decision.toJson(),
      'screen_context': screenCtx,
      'conflicts': conflicts,
      'images': [
        for (var i = 0; i < photos.length; i++)
          {'index': i, 'name': photos[i].name, 'role': roles[i], 'taken_at': decoded[i].takenAt, 'size': [rgbs[i].width, rgbs[i].height]},
      ],
      'lines': lines,
      'timings_ms': {
        'detect': sum('detect'),
        'recognise': sum('recognise'),
        'decode': sum('decode'),
        'padnet': padWatch.elapsedMicroseconds / 1000.0,
        'authenticity': tAuth.elapsedMicroseconds / 1000.0,
        'total': total.elapsedMicroseconds / 1000.0,
      },
    };
  }
}
