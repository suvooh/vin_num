import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/checks.dart';
import 'package:vinplate_tester/data/models.dart';
import 'package:vinplate_tester/engine/types.dart';
import 'package:vinplate_tester/reader/liveness.dart';
import 'package:vinplate_tester/reader/planarity.dart';

Map<String, dynamic> loadGolden(String name) => (jsonDecode(File('test/fixtures/golden/$name.json').readAsStringSync()) as Map).cast<String, dynamic>();

RgbImage gray(String b64, int w, int h) {
  final g = zlib.decode(base64Decode(b64));
  final out = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    out[3 * i] = g[i];
    out[3 * i + 1] = g[i];
    out[3 * i + 2] = g[i];
  }
  return RgbImage(w, h, out);
}

FlashTest retroFrom(Map<String, dynamic> j) => FlashTest(j['verdict'] as String,
    ratio: (j['ratio'] as num?)?.toDouble(),
    gainPlate: (j['gain_plate'] as num?)?.toDouble(),
    gainSurround: (j['gain_surround'] as num?)?.toDouble(),
    reasons: [for (final r in (j['reasons'] as List)) '$r'],
    gainSpread: (j['gain_spread'] as num?)?.toDouble(),
    hotspot: (j['hotspot'] as bool?) ?? false);

PlanarityResult planFrom(Map<String, dynamic> j) => PlanarityResult(j['verdict'] as String,
    residualPx: (j['residual_px'] as num?)?.toDouble(),
    baselinePx: (j['baseline_px'] as num?)?.toDouble(),
    tiles: j['tiles'] as int,
    inlierFrac: (j['inlier_frac'] as num?)?.toDouble(),
    reasons: [for (final r in (j['reasons'] as List)) '$r']);

Finding hardFinding() => Finding('xmp_screencapture', 'hard', 'XMP DigitalSourceType = screencapture');
Finding softFinding() => Finding('editor_marker', 'soft', 'metadata mentions an image editor (photoshop)');

LivenessFrame f(String role, String name, {double? rec, List<Finding>? prov}) => LivenessFrame(role: role, name: name, recapture: rec, provenance: prov);

final retroOk = FlashTest('retroreflective', ratio: 4.2, gainPlate: 5.0, gainSurround: 1.19, gainSpread: 1.3);
final retroBad = FlashTest('not_retroreflective', ratio: 1.0, gainPlate: 1.1, gainSurround: 1.1, reasons: ['plate brightened 1.10x under flash']);
final retroSpot = FlashTest('specular_hotspot', ratio: 1.2, gainPlate: 1.3, gainSurround: 1.1, gainSpread: 9.0, hotspot: true, reasons: ['the flash lit a small hot-spot']);

/// what the scene reader reports for a photo of a monitor / of a weakly suspicious scene / of a plain car scene
const screenCtx = {'score': 3.0, 'url': true, 'ui_words': ['SEARCH', 'SETTINGS', 'YOUTUBE'], 'text_lines': 16, 'reasons': ['a web address (URL) is visible in the scene']};
const weakCtx = {'score': 0.5, 'url': false, 'ui_words': <String>[], 'text_lines': 14, 'reasons': ['14 separate lines of text are visible']};
const noCtx = {'score': 0.0, 'url': false, 'ui_words': <String>[], 'text_lines': 0, 'reasons': <String>[]};
const nonPlanar = PlanarityResult('non_planar', baselinePx: 6.0, tiles: 80, inlierFrac: 0.4);
const flatPlanar = PlanarityResult('planar', baselinePx: 4.0, tiles: 80, inlierFrac: 0.97, reasons: ['97% of tiles move as ONE plane']);

void main() {
  // =========================================================================== golden: decision logic
  group('golden: assess_bundle (Python, stubbed checks) vs assessLiveness (Dart)', () {
    final cases = (loadGolden('liveness_cases')['cases'] as List).cast<Map<String, dynamic>>();
    test('there are cases to compare', () => expect(cases.length, greaterThanOrEqualTo(25)));

    for (final c in cases) {
      test(c['name'] as String, () {
        final frames = [
          for (final fr in (c['frames'] as List).cast<Map<String, dynamic>>())
            LivenessFrame(
              role: fr['role'] as String,
              name: fr['name'] as String,
              recapture: (fr['recapture'] as num?)?.toDouble(),
              provenance: fr['provenance'] == null
                  ? null
                  : [for (final p in (fr['provenance'] as List).cast<Map<String, dynamic>>()) Finding(p['code'] as String, p['severity'] as String, p['detail'] as String)],
            )
        ];
        final angleNames = [for (final fr in frames) if (fr.role == 'angle') fr.name];
        final skipped = [for (final s in (c['skipped_angle_frames'] as List)) '$s'];
        final plans = (c['planarity'] as List).cast<Map<String, dynamic>>();
        final compared = angleNames.where((n) => !skipped.contains(n)).toList();
        final d = assessLiveness(
          frames: frames,
          retro: c['retro'] == null ? null : retroFrom((c['retro'] as Map).cast<String, dynamic>()),
          retroSkipReason: c['retro_skip_reason'] as String?,
          planarity: [for (var i = 0; i < plans.length; i++) PlanarityOutcome(compared[i], planFrom(plans[i]))],
          skippedAngleFrames: skipped,
          hsrpExpected: c['hsrp'] as bool,
          requireLiveness: c['require_liveness'] as bool,
          readings: (c['readings'] as List?)?.map((e) => e as String?).toList(),
          screenContext: (c['screen_context'] as Map?)?.cast<String, dynamic>(),
        );
        final exp = (c['expect'] as Map).cast<String, dynamic>();
        expect(d.status, exp['status'], reason: 'status');
        expect(d.liveness, exp['liveness'], reason: 'liveness');
        expect(d.paScore, closeTo((exp['pa_score'] as num).toDouble(), 1e-9), reason: 'pa_score');
        expect(d.positive.toSet(), {for (final x in exp['positive'] as List) '$x'}, reason: 'positive');
        // Dart additionally lists 'forensics' (not available on device); Python ran its forensics
        expect(d.notRun.where((x) => x != 'forensics').toSet(), {for (final x in exp['not_run'] as List) '$x'}, reason: 'not_run');
        expect(d.evidence['liveness_basis'], [for (final x in exp['basis'] as List) '$x'], reason: 'liveness basis');
        final why = Map<String, dynamic>.of((d.evidence['not_run_reasons'] as Map).cast<String, dynamic>())..remove('forensics');
        expect(why, (exp['not_run_reasons'] as Map).cast<String, dynamic>(), reason: 'not_run_reasons');
        expect(d.reasons.toSet(), {for (final x in exp['reasons'] as List) '$x'}, reason: 'reasons (wording is part of the contract)');
        expect(d.evidence['read_conflicts'], [for (final x in exp['read_conflicts'] as List) '$x']);
      });
    }
  });

  // ======================================================================== flash test cross-check
  group('flash test: judgeFlash (checks.dart retroreflection + uniformity) vs Python active.retroreflection on the same frames', () {
    final data = loadGolden('liveness_retro');
    final w = data['w'] as int, h = data['h'] as int;
    for (final e in (data['cases'] as Map).entries) {
      test('${e.key}', () {
        final c = (e.value as Map).cast<String, dynamic>();
        final b = (c['box'] as List).cast<int>();
        final r = judgeFlash(Rgb(w, h, gray(c['off'] as String, w, h).rgb), Rgb(w, h, gray(c['on'] as String, w, h).rgb), Box(b[0], b[1], b[2], b[3]), hsrpExpected: c['hsrp'] as bool);
        final exp = (c['expect'] as Map).cast<String, dynamic>();
        expect(r.verdict, exp['verdict'], reason: 'verdict (Dart: ${r.reasons}; Python: ${exp['reasons']})');
        if (exp['verdict'] == 'specular_hotspot') expect(r.hotspot, isTrue);
        if (exp['gain_spread'] != null && r.gainSpread != null) {
          // the same arithmetic over the same pixels, but a different alignment (SAD vs phase correlation)
          expect(r.gainSpread!, closeTo(exp['gain_spread'] as double, 0.25 * (exp['gain_spread'] as double)), reason: 'gain spread p90/p10');
        }
        if (exp['ratio'] != null && r.ratio != null) {
          // different alignment (SAD vs phase correlation) and ring geometry (boxes vs mask dilation):
          // the gains agree to within ~20%, not bit-for-bit
          expect(r.gainPlate!, closeTo(exp['gain_plate'] as double, 0.2 * (exp['gain_plate'] as double)), reason: 'plate gain');
          expect(r.gainSurround!, closeTo(exp['gain_surround'] as double, 0.2 * (exp['gain_surround'] as double)), reason: 'surround gain');
        }
        if (exp['verdict'] == 'inconclusive') expect(r.reasons, isNotEmpty, reason: 'an inconclusive flash test must say why');
      });
    }

    test('runFlashTest wraps it with the bundle rules: eroded plate box, ROI required, aspect-ratio check', () {
      final c = (data['cases'] as Map)['real_hsrp'] as Map;
      final off = gray(c['off'] as String, w, h), on = gray(c['on'] as String, w, h);
      final (r, why) = runFlashTest(off, on, (100, 90, 300, 150), hsrpExpected: true);
      expect(why, isNull);
      expect(r!.verdict, 'retroreflective');
      expect(runFlashTest(off, on, null, hsrpExpected: true).$1, isNull);
      expect(runFlashTest(off, on, null, hsrpExpected: true).$2, contains('no plate region'));
      final portrait = RgbImage(h, w, Uint8List(h * w * 3));
      expect(runFlashTest(off, portrait, (100, 90, 300, 150), hsrpExpected: true).$2, contains('different aspect ratio'));
      expect(erodedPlateBox((100, 90, 300, 150)).x0, 120); // 10% of 200
      expect(erodedPlateBox((100, 90, 300, 150)).y1, 144); // 10% of 60 = 6
    });

    test('matchSize resizes frames of the same aspect ratio and refuses a different one', () {
      final a = RgbImage(40, 30, Uint8List(40 * 30 * 3)), b = RgbImage(80, 60, Uint8List.fromList(List.filled(80 * 60 * 3, 200)));
      final m = matchSize(a, b)!;
      expect((m.width, m.height), (40, 30));
      expect(m.rgb[0], 200);
      expect(matchSize(a, RgbImage(30, 40, Uint8List(30 * 40 * 3))), isNull);
      expect(identical(matchSize(a, a), a), isTrue);
    });
  });

  // ============================================================================== decision table
  group('decision table (mirrors ml/tests/test_bundle.py)', () {
    test('genuine HSRP bundle: confirmed through the flash test, parallax listed but not the basis', () {
      final d = assessLiveness(
          frames: [f('primary', 'front'), f('flash_on', 'flash'), f('angle', 'left')], retro: retroOk, planarity: [const PlanarityOutcome('left', nonPlanar)], hsrpExpected: true);
      expect(d.liveness, 'confirmed');
      expect(d.status, isNot('reject'));
      expect(d.evidence['liveness_basis'], ['retroreflection_passed']);
      expect(d.positive, containsAll(['retroreflection_passed', 'parallax_non_planar']));
      expect(d.notRun, isNot(contains('retroreflection')));
    });

    test('screen / print replay with a flash pair is a suspected spoof, never confirmed', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroBad, hsrpExpected: true);
      expect(d.liveness, 'suspected_spoof');
      expect(d.status, anyOf('review', 'reject'));
      expect(d.paScore, greaterThanOrEqualTo(2.0));
      expect(d.positive, isNot(contains('retroreflection_passed')));
    });

    test('replay + a high learned recapture score is rejected', () {
      final d = assessLiveness(
          frames: [f('primary', 'front', rec: 0.97), f('flash_on', 'flash', rec: 0.97)], retro: retroBad, planarity: const [], hsrpExpected: true);
      expect(d.status, 'reject');
      expect(d.liveness, 'suspected_spoof');
      expect(d.paScore, greaterThanOrEqualTo(3.0));
    });

    test('parallax alone does NOT confirm liveness (a screen in front of a real background has depth)', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('angle', 'left')], planarity: [const PlanarityOutcome('left', nonPlanar)]);
      expect(d.liveness, 'not_established');
      expect(d.evidence['liveness_basis'], isEmpty);
      expect(d.positive, contains('parallax_non_planar'));
      expect(d.reasons.join(' '), contains('depth alone does not establish liveness'));
    });

    test('parallax + a CLEAN learned-recapture score does NOT confirm either (PadNet scored a real monitor photo 0.003)', () {
      final d = assessLiveness(
          frames: [f('primary', 'front', rec: 0.05), f('angle', 'left', rec: 0.05)], planarity: [const PlanarityOutcome('left', nonPlanar)]);
      expect(d.liveness, 'not_established');
      expect(d.evidence['liveness_basis'], isEmpty);
      expect(d.positive, contains('parallax_non_planar')); // listed as supporting evidence only
      expect(d.reasons.join(' '), contains('a clean score is never used as proof of a live scene'));
      expect(d.reasons.join(' '), contains('only the flash retroreflection test can confirm it'));
    });

    test('a clean learned-recapture score alone is never a basis (and never a positive)', () {
      final d = assessLiveness(frames: [f('primary', 'front', rec: 0.003)]);
      expect(d.liveness, 'not_established');
      expect(d.positive, isEmpty);
      expect(d.evidence['liveness_basis'], isEmpty);
    });

    test('the real-world failure: a photo of a monitor, parallax from a real background, PadNet 0.003 -> NOT confirmed, suspected spoof', () {
      final d = assessLiveness(
          frames: [f('primary', 'front', rec: 0.003), f('angle', 'left', rec: 0.004), f('angle', 'right', rec: 0.002)],
          planarity: [const PlanarityOutcome('left', nonPlanar), const PlanarityOutcome('right', nonPlanar)],
          screenContext: screenCtx);
      expect(d.liveness, 'suspected_spoof');
      expect(d.liveness, isNot('confirmed'));
      expect(d.evidence['liveness_basis'], isEmpty);
      expect(d.status, 'reject'); // context 3.0 >= rejectAt
      expect(d.paScore, 3.0);
      expect(d.reasons.join(' '), contains('screen / interface context in the scene (+3): a web address (URL) is visible in the scene'));
      expect(d.evidence['screen_context'], screenCtx);
    });

    test('screen text at a score of 1.0 or more makes a suspected spoof by itself, even next to a PASSED flash test', () {
      final d = assessLiveness(
          frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: true, screenContext: {...weakCtx, 'score': 1.0});
      expect(d.liveness, 'suspected_spoof');
      expect(d.evidence['liveness_basis'], ['retroreflection_passed']); // the cue was seen, but it does not outweigh the screen
      expect(d.status, 'review');
    });

    test('weak screen context (< 1.0) is a hint that adds to the score but does not make a spoof by itself', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: true, screenContext: weakCtx);
      expect(d.paScore, 0.5);
      expect(d.liveness, 'confirmed');
      expect(d.reasons.join(' '), contains('weak presentation-attack hints only: screen / interface context in the scene (+0.5): 14 separate lines of text are visible'));
    });

    test('no screen context changes nothing', () {
      final a = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: true);
      final b = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: true, screenContext: noCtx);
      expect((b.status, b.liveness, b.paScore), (a.status, a.liveness, a.paScore));
      expect(b.reasons, a.reasons);
    });

    test('a specular hot-spot under flash is glare on glass: +1.0, a suspected spoof, never confirmed', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroSpot, hsrpExpected: true);
      expect(d.liveness, 'suspected_spoof');
      expect(d.paScore, 1.0);
      expect(d.status, 'review');
      expect(d.positive, isNot(contains('retroreflection_passed')));
      expect(d.evidence['liveness_basis'], isEmpty);
      expect(d.reasons.join(' '), contains('plate showed a specular hot-spot under flash, not a retroreflective response (+1.0)'));
      expect(d.notRun, isNot(contains('retroreflection'))); // the test ran
      final active = (d.evidence['active'] as Map)['retroreflection'] as Map;
      expect(active['verdict'], 'specular_hotspot');
      expect(active['hotspot'], isTrue);
      expect(active['gain_spread'], 9.0);
    });

    test('confirmation needs a PASSED flash test on an HSRP plate: parallax, clean PadNet and a calm scene together are not enough', () {
      final d = assessLiveness(
          frames: [f('primary', 'front', rec: 0.01), f('flash_on', 'flash', rec: 0.01), f('angle', 'left', rec: 0.01)],
          retro: FlashTest.fromRetro(RetroResult('inconclusive', reasons: ['ratio in the grey zone'])),
          planarity: [const PlanarityOutcome('left', nonPlanar)],
          hsrpExpected: true,
          screenContext: noCtx);
      expect(d.liveness, 'not_established');
    });

    test('...nor when any frame looks suspicious', () {
      final d = assessLiveness(
          frames: [f('primary', 'front', rec: 0.05), f('angle', 'left', rec: 0.6)], planarity: [const PlanarityOutcome('left', nonPlanar)]);
      expect(d.liveness, isNot('confirmed'));
      expect(d.paScore, 1.0);
    });

    test('a flat angle pair is one weak cue, counted once however many angles agree, and never confirms', () {
      final d = assessLiveness(
          frames: [f('primary', 'front'), f('angle', 'a'), f('angle', 'b')],
          planarity: [const PlanarityOutcome('a', flatPlanar), const PlanarityOutcome('b', flatPlanar)]);
      expect(d.paScore, 0.5);
      expect(d.liveness, 'not_established');
    });

    test('a single image: no_flags is NOT liveness, and it says so', () {
      final d = assessLiveness(frames: [f('primary', 'front')]);
      expect(d.status, 'no_flags');
      expect(d.liveness, 'not_established');
      expect(d.positive, isEmpty);
      expect(d.reasons.join(' '), contains('NO positive evidence of a live scene'));
      expect(d.notRun, containsAll(['retroreflection', 'planarity', 'recapture', 'forensics']));
      expect((d.evidence['not_run_reasons'] as Map)['retroreflection'], 'no flash-on frame in the bundle');
      expect((d.evidence['not_run_reasons'] as Map)['planarity'], 'no angle frames in the bundle');
      expect((d.evidence['not_run_reasons'] as Map)['forensics'], 'pixel forensics not available on device');
    });

    test('nothing is silently a pass: the two detectors that do not exist are always "not run"', () {
      final d = assessLiveness(frames: [f('primary', 'front', rec: 0.01)]);
      expect(d.notRun, containsAll(['tamper', 'ai_generated']));
    });

    test('require_liveness raises an unproven capture to review, never lowers a reject', () {
      expect(assessLiveness(frames: [f('primary', 'front')], requireLiveness: true).status, 'review');
      expect(assessLiveness(frames: [f('primary', 'front', prov: [hardFinding()])], requireLiveness: true).status, 'reject');
      final ok = assessLiveness(
          frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: true, requireLiveness: true);
      expect(ok.liveness, 'confirmed');
      expect(ok.status, 'no_flags');
    });

    test('a retroreflective result on a non-HSRP plate is a positive hint but never the basis of liveness', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroOk, hsrpExpected: false);
      expect(d.liveness, 'not_established');
      expect(d.positive, contains('retroreflection_passed'));
    });

    test('conflicting plate reads between frames raise a review; "?" matches anything', () {
      final bad = assessLiveness(frames: [f('primary', 'front'), f('other', 'second')], readings: ['MH12AB1234', 'KA01XY9876']);
      expect(bad.status, 'review');
      expect(bad.reasons.join(), contains("plate text differs between frames: 'MH12AB1234' (front) vs 'KA01XY9876' (second)"));
      final ok = assessLiveness(frames: [f('primary', 'front'), f('other', 'second')], readings: ['MH12?B1234', 'MH12AB1234']);
      expect(ok.status, 'no_flags');
    });

    test('a hard provenance flag (declared screen capture / AI) rejects and marks a spoof; a soft one only reviews', () {
      final h = assessLiveness(frames: [f('primary', 'front', prov: [hardFinding()])]);
      expect((h.status, h.liveness), ('reject', 'suspected_spoof'));
      expect(h.reasons.join(), contains('xmp_screencapture: [front] XMP DigitalSourceType = screencapture'));
      final s = assessLiveness(frames: [f('primary', 'front', prov: [softFinding()])]);
      expect((s.status, s.liveness), ('review', 'not_established'));
    });

    test('the same finding in two frames is listed once, tagged with both', () {
      final d = assessLiveness(frames: [f('primary', 'a', prov: [hardFinding()]), f('other', 'b', prov: [hardFinding()])]);
      expect(d.reasons.where((r) => r.startsWith('xmp_screencapture')).length, 1);
      expect(d.reasons.join(), contains('[a, b]'));
    });

    test('a flash frame whose test could not run says why, and is not a pass', () {
      final d = assessLiveness(
          frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: null, retroSkipReason: 'no plate region (ROI) for the primary frame, so no plate mask can be built', hsrpExpected: true);
      expect(d.notRun, contains('retroreflection'));
      expect(d.liveness, 'not_established');
      expect((d.evidence['not_run_reasons'] as Map)['retroreflection'], contains('no plate region'));
    });

    test('the recapture score of a bundle is the MAX over frames', () {
      final d = assessLiveness(frames: [f('primary', 'a', rec: 0.1), f('other', 'b', rec: 0.95)]);
      expect(d.paScore, 2.0);
      expect(d.reasons.join(), contains("highest learned recapture score is in frame 'b' (0.95)"));
    });

    test('unknown roles and an empty bundle are programming errors', () {
      expect(() => assessLiveness(frames: const []), throwsArgumentError);
      expect(() => assessLiveness(frames: [f('selfie', 'x')]), throwsArgumentError);
    });
  });

  // ============================================================================ read consistency
  group('read consistency (bundle._compatible)', () {
    test('compatibility', () {
      expect(readsCompatible('MH12AB1234', 'MH12AB1234'), isTrue);
      expect(readsCompatible('MH12?B1234', 'MH12AB1234'), isTrue);
      expect(readsCompatible('MH12AB1234', 'MH12AB123'), isFalse); // a known character cannot be dropped
      expect(readsCompatible('MH12?AB1234', 'MH12AB1234'), isTrue); // '?' may stand for nothing (a suspected gap)
      expect(readsCompatible('MH12AB1234', 'KA01XY9876'), isFalse);
      expect(readsCompatible('', ''), isTrue);
    });

    test('conflicts are labelled; empty reads are ignored', () {
      expect(readConflicts(['MH12AB1234', null, '  ', 'KA01XY9876'], ['p', 'q', 'r', 's']),
          ["plate text differs between frames: 'MH12AB1234' (p) vs 'KA01XY9876' (s)"]);
      expect(readConflicts(['mh 12 ab 1234', 'MH12AB1234']), isEmpty);
    });
  });

  // ============================================================================== wire format
  group('wire format', () {
    test('the JSON is what the app\'s Authenticity.fromJson reads, and round-trips', () {
      final d = assessLiveness(
          frames: [f('primary', 'front'), f('flash_on', 'flash'), f('angle', 'left')], retro: retroOk, planarity: [const PlanarityOutcome('left', nonPlanar)], hsrpExpected: true);
      final j = jsonDecode(jsonEncode(d.toJson())) as Map<String, dynamic>; // must be plain JSON
      final a = Authenticity.fromJson(j);
      expect(a.status, d.status);
      expect(a.liveness, 'confirmed');
      expect(a.isConfirmed, isTrue);
      expect(a.positive, d.positive);
      expect(a.notRun, d.notRun);
      expect(a.paScore, d.paScore);
      expect(a.reasons, d.reasons);
      expect(a.notRunReasons, isA<Map<String, String>>());
      expect(a.notRunReasons.containsKey('forensics'), isTrue);
    });

    test('a suspected spoof is reported as such by the app model', () {
      final d = assessLiveness(frames: [f('primary', 'front'), f('flash_on', 'flash')], retro: retroBad, hsrpExpected: true);
      final a = Authenticity.fromJson(jsonDecode(jsonEncode(d.toJson())) as Map<String, dynamic>);
      expect(a.isSpoofSuspected, isTrue);
      expect(a.isConfirmed, isFalse);
    });
  });
}
