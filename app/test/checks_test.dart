import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vinplate_tester/checks.dart';

// Toy physics scenes (same model as ml/tests/test_active.py). These prove the Dart port computes
// what the physics says; they do NOT prove it works on real phones, flashes and plates.
const w = 400, h = 240;
const plate = Box(100, 90, 300, 150);

Rgb scene(String kind) {
  final rng = math.Random(3);
  final off = Uint8List(w * h * 3), on = Uint8List(w * h * 3);
  int u8(double lin) => (255 * math.pow(lin.clamp(0.0, 1.0), 1 / 2.2)).round();
  final bytes = <String, Uint8List>{'off': off, 'on': on};
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final inPlate = plate.contains(x, y);
      final base = inPlate ? 0.18 : 0.12 + 0.2 * (0.5 + 0.5 * math.sin(x / 11.0) * math.cos(y / 7.0)) + 0.01 * rng.nextDouble();
      final lit = 0.9 + 0.1 * rng.nextDouble();
      late double l2;
      if (kind == 'real') {
        l2 = inPlate ? base * 1.25 + 1.4 : base * (1 + 0.25 * lit);
      } else if (kind == 'screen') {
        l2 = base + 0.015 * lit;
      } else {
        l2 = base * (1 + 0.25 * lit);
      }
      for (var c = 0; c < 3; c++) {
        off[(y * w + x) * 3 + c] = u8(base);
        on[(y * w + x) * 3 + c] = u8(l2);
      }
    }
  }
  return Rgb(w, h, bytes[kind == 'x' ? 'off' : 'on']!);
}

Rgb offFrame() => Rgb(w, h, (() {
      return Uint8List.fromList(List.generate(w * h * 3, (i) {
        final y = i ~/ 3 ~/ w, x = (i ~/ 3) % w;
        final rng = math.Random(3);
        rng.nextDouble();
        final inPlate = plate.contains(x, y);
        final base = inPlate ? 0.18 : 0.12 + 0.2 * (0.5 + 0.5 * math.sin(x / 11.0) * math.cos(y / 7.0)) + 0.01 * 0.5;
        return (255 * math.pow(base.clamp(0.0, 1.0), 1 / 2.2)).round();
      }));
    })());

void main() {
  test('flash gain separates real HSRP, screen and print', () {
    final off = offFrame();
    expect(retroreflection(off, scene('real'), plate, align: false).verdict, 'retroreflective');
    expect(retroreflection(off, scene('screen'), plate, align: false).verdict, 'not_retroreflective');
    expect(retroreflection(off, scene('print'), plate, align: false).verdict, 'not_retroreflective');
  });

  test('inconclusive (never a fraud signal) for non-HSRP, no flash, or no headroom', () {
    final off = offFrame();
    expect(retroreflection(off, scene('screen'), plate, hsrpExpected: false).verdict, 'inconclusive');
    expect(retroreflection(off, off, plate, align: false).verdict, 'inconclusive');
    final bright = Rgb(w, h, Uint8List.fromList(List.filled(w * h * 3, 170)));
    final brighter = Rgb(w, h, Uint8List.fromList(List.filled(w * h * 3, 200)));
    final r = retroreflection(bright, brighter, plate, align: false);
    expect(r.verdict, 'inconclusive');
    expect(r.reasons.join(), contains('too strong'));
  });

  test('provenance: AI declared is hard, absence is not a pass, no false hit on random bytes', () {
    Uint8List b(String s) => Uint8List.fromList(s.codeUnits);
    final xmp = '<x:xmpmeta Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/></x:xmpmeta>';
    expect(scanProvenance(b('Exif $xmp')).any((f) => f.severity == 'hard'), isTrue);
    expect(scanProvenance(b('Exif <x:xmpmeta>made with Midjourney v6</x:xmpmeta>')).any((f) => f.code == 'ai_marker'), isTrue);
    final rnd = math.Random(1);
    final noise = Uint8List.fromList(List.generate(200000, (_) => rnd.nextInt(256)));
    expect(scanProvenance(noise).any((f) => f.severity == 'hard'), isFalse);
  });

  test('fusion: one weak cue reviews, agreement rejects, missing checks are listed', () {
    final d1 = decide(provenance: [], recaptureScore: 0.6, flashCaptured: false, fromLiveCamera: true);
    expect(d1.status, 'REVIEW');
    final fail = RetroResult('not_retroreflective');
    final d2 = decide(provenance: [], recaptureScore: 0.95, retro: fail, flashCaptured: true, fromLiveCamera: true);
    expect(d2.status, 'REJECT');
    expect(d2.errors.join(), contains('LIVENESS'));
    final d3 = decide(provenance: [], recaptureScore: null, flashCaptured: false, fromLiveCamera: false);
    expect(d3.status, 'CANNOT VERIFY'); // nothing affirmative ran: never a pass
    expect(d3.notRun.length, 3); // screen model, liveness, experimental tamper check (never a pass)
    final d4 = decide(provenance: [], recaptureScore: 0.1, retro: RetroResult('retroreflective'), flashCaptured: true, fromLiveCamera: true);
    expect(d4.status, 'NO FLAGS');
  });

  test('patch picker returns 4 native 96px patches in NCHW and refuses tiny regions', () {
    final rng = math.Random(2);
    final im = Rgb(300, 300, Uint8List.fromList(List.generate(300 * 300 * 3, (_) => rng.nextInt(256))));
    final p = pickPatches(im, const Box(0, 0, 300, 300))!;
    expect(p.length, 4 * 3 * 96 * 96);
    expect(pickPatches(im, const Box(0, 0, 60, 60)), isNull);
  });
}
