import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;

import 'checks.dart';
import 'id_reader.dart';
import 'native.dart';

class BackendTiming {
  final Backend backend;
  final double? medianMs, minMs;
  final String? error;
  BackendTiming(this.backend, {this.medianMs, this.minMs, this.error});
}

class Report {
  final Decision decision;
  final double? recaptureScore;
  final RetroResult? retro;
  final List<Finding> provenance;
  final List<BackendTiming> timings;
  final String model;
  final int imageW, imageH;
  final Uint8List offBytes;
  final Uint8List? onBytes;
  final double? tamperScore;
  final String located; // where the regions came from, shown in the Checks tab
  Report(this.decision, this.recaptureScore, this.retro, this.provenance, this.timings, this.model, this.imageW, this.imageH, this.offBytes, this.onBytes, this.tamperScore, this.located);
}

Rgb _decode(Uint8List bytes, {int? maxW}) {
  var im = img.decodeImage(bytes);
  if (im == null) throw const FormatException('could not decode image');
  im = img.bakeOrientation(im);
  if (maxW != null && im.width > maxW) im = img.copyResize(im, width: maxW);
  return Rgb(im.width, im.height, im.convert(numChannels: 3).getBytes(order: img.ChannelOrder.rgb));
}

// top-level so it can run in an isolate (decoding + pixel loops would jank the UI)
Map<String, dynamic> _prepare(Map<String, dynamic> a) {
  final off = _decode(a['off'] as Uint8List);
  final prov = scanProvenance(a['off'] as Uint8List);
  final cx = off.w ~/ 2, cy = off.h ~/ 2;
  Box? grow(List? v, double fx, double fy) {
    if (v == null) return null;
    final w = v[2] - v[0], h = v[3] - v[1];
    return Box((v[0] - w * fx).round(), (v[1] - h * fy).round(), (v[2] + w * fx).round(), (v[3] + h * fy).round());
  }
  // The plate sheeting extends beyond the characters, so grow the OCR text box a little.
  final located = grow(a['plateBox'] as List?, 0.12, 0.45);
  final plate = located ?? Box(cx - (off.w * 0.30).round(), cy - (off.h * 0.09).round(), cx + (off.w * 0.30).round(), cy + (off.h * 0.09).round());
  final tamperPlate = a['plateBox'] == null ? null : patchesAround(off, grow(a['plateBox'] as List, 0.0, 0.0)!);
  final tamperVin = a['vinBox'] == null ? null : patchesAround(off, grow(a['vinBox'] as List, 0.0, 0.0)!);
  // patches come from the plate and its immediate surround (the object being claimed), never from
  // whatever else is textured in the frame (a blanket, a keyboard)
  final roi = located == null
      ? Box(cx - (off.w * 0.45).round(), cy - (off.h * 0.12).round(), cx + (off.w * 0.45).round(), cy + (off.h * 0.12).round())
      : Box(located.x0 - 40, located.y0 - 40, located.x1 + 40, located.y1 + 40);
  final patches = pickPatches(off, roi);
  RetroResult? retro;
  if (a['on'] != null) {
    final on = _decode(a['on'] as Uint8List);
    if (on.w == off.w && on.h == off.h) {
      retro = retroreflection(off, on, plate, hsrpExpected: a['hsrp'] as bool);
    }
  }
  return {'prov': prov, 'patches': patches, 'retro': retro, 'w': off.w, 'h': off.h, 'tamper': [tamperPlate, tamperVin], 'located': a['plateBox'] != null};
}

class Pipeline {
  static const modelInt8 = 'assets/models/pad_int8.onnx';
  static const modelFp32 = 'assets/models/pad_fp32.onnx';
  static const tamperModel = 'assets/models/tamper_fp32.onnx';

  /// [off]: flash-off JPEG bytes (primary). [on]: flash-on JPEG bytes (live capture only).
  static Future<Report> run({
    required Uint8List off,
    Uint8List? on,
    required bool fromLiveCamera,
    required Backend backend,
    required bool hsrpExpected,
    IdResult? id,
    bool int8 = true,
  }) async {
    List<int>? bx(Box? b) => b == null ? null : [b.x0, b.y0, b.x1, b.y1];
    final prep = await compute(_prepare, {'off': off, 'on': on, 'hsrp': hsrpExpected, 'plateBox': bx(id?.plateBox), 'vinBox': bx(id?.vinBox)});
    final prov = (prep['prov'] as List).cast<Finding>();
    final patches = prep['patches'] as Float32List?;
    final retro = prep['retro'] as RetroResult?;
    final model = int8 ? modelInt8 : modelFp32;

    double? score;
    final timings = <BackendTiming>[];
    if (patches != null) {
      // Run both backends so every test also reports CPU vs GPU/NNAPI; the decision uses [backend].
      for (final b in Backend.values) {
        try {
          final r = await Ort.run(asset: model, backend: b, input: patches, shape: const [4, 3, 96, 96], repeats: b == backend ? 12 : 6);
          timings.add(BackendTiming(b, medianMs: r.medianMs, minMs: r.minMs));
          if (b == backend) {
            score = sigmoid(r.output.reduce((x, y) => x + y) / r.output.length);
          }
        } catch (e) {
          timings.add(BackendTiming(b, error: e.toString()));
        }
      }
      if (score == null) {
        throw StateError('Selected backend ${backend.name} failed: '
            '${timings.firstWhere((t) => t.backend == backend).error}');
      }
    }
    // Text-edit check on whichever of plate / VIN was located; the worst region counts.
    double? tamper;
    var haveTamperModel = true;
    try {
      await rootBundle.load(tamperModel); // shipped only if it passed its held-out test
    } catch (_) {
      haveTamperModel = false;
    }
    for (final tp in (prep['tamper'] as List).cast<Float32List?>()) {
      if (tp == null || !haveTamperModel) continue;
      try {
        final r = await Ort.run(asset: tamperModel, backend: Backend.cpu, input: tp, shape: const [4, 3, 96, 96]);
        final sc = sigmoid(r.output.reduce((x, y) => x + y) / r.output.length);
        tamper = tamper == null ? sc : (sc > tamper ? sc : tamper);
      } catch (e) {
        timings.add(BackendTiming(Backend.cpu, error: 'tamper model: $e'));
      }
    }
    final d = decide(
      tamperScore: tamper,
      located: prep['located'] as bool,
      provenance: prov,
      recaptureScore: score,
      retro: retro,
      flashCaptured: on != null,
      fromLiveCamera: fromLiveCamera,
    );
    if (patches == null) d.notRun.add('screen/print model: image too small for a 96px native patch');
    return Report(d, score, retro, prov, timings, int8 ? 'PadNet INT8' : 'PadNet FP32', prep['w'] as int, prep['h'] as int, off, on, tamper,
        (prep['located'] as bool) ? 'plate region found by OCR' : 'plate NOT located: used the guide box');
  }
}

String fmtMs(double? v) => v == null ? 'n/a' : '${v.toStringAsFixed(v < 10 ? 2 : 1)} ms';
double clamp01(double v) => math.min(1, math.max(0, v));
