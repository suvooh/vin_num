import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'engine/ort.dart';
import 'engine/types.dart';

enum Backend { cpu, nnapi }

class OrtRun {
  final List<double> output;
  final double medianMs, minMs;
  OrtRun(this.output, this.medianMs, this.minMs);
}

/// Screen/print model runner on the shared ONNX Runtime engine (engine/ort.dart).
/// A backend that cannot initialise throws - callers must show it, never substitute silently.
class Ort {
  static final _cache = <String, Model>{};

  static Future<Model> _model(String asset, Backend b) async {
    final key = '$asset|${b.name}';
    final hit = _cache[key];
    if (hit != null) return hit;
    final bytes = (await rootBundle.load(asset)).buffer.asUint8List();
    final want = b == Backend.cpu ? Accelerator.xnnpack : Accelerator.nnapi;
    final m = Model.load(asset, bytes, accelerator: want);
    if (b == Backend.nnapi && m.accelerator != Accelerator.nnapi) {
      m.release();
      throw StateError('NNAPI could not be enabled on this device (got ${m.accelerator.name})');
    }
    return _cache[key] = m;
  }

  static Future<OrtRun> run({
    required String asset,
    required Backend backend,
    required Float32List input,
    required List<int> shape,
    int repeats = 1,
  }) async {
    final m = await _model(asset, backend);
    final times = <double>[];
    var out = <double>[];
    for (var i = 0; i < repeats; i++) {
      final sw = Stopwatch()..start();
      final r = m.run(m.inputNames.first, input, shape);
      sw.stop();
      times.add(sw.elapsedMicroseconds / 1000.0);
      out = r.values.first.$1.map((e) => e.toDouble()).toList();
    }
    times.sort();
    return OrtRun(out, times[times.length ~/ 2], times.first);
  }
}
