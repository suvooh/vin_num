// The screen / print ("recapture") detector on the phone: PadNet INT8 on 4 native-resolution 96x96
// patches. Mirrors lib/pipeline.dart (same model, input shape and score) and Python's
// `Authenticator.recapture_score` (patches from the plate region; when the plate is smaller than a
// patch, judge a window around it instead of giving up).
import 'dart:typed_data';

import '../checks.dart' show Box, Rgb, pickPatches, sigmoid;
import '../engine/ort.dart';
import '../engine/types.dart';

const _patch = 96;

/// Scores one frame: (probability the frame is a screen / print, or null; a note or reason).
abstract interface class PadScorer {
  (double?, String?) score(RgbImage frame, (int, int, int, int)? roi);
  void dispose();
}

/// Patch selection + score, with the model run injected so it is testable without native code.
/// [roi] null = the whole frame. Port of `Authenticator.recapture_score` (+ `pick_patches`).
(double?, String?) recaptureScore(RgbImage frame, (int, int, int, int)? roi, double Function(Float32List patches) run) {
  final im = Rgb(frame.width, frame.height, frame.rgb);
  String? note;
  Float32List? patches;
  if (roi == null) {
    patches = pickPatches(im, Box(0, 0, frame.width, frame.height));
  } else {
    final (x0, y0, x1, y1) = roi;
    patches = pickPatches(im, Box(x0, y0, x1, y1));
    if (patches == null) {
      // A plate in a full-car framing is often smaller than one native patch. Screen / print artefacts
      // (moire, halftone, glare) cover the whole recaptured frame, so judge a window at least 3 patches
      // across around the plate instead of giving up.
      final cx = (x0 + x1) ~/ 2, cy = (y0 + y1) ~/ 2, half = (1.5 * _patch).toInt();
      patches = pickPatches(
          im,
          Box(cx - half < 0 ? 0 : cx - half, cy - half < 0 ? 0 : cy - half, cx + half > frame.width ? frame.width : cx + half,
              cy + half > frame.height ? frame.height : cy + half));
      if (patches != null) note = 'plate smaller than one native patch: recapture model judged the surrounding area';
    }
  }
  if (patches == null) return (null, 'recapture model not run: region smaller than one native patch');
  return (run(patches), note);
}

/// PadNet through ONNX Runtime. XNNPACK with automatic CPU fallback (fixed input shape, unlike PP-OCR).
class PadNetScorer implements PadScorer {
  PadNetScorer._(this._model);
  final Model _model;

  /// Loads the model; null (never throws) when it cannot be loaded - the caller then reports the
  /// check as not run. [error] receives the reason.
  static PadNetScorer? tryLoad(Uint8List? bytes, {void Function(String reason)? error}) {
    if (bytes == null || bytes.isEmpty) {
      error?.call('the screen / print model is not bundled with this build');
      return null;
    }
    try {
      final m = Model.load('pad', bytes, accelerator: Accelerator.xnnpack);
      final s = PadNetScorer._(m);
      try {
        s._run(Float32List(4 * 3 * _patch * _patch)); // warm-up: the first run compiles kernels
      } catch (_) {}
      return s;
    } catch (e) {
      error?.call('the screen / print model could not be loaded: $e');
      return null;
    }
  }

  double _run(Float32List patches) {
    final out = _model.run(_model.inputNames.first, patches, const [4, 3, _patch, _patch]).values.first.$1;
    var sum = 0.0;
    for (final v in out) {
      sum += v;
    }
    return sigmoid(sum / out.length);
  }

  @override
  (double?, String?) score(RgbImage frame, (int, int, int, int)? roi) => recaptureScore(frame, roi, _run);

  @override
  void dispose() => _model.release();
}
