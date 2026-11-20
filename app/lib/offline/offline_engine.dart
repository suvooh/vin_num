// The on-device pipeline: PP-OCR (CPU) + grammar decoder + screen/print model + flash / parallax /
// liveness decision, all inside the worker isolate. This file is only the WIRING of the models;
// the orchestration logic lives in scan_pipeline.dart.
import 'dart:typed_data';

import '../data/scanner.dart';
import '../engine/types.dart';
import '../reader/pp_engine.dart';
import '../reader/scene_reader.dart';
import 'engine_host.dart';
import 'padnet.dart';
import 'scan_pipeline.dart';

class OfflineEngine implements ScanEngine {
  OfflineEngine._(this._pipeline, this._pp, this._pad, this.status);

  final ScanPipeline _pipeline;
  final PpEngine? _pp; // null when built over a fake recogniser in tests
  final PadScorer? _pad;

  @override
  final EngineStatus status;

  /// Builds the engine inside the worker isolate. Never throws: a model that cannot be loaded gives
  /// an engine whose status is not ready and carries the real reason.
  static ScanEngine create(EngineAssets assets) {
    final sw = Stopwatch()..start();
    PpEngine pp;
    try {
      // det / rec: CPU provider only (dynamic input shapes crash XNNPACK / NNAPI on Android)
      pp = PpEngine.fromBytes(assets.det, assets.rec, assets.charset);
    } catch (e) {
      return _FailedEngine('The text models could not be loaded: $e');
    }
    String? padReason;
    final pad = PadNetScorer.tryLoad(assets.pad, error: (r) => padReason = r);
    try {
      _warmUp(pp); // so the first real scan is not the slow one; a failure here must not disable the engine
    } catch (_) {}
    final pipeline = ScanPipeline(reader: _SceneAdapter(SceneReader(PpRecEngine(pp))), pad: pad, padUnavailableReason: padReason);
    return OfflineEngine._(
      pipeline,
      pp,
      pad,
      EngineStatus(
        ready: true,
        recaptureModel: pad != null,
        detail: 'PP-OCR v3 + grammar decoder, CPU${pad == null ? '' : ' · screen/print model'}',
        loadMs: sw.elapsedMilliseconds,
      ),
    );
  }

  /// Test seam: the whole pipeline over ANY recogniser, so everything above the ONNX layer (scene reader,
  /// decoder, geometry, authenticity) can run under `flutter test`, which has no native ONNX Runtime.
  static OfflineEngine withRecEngine(RecEngine rec, {PadScorer? pad, String? padUnavailableReason}) {
    final pipeline = ScanPipeline(reader: _SceneAdapter(SceneReader(rec)), pad: pad, padUnavailableReason: padUnavailableReason);
    return OfflineEngine._(pipeline, null, pad, EngineStatus(ready: true, recaptureModel: pad != null, detail: 'test recogniser'));
  }

  static void _warmUp(PpEngine pp) {
    final gray = Uint8List(160 * 64 * 3)..fillRange(0, 160 * 64 * 3, 200);
    pp.detect(RgbImage(160, 64, gray));
    final crop = Uint8List(96 * 48 * 3)..fillRange(0, 96 * 48 * 3, 200);
    pp.recProbsMany([RgbImage(96, 48, crop)]);
  }

  @override
  Map<String, dynamic> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) =>
      _pipeline.run(photos, hsrp: hsrp, requireLiveness: requireLiveness);

  @override
  void dispose() {
    _pp?.dispose();
    _pad?.dispose();
  }
}

class _FailedEngine implements ScanEngine {
  _FailedEngine(String error) : status = EngineStatus(ready: false, error: error);

  @override
  final EngineStatus status;

  @override
  Map<String, dynamic> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) =>
      throw StateError(status.error ?? 'engine not ready');

  @override
  void dispose() {}
}

/// Adapts the scene reader's result types to the small ports the pipeline works with.
class _SceneAdapter implements SceneReaderPort {
  _SceneAdapter(this._reader);
  final SceneReader _reader;

  @override
  ReadScene readFrames(List<RgbImage> rgbs, {required Set<String> domains, bool allowLegacyVin = false}) =>
      _scene(_reader.readFrames(rgbs, domains: domains, allowLegacyVin: allowLegacyVin));

  static ReadScene _scene(SceneResult r) => ReadScene(
        plate: _field(r.plate),
        vin: _field(r.vin),
        lines: r.lines,
        conflicts: r.conflicts,
        perImage: [for (final s in r.perImage) _scene(s)],
        timings: r.timings,
        screenContext: r.screenContext,
      );

  static ReadField? _field(FieldReading? f) => f == null
      ? null
      : ReadField(
          domain: f.domain,
          status: f.status,
          rank: f.rank(),
          image: f.image,
          box: f.box,
          text: f.reading.text,
          masked: f.reading.masked,
          json: f.toJson(),
        );
}
