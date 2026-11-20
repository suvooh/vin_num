import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../data/models.dart';
import '../data/scanner.dart';

/// Model files, as plain bytes, so they can cross an isolate boundary (rootBundle only exists in
/// the main isolate; the engine runs in a worker).
class EngineAssets {
  EngineAssets({required this.det, required this.rec, required this.charset, this.pad});
  final Uint8List det, rec;
  final String charset;
  final Uint8List? pad; // screen / print detector; absent = that check is reported as not run

  static Future<EngineAssets> fromBundle() async {
    Uint8List b(ByteData d) => d.buffer.asUint8List(d.offsetInBytes, d.lengthInBytes);
    Uint8List? pad;
    try {
      pad = b(await rootBundle.load('assets/models/pad_int8.onnx'));
    } catch (_) {}
    return EngineAssets(
      det: b(await rootBundle.load('assets/models/ocr_det.onnx')),
      rec: b(await rootBundle.load('assets/models/ocr_rec.onnx')),
      charset: await rootBundle.loadString('assets/models/ocr_charset.txt'),
      pad: pad,
    );
  }
}

/// The pipeline itself: lives in the worker isolate, owns the ONNX sessions.
abstract interface class ScanEngine {
  EngineStatus get status;

  /// Returns the response map (same shape `ScanResult.fromJson` reads).
  Map<String, dynamic> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness});

  void dispose();
}

class EnginePhoto {
  const EnginePhoto(this.role, this.name, this.bytes);
  final String role, name;
  final Uint8List bytes;
}

typedef EngineFactory = ScanEngine Function(EngineAssets assets);

/// Where the engine runs. [IsolateRunner] in the app; [InlineRunner] in tests.
abstract interface class EngineRunner {
  Future<EngineStatus> start(EngineAssets assets);
  Future<Map<String, dynamic>> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness});

  /// Stops any scan in progress. The next [start]/[scan] begins from a clean state.
  void cancel();
  void dispose();
}

class InlineRunner implements EngineRunner {
  InlineRunner(this.factory);
  final EngineFactory factory;
  ScanEngine? _e;

  @override
  Future<EngineStatus> start(EngineAssets assets) async => (_e ??= factory(assets)).status;

  @override
  Future<Map<String, dynamic>> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) async =>
      _e!.scan(photos, hsrp: hsrp, requireLiveness: requireLiveness);

  @override
  void cancel() {}

  @override
  void dispose() => _e?.dispose();
}

// ===================================================================== isolate runner
/// Keeps the models loaded in one long-lived worker isolate, so only the first scan pays the load
/// cost. Cancelling kills the worker (a running native inference cannot be interrupted any other
/// way); the next scan respawns it and reloads the models.
class IsolateRunner implements EngineRunner {
  IsolateRunner(this.factory);
  final EngineFactory factory;

  Isolate? _iso;
  SendPort? _tx;
  ReceivePort? _rx;
  EngineAssets? _assets;
  int _next = 0;
  final Map<int, Completer<Map<String, dynamic>>> _pending = {};
  EngineStatus? _status;
  Future<EngineStatus>? _starting;

  @override
  Future<EngineStatus> start(EngineAssets assets) {
    _assets = assets;
    if (_status != null && _tx != null) return Future.value(_status);
    return _starting ??= _spawn(assets).whenComplete(() => _starting = null);
  }

  Future<EngineStatus> _spawn(EngineAssets a) async {
    final rx = ReceivePort();
    final ready = Completer<SendPort>();
    rx.listen((m) {
      if (m is SendPort) {
        ready.complete(m);
      } else if (m is Map) {
        final c = _pending.remove(m['id'] as int);
        if (c == null) return;
        if (m['error'] != null) {
          c.completeError(m['bad'] == true
              ? ScanException(ScanErrorKind.badImage, "One of the photos couldn't be opened. Retake it.", detail: m['error'] as String)
              : ScanException(ScanErrorKind.failed, 'Something went wrong while reading.', detail: m['error'] as String));
        } else {
          c.complete((m['result'] as Map).cast<String, dynamic>());
        }
      }
    });
    _rx = rx;
    _iso = await Isolate.spawn(_workerMain, [rx.sendPort, factory]);
    _tx = await ready.future;
    final sw = Stopwatch()..start();
    final r = await _call({
      'cmd': 'init',
      'det': TransferableTypedData.fromList([a.det]),
      'rec': TransferableTypedData.fromList([a.rec]),
      'pad': a.pad == null ? null : TransferableTypedData.fromList([a.pad!]),
      'charset': a.charset,
    });
    return _status = EngineStatus(
      ready: r['ready'] == true,
      recaptureModel: r['recapture_model'] == true,
      detail: r['detail'] as String?,
      error: r['error'] as String?,
      loadMs: sw.elapsedMilliseconds,
    );
  }

  Future<Map<String, dynamic>> _call(Map<String, dynamic> msg) {
    final id = _next++;
    final c = Completer<Map<String, dynamic>>();
    _pending[id] = c;
    _tx!.send({...msg, 'id': id});
    return c.future;
  }

  @override
  Future<Map<String, dynamic>> scan(List<EnginePhoto> photos, {required bool hsrp, required bool requireLiveness}) async {
    if (_tx == null) await start(_assets!);
    return _call({
      'cmd': 'scan',
      'hsrp': hsrp,
      'live': requireLiveness,
      'photos': [for (final p in photos) {'role': p.role, 'name': p.name, 'bytes': TransferableTypedData.fromList([p.bytes])}],
    });
  }

  @override
  void cancel() {
    final err = const ScanException(ScanErrorKind.cancelled, 'Cancelled.');
    for (final c in _pending.values) {
      if (!c.isCompleted) c.completeError(err);
    }
    _pending.clear();
    _kill();
  }

  void _kill() {
    _iso?.kill(priority: Isolate.immediate);
    _rx?.close();
    _iso = null;
    _tx = null;
    _rx = null;
    _status = null;
  }

  @override
  void dispose() => _kill();
}

/// Worker isolate entry: [args] = [SendPort, EngineFactory]. Top-level so it can be spawned.
void _workerMain(List<Object> args) {
  final out = args[0] as SendPort;
  final factory = args[1] as EngineFactory;
  final inbox = ReceivePort();
  out.send(inbox.sendPort);
  ScanEngine? engine;
  inbox.listen((m) {
    final msg = (m as Map).cast<String, dynamic>();
    final id = msg['id'] as int;
    try {
      switch (msg['cmd']) {
        case 'init':
          Uint8List bytes(Object t) => (t as TransferableTypedData).materialize().asUint8List();
          engine?.dispose();
          engine = factory(EngineAssets(
              det: bytes(msg['det']!), rec: bytes(msg['rec']!), charset: msg['charset'] as String, pad: msg['pad'] == null ? null : bytes(msg['pad']!)));
          final s = engine!.status;
          out.send({'id': id, 'result': {'ready': s.ready, 'recapture_model': s.recaptureModel, 'detail': s.detail, 'error': s.error}});
        case 'scan':
          final photos = [
            for (final p in (msg['photos'] as List).cast<Map>())
              EnginePhoto(p['role'] as String, p['name'] as String, (p['bytes'] as TransferableTypedData).materialize().asUint8List())
          ];
          final r = engine!.scan(photos, hsrp: msg['hsrp'] as bool, requireLiveness: msg['live'] as bool);
          out.send({'id': id, 'result': r});
      }
    } catch (e, st) {
      out.send({'id': id, 'error': '$e\n$st', 'bad': e is FormatException});
    }
  });
}

// ===================================================================== the scanner
/// The app's [Scanner]: an [EngineRunner] plus the bookkeeping the UI needs.
class OnDeviceScanner implements Scanner {
  OnDeviceScanner({required this.runner, Future<EngineAssets> Function()? assets}) : _assets = assets ?? EngineAssets.fromBundle;
  final EngineRunner runner;
  final Future<EngineAssets> Function() _assets;
  EngineStatus? _status;
  Future<EngineStatus>? _loading;

  @override
  Future<EngineStatus> prepare() {
    final s = _status;
    if (s != null && s.ready) return Future.value(s);
    return _loading ??= () async {
      try {
        return _status = await runner.start(await _assets());
      } catch (e) {
        return _status = EngineStatus(ready: false, error: 'The reading engine could not start: $e');
      } finally {
        _loading = null;
      }
    }();
  }

  @override
  Future<ScanResult> scan(List<CapturedPhoto> photos, {required bool hsrp, required bool requireLiveness}) async {
    final st = await prepare();
    if (!st.ready) throw ScanException(ScanErrorKind.engine, 'The reading engine is not available.', detail: st.error);
    final sw = Stopwatch()..start();
    try {
      final map = await runner.scan([for (final p in photos) EnginePhoto(p.role, p.name, p.bytes)], hsrp: hsrp, requireLiveness: requireLiveness);
      map['capture_id'] ??= DateTime.now().microsecondsSinceEpoch.toRadixString(36);
      map['processed_at'] ??= DateTime.now().toIso8601String().split('.').first;
      final r = ScanResult.fromJson(map);
      return r;
    } on ScanException catch (e) {
      if (e.kind == ScanErrorKind.cancelled) _status = null; // the worker was killed: reload on next use
      rethrow;
    } on FormatException catch (e) {
      throw ScanException(ScanErrorKind.failed, 'The reader returned something unexpected.', detail: e.message);
    } finally {
      sw.stop();
    }
  }

  @override
  void cancel() => runner.cancel();

  @override
  void dispose() => runner.dispose();
}
