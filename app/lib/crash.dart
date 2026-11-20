import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Crash capture. Three sources, shown together in a copyable dialog on the next launch:
///  1. Dart errors (unhandled, Flutter framework, or handled-but-reported) -> crash.log
///  2. an "in progress" marker written before each risky native step; if it is still there at
///     launch, the previous run died inside that step (a native crash kills the process before
///     any Dart handler can run)
///  3. Android's own record of why the last process exited (crash / native crash / killed for
///     memory / ANR), via ApplicationExitInfo (Android 11+)
class CrashLog {
  static const _sys = MethodChannel('vinplate/sys');
  static late File _log, _marker;
  static var _ready = false;

  static Future<void> init() async {
    final d = await getApplicationSupportDirectory();
    _log = File('${d.path}/crash.log');
    _marker = File('${d.path}/in_progress.txt');
    _ready = true;
  }

  /// Call before a step that can kill the process; [done] after it returns.
  static void start(String stage, [String detail = '']) {
    if (!_ready) return;
    try {
      _marker.writeAsStringSync('${DateTime.now().toIso8601String()}\n$stage\n$detail', flush: true);
    } catch (_) {}
  }

  static void done() {
    if (!_ready) return;
    try {
      if (_marker.existsSync()) _marker.deleteSync();
    } catch (_) {}
  }

  static void record(Object error, StackTrace? st, {String kind = 'UNHANDLED'}) {
    if (!_ready) return;
    try {
      _log.writeAsStringSync('--- ${DateTime.now().toIso8601String()}  $kind\n$error\n${st ?? ''}\n\n', mode: FileMode.append, flush: true);
    } catch (_) {}
  }

  /// null when the previous run ended cleanly.
  static Future<String?> pendingReport() async {
    if (!_ready) return null;
    final parts = <String>[];
    try {
      if (_marker.existsSync()) {
        parts.add('PREVIOUS RUN DID NOT FINISH. It was in this step when it stopped:\n${_marker.readAsStringSync().trim()}');
      }
      if (_log.existsSync() && _log.lengthSync() > 0) {
        final t = _log.readAsStringSync();
        parts.add('ERRORS RECORDED:\n${t.length > 6000 ? t.substring(t.length - 6000) : t}');
      }
      if (parts.isNotEmpty) {
        try {
          final exits = (await _sys.invokeMethod<List<dynamic>>('lastExits'))?.cast<String>() ?? const <String>[];
          if (exits.isNotEmpty) parts.add('ANDROID SAYS THE LAST PROCESSES EXITED AS:\n${exits.join('\n')}');
        } catch (e) {
          parts.add('(could not read Android exit info: $e)');
        }
        parts.add('DEVICE: ${Platform.operatingSystemVersion}');
      }
    } catch (e) {
      parts.add('(crash report itself failed: $e)');
    }
    return parts.isEmpty ? null : parts.join('\n\n');
  }

  static Future<void> clear() async {
    try {
      if (_marker.existsSync()) _marker.deleteSync();
      if (_log.existsSync()) _log.deleteSync();
    } catch (_) {}
  }
}
