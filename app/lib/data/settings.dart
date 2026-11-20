import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// User settings, persisted as one small JSON file. Reads never throw: a missing or corrupt file
/// simply yields the defaults.
class AppSettings extends ChangeNotifier {
  AppSettings({File? file, this.persist = true}) : _fileOverride = file;

  /// false: keep everything in memory (tests; real file I/O never completes under fake async).
  final bool persist;

  final File? _fileOverride;
  bool hsrp = true;
  bool requireLiveness = false;
  bool saveHistory = true;
  bool onboarded = false;
  bool cameraRationaleShown = false;

  Future<File> _file() async => _fileOverride ?? File('${(await getApplicationSupportDirectory()).path}/settings.json');

  Future<void> load() async {
    if (!persist) return;
    try {
      final f = await _file();
      if (!await f.exists()) return;
      final j = (jsonDecode(await f.readAsString()) as Map).cast<String, dynamic>();
      hsrp = j['hsrp'] as bool? ?? hsrp;
      requireLiveness = j['requireLiveness'] as bool? ?? requireLiveness;
      saveHistory = j['saveHistory'] as bool? ?? saveHistory;
      onboarded = j['onboarded'] as bool? ?? onboarded;
      cameraRationaleShown = j['cameraRationaleShown'] as bool? ?? cameraRationaleShown;
      notifyListeners();
    } catch (_) {
      // corrupt settings: keep defaults
    }
  }

  Future<void> update({
    bool? hsrp,
    bool? requireLiveness,
    bool? saveHistory,
    bool? onboarded,
    bool? cameraRationaleShown,
  }) async {
    if (hsrp != null) this.hsrp = hsrp;
    if (requireLiveness != null) this.requireLiveness = requireLiveness;
    if (saveHistory != null) this.saveHistory = saveHistory;
    if (onboarded != null) this.onboarded = onboarded;
    if (cameraRationaleShown != null) this.cameraRationaleShown = cameraRationaleShown;
    notifyListeners();
    if (!persist) return;
    try {
      await (await _file()).writeAsString(jsonEncode({
        'hsrp': this.hsrp,
        'requireLiveness': this.requireLiveness,
        'saveHistory': this.saveHistory,
        'onboarded': this.onboarded,
        'cameraRationaleShown': this.cameraRationaleShown,
      }));
    } catch (_) {
      // a read-only disk must not break the app; the in-memory value is still used
    }
  }
}
