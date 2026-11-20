import 'package:flutter/widgets.dart';

import 'data/history.dart';
import 'data/scanner.dart';
import 'data/settings.dart';

/// What every screen needs, injected once at the top so tests can substitute fakes. The scanner is
/// ONE long-lived object: it keeps the models loaded between scans.
class AppServices {
  AppServices({required this.settings, required this.history, required this.scanner});

  final AppSettings settings;
  final HistoryStore history;
  final Scanner scanner;
}

class AppScope extends InheritedWidget {
  const AppScope({super.key, required this.services, required super.child});
  final AppServices services;

  static AppServices of(BuildContext context) {
    final s = context.dependOnInheritedWidgetOfExactType<AppScope>();
    assert(s != null, 'AppScope missing above this widget');
    return s!.services;
  }

  @override
  bool updateShouldNotify(AppScope old) => services != old.services;
}
