import 'dart:async';
import 'dart:ui' show PlatformDispatcher;

import 'package:flutter/material.dart';

import 'app_services.dart';
import 'crash.dart';
import 'data/history.dart';
import 'offline/engine_host.dart';
import 'offline/offline_engine.dart';
import 'data/settings.dart';
import 'ui/home_page.dart';
import 'ui/theme.dart';

Future<void> main() async {
  await runZonedGuarded(() async {
    WidgetsFlutterBinding.ensureInitialized();
    await CrashLog.init();
    FlutterError.onError = (d) {
      CrashLog.record(d.exception, d.stack, kind: 'FLUTTER');
      FlutterError.presentError(d);
    };
    PlatformDispatcher.instance.onError = (e, st) {
      CrashLog.record(e, st);
      return true;
    };
    final settings = AppSettings();
    await settings.load();
    runApp(App(services: AppServices(
      settings: settings,
      history: HistoryStore(),
      scanner: OnDeviceScanner(runner: IsolateRunner(OfflineEngine.create)),
    )));
  }, (e, st) => CrashLog.record(e, st));
}

class App extends StatelessWidget {
  const App({super.key, required this.services});
  final AppServices services;

  @override
  Widget build(BuildContext context) => AppScope(
        services: services,
        child: MaterialApp(
          title: 'VinPlate Scanner',
          debugShowCheckedModeBanner: false,
          theme: buildTheme(Brightness.light),
          darkTheme: buildTheme(Brightness.dark),
          themeMode: ThemeMode.system,
          home: const HomePage(),
        ),
      );
}
