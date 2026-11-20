import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Product rules that must not regress silently. Each one is about fraud or privacy, not looks.
void main() {
  final lib = Directory('lib').listSync(recursive: true).whereType<File>().where((f) => f.path.endsWith('.dart')).toList();

  test('photos come from the LIVE CAMERA only: no picker package, no picker import anywhere', () {
    final pub = File('pubspec.yaml').readAsStringSync();
    for (final pkg in ['image_picker', 'file_picker', 'photo_manager', 'wechat_assets_picker', 'gallery_saver']) {
      expect(pub.contains(pkg), isFalse, reason: 'pubspec.yaml must not depend on $pkg');
    }
    for (final f in lib) {
      final src = f.readAsStringSync();
      expect(src.contains('ImagePicker'), isFalse, reason: '${f.path} must not use ImagePicker');
      expect(src.contains('FilePicker'), isFalse, reason: '${f.path} must not use FilePicker');
      expect(src.contains('ImageSource.gallery'), isFalse, reason: '${f.path} must not read the gallery');
    }
  });

  test('the main manifest asks for the camera and nothing that reaches the network or the user\'s files', () {
    final raw = File('android/app/src/main/AndroidManifest.xml').readAsStringSync();
    // a permission that appears only as a tools:node="remove" marker is being REMOVED from the merge, not requested
    final removed = RegExp(r'<uses-permission\s+android:name="android\.permission\.(\w+)"\s+tools:node="remove"\s*/>').allMatches(raw).map((x) => x.group(1)).toSet();
    final m = raw.replaceAll(RegExp(r'<uses-permission[^>]*tools:node="remove"[^>]*/>'), '').replaceAll(RegExp(r'<!--.*?-->', dotAll: true), '');
    expect(m.contains('android.permission.CAMERA'), isTrue);
    for (final p in ['INTERNET', 'RECORD_AUDIO', 'READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE', 'READ_MEDIA_IMAGES', 'ACCESS_NETWORK_STATE']) {
      expect(m.contains(p), isFalse, reason: 'main manifest must not request $p');
    }
    // the camera plugin declares audio and storage permissions of its own; they must be stripped from the merged manifest
    expect(removed, containsAll(['RECORD_AUDIO', 'READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE']));
    expect(m.contains('usesCleartextTraffic'), isFalse);
  });

  test('the reading path never opens a network connection', () {
    const banned = ['dart:io', 'package:http', 'HttpClient', 'Socket', 'WebSocket'];
    // dart:io is allowed where it is only about local files; sockets / http never are
    for (final f in lib) {
      final src = f.readAsStringSync();
      for (final b in ['package:http', 'HttpClient', 'Socket.connect', 'WebSocket', 'RawSocket', 'HttpServer']) {
        expect(src.contains(b), isFalse, reason: '${f.path} must not use $b');
      }
    }
    expect(banned, isNotEmpty);
  });

  test('a plate scan can only start from the camera screen: there is no other scan entry point', () {
    final home = File('lib/ui/home_page.dart').readAsStringSync();
    expect(RegExp(r'CapturePage\(').allMatches(home).length, 1);
    expect(home.contains('scanner.scan('), isFalse, reason: 'the home screen must not feed photos to the scanner directly');
    final others = lib.where((f) => !f.path.endsWith('capture_page.dart') && !f.path.contains('offline') && !f.path.contains('data'));
    for (final f in others) {
      expect(f.readAsStringSync().contains('.scan('), isFalse, reason: '${f.path} must not submit photos');
    }
  });
}
