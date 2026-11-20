import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

import 'models.dart';

/// One finished scan as kept on this phone: the server response, the value the person finally
/// confirmed, and the photos (so the result can be shown again with its read region).
class HistoryEntry {
  HistoryEntry({required this.id, required this.savedAt, required this.result, this.confirmedPlate, this.confirmedVin, this.confirmedCategory, this.dir});
  final String id;
  final DateTime savedAt;
  final ScanResult result;
  final String? confirmedPlate, confirmedVin;

  /// The vehicle type (a PlateCategory key) the person chose, when they overrode what the reader worked out.
  final String? confirmedCategory;
  final Directory? dir;

  /// The type to show for this scan: the person's choice first, else what the reader worked out.
  String? get categoryKey => confirmedCategory ?? result.plate?.category?.key;

  String get title => confirmedPlate ?? confirmedVin ?? result.plate?.shown ?? result.vin?.shown ?? 'Nothing read';
  File? photoFile(int i) => dir == null ? null : File('${dir!.path}/photo$i.jpg');
}

class HistoryStore {
  HistoryStore({Directory? root}) : _rootOverride = root;
  static const maxEntries = 30;
  final Directory? _rootOverride;

  Future<Directory> _root() async {
    final d = _rootOverride ?? Directory('${(await getApplicationSupportDirectory()).path}/history');
    if (!await d.exists()) await d.create(recursive: true);
    return d;
  }

  Future<HistoryEntry> add(ScanResult r, List<Uint8List> photos, {String? confirmedPlate, String? confirmedVin, String? confirmedCategory}) async {
    final root = await _root();
    final id = DateTime.now().toIso8601String().replaceAll(RegExp(r'[:.]'), '-');
    final dir = Directory('${root.path}/$id')..createSync(recursive: true);
    for (var i = 0; i < photos.length; i++) {
      await File('${dir.path}/photo$i.jpg').writeAsBytes(photos[i]);
    }
    await File('${dir.path}/entry.json').writeAsString(jsonEncode({
      'savedAt': DateTime.now().toIso8601String(),
      'confirmedPlate': confirmedPlate,
      'confirmedVin': confirmedVin,
      'confirmedCategory': confirmedCategory,
      'result': r.raw,
    }));
    await _trim(root);
    return HistoryEntry(
        id: id, savedAt: DateTime.now(), result: r, confirmedPlate: confirmedPlate, confirmedVin: confirmedVin, confirmedCategory: confirmedCategory, dir: dir);
  }

  Future<List<HistoryEntry>> list() async {
    final root = await _root();
    final out = <HistoryEntry>[];
    for (final e in root.listSync().whereType<Directory>()) {
      try {
        final j = (jsonDecode(File('${e.path}/entry.json').readAsStringSync()) as Map).cast<String, dynamic>();
        out.add(HistoryEntry(
          id: e.uri.pathSegments.where((s) => s.isNotEmpty).last,
          savedAt: DateTime.parse(j['savedAt'] as String),
          result: ScanResult.fromJson((j['result'] as Map).cast<String, dynamic>()),
          confirmedPlate: j['confirmedPlate'] as String?,
          confirmedVin: j['confirmedVin'] as String?,
          confirmedCategory: j['confirmedCategory'] as String?,
          dir: e,
        ));
      } catch (_) {
        // an unreadable entry is skipped, never fatal
      }
    }
    out.sort((a, b) => b.savedAt.compareTo(a.savedAt));
    return out;
  }

  Future<void> delete(HistoryEntry e) async {
    if (e.dir != null && await e.dir!.exists()) await e.dir!.delete(recursive: true);
  }

  Future<void> clear() async {
    final root = await _root();
    for (final d in root.listSync()) {
      await d.delete(recursive: true);
    }
  }

  Future<void> _trim(Directory root) async {
    final dirs = root.listSync().whereType<Directory>().toList()..sort((a, b) => b.path.compareTo(a.path));
    for (final d in dirs.skip(maxEntries)) {
      await d.delete(recursive: true);
    }
  }
}
