import 'dart:typed_data';

import 'models.dart';

/// One photo of a capture bundle. [role] says what the photo is for, so the right checks run:
/// plate | plate_flash | angle | vin | auto.
class CapturedPhoto {
  const CapturedPhoto({required this.role, required this.bytes, required this.name});
  final String role;
  final Uint8List bytes;
  final String name;
}

enum ScanErrorKind { engine, badImage, cancelled, failed }

/// A scan that could not complete. [message] is for the person; [detail] is technical.
class ScanException implements Exception {
  const ScanException(this.kind, this.message, {this.detail});
  final ScanErrorKind kind;
  final String message;
  final String? detail;

  @override
  String toString() => message;
}

/// State of the reading engine that lives inside this app (models + runtime). There is no server.
class EngineStatus {
  const EngineStatus({required this.ready, this.recaptureModel = false, this.detail, this.error, this.loadMs});
  final bool ready;
  final bool recaptureModel; // the screen / print detector loaded
  final String? detail; // e.g. "PP-OCR v3 + grammar decoder, CPU"
  final String? error;
  final int? loadMs;
}

/// Reads plates and VINs from photos and judges whether a real car was photographed - completely
/// on the device. Implementations must not use the network.
abstract class Scanner {
  /// Loads the models if they are not loaded yet. Safe to call repeatedly.
  Future<EngineStatus> prepare();

  Future<ScanResult> scan(List<CapturedPhoto> photos, {required bool hsrp, required bool requireLiveness});

  /// Aborts an in-flight [scan]; it then throws [ScanErrorKind.cancelled].
  void cancel();

  void dispose();
}
