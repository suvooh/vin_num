import 'dart:convert';
import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import 'checks.dart' show Box;
import 'engine/fields.dart';
import 'engine/ocr.dart';
import 'engine/ort.dart';
import 'engine/types.dart';

class IdResult {
  final List<OcrLine> lines;
  final VinRead? vin;
  final PlateRead? plate;
  final int ms;
  /// Where the read text sits in the (upright) photo, from the OCR detector's own boxes. null =
  /// not located: callers must say so, never fall back silently to a guess.
  final Box? plateBox, vinBox;
  IdResult(this.lines, this.vin, this.plate, this.ms, {this.plateBox, this.vinBox});
}

String _norm(String t) => t.toUpperCase().replaceAll(RegExp(r'[^A-Z0-9]'), '');

/// Union of the OCR lines whose text makes up [value] (a line counts if its alphanumerics are a
/// >=3 character piece of the value, or contain all of it).
Box? _regionOf(List<OcrLine> lines, String? value) {
  if (value == null) return null;
  final v = _norm(value);
  double? x0, y0, x1, y1;
  for (final l in lines) {
    final t = _norm(l.text);
    if (t.length >= 3 && (v.contains(t) || t.contains(v))) {
      x0 = x0 == null ? l.x : (l.x < x0 ? l.x : x0);
      y0 = y0 == null ? l.y0 : (l.y0 < y0 ? l.y0 : y0);
      x1 = x1 == null ? l.x1 : (l.x1 > x1 ? l.x1 : x1);
      final b = l.y0 + l.h;
      y1 = y1 == null ? b : (b > y1 ? b : y1);
    }
  }
  return x0 == null ? null : Box(x0.floor(), y0!.floor(), x1!.ceil(), y1!.ceil());
}

/// Whole-photo OCR with the dealer platform's PP-OCR detector + recogniser (PaddleOCR v3
/// via the Dart port in engine/), then the VIN / Indian-plate extractors in engine/fields.dart.
/// Runs in a background isolate. Plain CPU provider ONLY: these models have dynamic input
/// shapes and XNNPACK/NNAPI crash natively on them on Android (the dealer app documents this).
Future<IdResult> readIdentifiers(Uint8List jpeg) async {
  final det = (await rootBundle.load('assets/models/ocr_det.onnx')).buffer.asUint8List();
  final rec = (await rootBundle.load('assets/models/ocr_rec.onnx')).buffer.asUint8List();
  final charset = await rootBundle.loadString('assets/models/ocr_charset.txt');
  return Isolate.run(() => _work(jpeg, det, rec, charset));
}

IdResult _work(Uint8List jpeg, Uint8List detB, Uint8List recB, String charset) {
  final sw = Stopwatch()..start();
  var im = img.decodeImage(jpeg);
  if (im == null) throw const FormatException('could not decode image');
  im = img.bakeOrientation(im);
  final rgb = RgbImage(im.width, im.height, im.convert(numChannels: 3).getBytes(order: img.ChannelOrder.rgb));
  final det = Model.load('det', detB, accelerator: Accelerator.cpu);
  final rec = Model.load('rec', recB, accelerator: Accelerator.cpu);
  try {
    final lines = Ocr(det, rec, const LineSplitter().convert(charset)).read(rgb);
    final vin = findVin(lines);
    final plate = findPlate(lines, vin: vin?.value);
    return IdResult(lines, vin, plate, sw.elapsedMilliseconds,
        plateBox: _regionOf(lines, plate?.value), vinBox: _regionOf(lines, vin?.value));
  } finally {
    det.release();
    rec.release();
  }
}
