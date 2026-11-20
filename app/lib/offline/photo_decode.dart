// JPEG / PNG bytes -> an upright RGB image for the reader. Port of backend/app/imaging.py `decode`:
// EXIF orientation applied, longest side capped (the Python MAX_SIDE), EXIF capture time returned.
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../engine/types.dart';

/// The longest side the reader works at. Larger photos are shrunk (area average) before anything else.
const maxSide = 1600;

class DecodedPhoto {
  const DecodedPhoto(this.rgb, {this.takenAt});
  final RgbImage rgb;

  /// EXIF DateTimeOriginal as an ISO-8601 string, or null when the photo has none.
  final String? takenAt;
}

typedef DecodeFn = DecodedPhoto Function(Uint8List bytes);

/// Throws [FormatException] when the bytes are not an image (the worker reports that as a bad photo).
DecodedPhoto decodePhoto(Uint8List bytes, {int maxLongSide = maxSide}) {
  img.Image? im;
  try {
    im = img.decodeImage(bytes);
  } catch (e) {
    throw FormatException('could not decode image: $e');
  }
  if (im == null) throw const FormatException('could not decode image');
  final takenAt = _takenAt(im);
  final longest = im.width > im.height ? im.width : im.height;
  if (longest > maxLongSide) {
    // shrink BEFORE rotating: rotating a 12 MP image just to throw most of it away is the slow part
    final s = maxLongSide / longest;
    final small = img.copyResize(im,
        width: (im.width * s).round().clamp(1, maxLongSide), height: (im.height * s).round().clamp(1, maxLongSide), interpolation: img.Interpolation.average);
    small.exif = img.ExifData.from(im.exif);
    im = small;
  }
  im = img.bakeOrientation(im);
  final rgb = im.convert(numChannels: 3).getBytes(order: img.ChannelOrder.rgb);
  return DecodedPhoto(RgbImage(im.width, im.height, rgb), takenAt: takenAt);
}

String? _takenAt(img.Image im) {
  try {
    final raw = (im.exif.exifIfd['DateTimeOriginal'] ?? im.exif.imageIfd['DateTime'])?.toString().trim();
    if (raw == null) return null;
    final m = RegExp(r'^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})').firstMatch(raw);
    return m == null ? null : '${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}';
  } catch (_) {
    return null;
  }
}
