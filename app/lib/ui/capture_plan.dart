import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../data/scanner.dart';

enum ScanMode { full, plate, vin }

enum GuideKind { plate, vin }

/// One photo the person is asked for. [role] is what the reader needs (plate | angle | vin).
/// A step with [withFlash] also takes a second frame with the torch on, automatically, from the
/// same spot (key `<id>_flash`, role `plate_flash`) - the person never has to repeat the pose.
class StepSpec {
  const StepSpec({
    required this.id,
    required this.role,
    required this.label,
    required this.title,
    required this.instruction,
    required this.tips,
    required this.icon,
    this.guide = GuideKind.plate,
    this.optional = false,
    this.withFlash = false,
    this.skipWarning = '',
  });

  final String id, role, label, title, instruction;
  final List<String> tips;
  final IconData icon;
  final GuideKind guide;
  final bool optional;
  final bool withFlash;
  final String skipWarning;

  String get flashKey => '${id}_flash';
}

const _front = StepSpec(
  id: 'front',
  role: 'plate',
  label: 'Front',
  title: 'Front: photograph the number plate',
  instruction: 'Stand square-on to the plate so it fills the frame and hold still. The app takes the photo and, a moment later, '
      'a second one with the torch on - the pair is how it checks that a real, reflective plate is in front of the camera.',
  tips: ['Plate fills ~1/3 of the width', 'No reflection on the characters', 'Hold still until both photos are done'],
  icon: Icons.directions_car,
  withFlash: true,
);

const _left = StepSpec(
  id: 'left',
  role: 'angle',
  label: 'Left',
  title: 'Left: step about 30 cm to your left',
  instruction: 'Move sideways to your left (do not just turn the phone) and photograph the plate again. A second angle moves glare off the characters and lets the app compare the photos.',
  tips: ['Move, do not just rotate', 'Plate still fully in frame', 'Another angle also moves glare off the characters'],
  icon: Icons.keyboard_double_arrow_left,
);

const _right = StepSpec(
  id: 'right',
  role: 'angle',
  label: 'Right',
  title: 'Right: step about 30 cm to your right',
  instruction: 'Now move sideways to your right, past the front position, and photograph the plate once more.',
  tips: ['Move, do not just rotate', 'Plate still fully in frame', 'Keep the same distance as the front photo'],
  icon: Icons.keyboard_double_arrow_right,
);

StepSpec _vinStep({required bool optional}) => StepSpec(
      id: 'vin',
      role: 'vin',
      label: 'VIN',
      title: 'Photograph the VIN',
      instruction: 'Get close to the VIN sticker or stamping, square-on. Tilt the phone slightly so glare slides off the characters.',
      tips: const ['Whole 17-character line in frame', 'Tilt a little to remove reflections', 'Wipe dirt off the stamping if you can'],
      icon: Icons.pin,
      guide: GuideKind.vin,
      optional: optional,
      skipWarning: optional ? 'No VIN photo means no VIN is read for this vehicle.' : '',
    );

/// What a scan asks for. Photos always come from the live camera; there is no way to add one from
/// the gallery, so a saved or edited image cannot be submitted.
class CapturePlan {
  const CapturePlan(this.mode, this.steps);
  final ScanMode mode;
  final List<StepSpec> steps;

  factory CapturePlan.forMode(ScanMode m) => switch (m) {
        ScanMode.full => CapturePlan(ScanMode.full, [_front, _left, _right, _vinStep(optional: true)]),
        ScanMode.plate => const CapturePlan(ScanMode.plate, [_front, _left, _right]),
        ScanMode.vin => CapturePlan(ScanMode.vin, [_vinStep(optional: false)]),
      };

  String get title => switch (mode) { ScanMode.full => 'Full vehicle check', ScanMode.plate => 'Scan number plate', ScanMode.vin => 'Scan VIN' };

  /// The photos to read, in step order, for what was captured. A flash frame follows its step.
  List<CapturedPhoto> photos(Map<String, Uint8List> shots) => [
        for (final s in steps) ...[
          if (shots[s.id] != null) CapturedPhoto(role: s.role, bytes: shots[s.id]!, name: '${s.id}.jpg'),
          if (s.withFlash && shots[s.flashKey] != null) CapturedPhoto(role: 'plate_flash', bytes: shots[s.flashKey]!, name: '${s.flashKey}.jpg'),
        ]
      ];

  /// Steps with no photo yet (the review sheet lists the optional ones that were skipped).
  List<StepSpec> missing(Map<String, Uint8List> shots) => [for (final s in steps) if (shots[s.id] == null) s];

  /// Every required step has its photo. For a plate scan that is front + left + right.
  bool canSubmit(Map<String, Uint8List> shots) => steps.where((s) => !s.optional).every((s) => shots[s.id] != null);

  bool hasFlash(Map<String, Uint8List> shots) => steps.any((s) => s.withFlash && shots[s.flashKey] != null);

  /// Whether this scan can establish that a real car was photographed at all.
  bool get checksLiveness => steps.any((s) => s.withFlash || s.role == 'angle');
}
