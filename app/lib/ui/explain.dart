/// Plain-language wording for what the backend reports. The backend's strings are precise but
/// technical; the app shows these instead and falls back to the original text for anything new.
library;

String humanReason(String raw) {
  final r = raw.toLowerCase();
  // most specific first: the I/O/Q reason also contains the words "look-alike"
  if (r.contains('i/o/q')) return 'A series letter I, O or Q is not used on modern plates, so it was most likely a different letter. It is shown as ?.';
  if (r.contains('look-alike')) return 'Some characters look like others (for example O, Q and 0) and the photo cannot tell which. They are shown as ?.';
  if (r.contains('no series letters') || r.contains('short number')) return 'The plate has no series letters or a short number. That is either a very old plate, or a character was missed.';
  if (r.contains('dropped character')) return 'A character may be missing from the read.';
  if (r.contains('ambiguous: top reading')) return 'More than one reading is almost equally likely.';
  if (r.contains('legacy chassis')) return 'This looks like a short or older chassis number, whose structure cannot be checked automatically.';
  if (r.contains('unreadable and not recoverable')) return 'Some characters could not be read from the photo.';
  if (r.contains('weak pixel evidence')) return 'Several characters are faint in the photo.';
  if (r.contains('grammar bent')) return 'The read had to be adjusted a lot to fit a valid plate or VIN, so it is mostly a guess.';
  if (r.contains('changed the network')) return ''; // informational only
  if (r.contains('no positive evidence')) return 'No red flags were found, but nothing proves the photo shows a real car (no flash or side-step photo).';
  if (r.contains('depth alone')) return 'The side-step photos show depth, but depth alone is not proof: a screen held in front of a real background also has depth.';
  if (r.contains('liveness required')) return 'You asked for liveness to be required, and it was not confirmed.';
  if (r.contains('presentation attack suspected')) return 'The photos look like a picture of a screen or a print.';
  if (r.contains('possible presentation attack')) return 'Some signs suggest the photo may be of a screen or a print.';
  if (r.contains('did not retroreflect')) return "The plate did not light up under the flash the way reflective sheeting does.";
  if (r.contains('plate text differs')) return 'The plate reads differently in different photos.';
  if (r.contains('pixel forensics') || r.contains('pixel-forensics')) return 'Low-level image statistics look unusual. This is a weak signal.';
  if (r.contains('recapture model not run')) return 'The screen/print detector could not run on this photo.';
  return raw;
}

List<String> humanReasons(Iterable<String> raw) => [for (final r in raw) humanReason(r)].where((s) => s.isNotEmpty).toSet().toList();

String humanCheck(String key) => switch (key) {
      'retroreflection' => 'Flash test (reflective plate)',
      'planarity' => 'Side-step test (depth)',
      'recapture' => 'Screen / print detector',
      'forensics' => 'Pixel forensics (not available on this phone)',
      'tamper' => 'Text-edit detector (not available)',
      'ai_generated' => 'AI-image detector (not available)',
      _ => key,
    };

String humanPositive(String key) => switch (key) {
      'retroreflection_passed' => 'The plate brightened under the flash like reflective sheeting does.',
      'parallax_non_planar' => 'Photos from two positions show the scene has depth.',
      _ => key.replaceAll('_', ' '),
    };

/// What the person can do to get a stronger answer, from what was not checked.
List<String> improvements(Iterable<String> notRun) {
  final s = notRun.toSet();
  return [
    if (s.contains('retroreflection')) 'Take the flash photo from exactly the same spot (shade or dusk works best; direct sun cannot be judged).',
    if (s.contains('planarity')) 'Take one more photo after stepping about 30 cm sideways.',
    if (s.contains('recapture')) 'Move closer so the plate fills more of the frame (the screen/print detector needs a sharp patch).',
  ];
}

/// Plate: 6-11 letters/digits. VIN: 17 characters without I, O, Q, or a shorter legacy chassis number.
String? validateIdentifier(String raw, {required bool plate}) {
  final v = raw.trim().toUpperCase();
  if (v.isEmpty) return 'Enter the value';
  if (!RegExp(r'^[A-Z0-9]+$').hasMatch(v)) return 'Letters and digits only';
  if (plate) return (v.length < 6 || v.length > 11) ? 'A plate has 6 to 11 characters' : null;
  if (v.length < 6 || v.length > 17) return 'A VIN has 17 characters (older chassis numbers: 6 to 17)';
  if (v.length == 17 && RegExp(r'[IOQ]').hasMatch(v)) return 'A 17-character VIN never contains I, O or Q';
  return null;
}
