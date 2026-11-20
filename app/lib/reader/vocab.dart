/// Recogniser vocabulary: CTC blank + 0-9 + A-Z + an explicit UNREADABLE symbol ('?').
/// Port of `ml/vinplate/vocab.py`.
library;

import 'types.dart';

const int blankId = 0;

/// The real characters (no '?'). The grammar treats '?' as a wildcard over exactly these.
const String alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const String unk = '?';

/// Label id (1-based) of a character. Throws [ArgumentError] outside the alphabet (Python: KeyError).
int charToId(String c) {
  final i = kChars.indexOf(c);
  if (c.length != 1 || i < 0) throw ArgumentError.value(c, 'c', 'not in the recogniser alphabet');
  return i + 1;
}

/// Text -> label ids.
List<int> encodeText(String text) => [for (var i = 0; i < text.length; i++) charToId(text[i])];

/// Label ids -> text (blank ids are skipped).
String decodeLabels(List<int> labels) => labels.where((i) => i != blankId).map((i) => kChars[i - 1]).join();
