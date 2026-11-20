"""Shared recogniser vocabulary: CTC blank + 0-9 + A-Z + an explicit UNREADABLE symbol.

One alphabet serves both plates and VINs. Two deliberate choices:

* I, O and Q are kept. A VIN may not contain them (ISO 3779) but a glared "0" is often
  *seen* as an O; the grammar layer resolves that from the posterior instead of the network
  pretending it knows.
* ``?`` is a real class: "this character is not legible". Training labels a character ``?``
  when glare / dirt / occlusion hides it (see ``synth``), so the network learns to ABSTAIN
  instead of hallucinating a confident digit. The decoder treats ``?`` as a wildcard that the
  grammar (and checksum, where mandated) may narrow - and always reports as inferred.
"""

from __future__ import annotations

BLANK = 0
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNK = "?"
CHARS = ALPHABET + UNK              # class i (1-based) <-> CHARS[i-1]
VOCAB_SIZE = len(CHARS) + 1         # +1 for the CTC blank  -> 38
_C2I = {c: i + 1 for i, c in enumerate(CHARS)}


def encode_text(text: str) -> list[int]:
    """Text -> label ids. Raises KeyError on characters outside the alphabet."""
    return [_C2I[c] for c in text]


def decode_labels(labels) -> str:
    return "".join(CHARS[i - 1] for i in labels if i != BLANK)


def char_to_id(c: str) -> int:
    return _C2I[c]
