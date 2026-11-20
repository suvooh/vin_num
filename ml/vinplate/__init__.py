"""India-specific vehicle plate + VIN/chassis reading, authenticity and capture-quality layer.

Inference-side code (this package) depends only on numpy / Pillow / onnxruntime so it can
be ported to the mobile runtime. Torch is only needed under ``training/``.
"""

from vinplate.vocab import ALPHABET, BLANK, CHARS, UNK, VOCAB_SIZE, decode_labels, encode_text

__all__ = ["ALPHABET", "BLANK", "CHARS", "UNK", "VOCAB_SIZE", "decode_labels", "encode_text"]
__version__ = "0.1.0"
