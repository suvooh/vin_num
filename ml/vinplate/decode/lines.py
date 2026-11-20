"""Joining the posteriors of several text lines into one sequence for grammar decoding.

A 2-row plate is read row by row (each row is an ordinary text line for the recogniser);
the grammar then needs ONE string ("MH12" + "AB1234" -> "MH12AB1234"), so the per-row
posteriors are concatenated. Blank margins are trimmed first, otherwise the junction looks
like a wide gap and would trip the dropped-character detector.
"""

from __future__ import annotations

import numpy as np

from vinplate.vocab import BLANK


def trim_blank(logp: np.ndarray, keep: int = 1, blank_p: float = 0.9) -> np.ndarray:
    """Drop leading/trailing frames that are confidently blank, keeping ``keep`` frames of margin."""
    active = np.nonzero(np.exp(logp[:, BLANK]) < blank_p)[0]
    if len(active) == 0:
        return logp[:1]
    a, b = active[0], active[-1]
    return logp[max(0, a - keep) : b + keep + 1]


def concat_lines(logps: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([trim_blank(lp) for lp in logps], axis=0)
