"""Posterior simulator used by tests and by ``eval/`` smoke checks.

Builds the [T, V] log-probabilities a CTC recogniser would emit for a known string, with
controllable failure modes (confusions, glared-out characters), so decoder behaviour can be
checked against ground truth without any trained model.
"""

from __future__ import annotations

import numpy as np

from vinplate.vocab import BLANK, CHARS, UNK, VOCAB_SIZE


def make_logp(
    text: str,
    *,
    frames_per_char: int = 3,
    p_peak: float = 0.93,
    confuse: dict[int, tuple[str, float]] | None = None,
    flat: tuple[int, ...] = (),
    unreadable: tuple[int, ...] = (),
    eps: float = 0.002,
    smeared: bool = False,
) -> np.ndarray:
    """``confuse[i] = (wrong_char, p_wrong)``: position i is read as wrong_char with p_wrong,
    and the true char keeps the remainder of the mass. ``flat`` positions are washed out:
    mostly blank, the rest spread evenly (what a glared/smudged character looks like to a
    well-trained network).

    By default the output is *peaky* - one spike frame per character, blanks elsewhere -
    which is what a CTC-trained network produces. ``unreadable`` positions are emitted as the
    explicit '?' class (what a network trained with illegible-character labels does when
    glare or dirt hides a character), as opposed to ``flat`` where it merely shrugs. ``smeared=True`` repeats the character
    distribution over all of its frames; that is a harsher, less realistic stress case in
    which two competing characters can both be (legally) emitted in one span.
    """
    confuse = confuse or {}
    rows = []
    for i, ch in enumerate(text):
        for f in range(frames_per_char):
            spike = smeared or f == 0
            p = np.full(VOCAB_SIZE, eps)
            if not spike:
                p[BLANK] = 0.95
            elif i in unreadable:
                p[CHARS.index(UNK) + 1] = p_peak
            elif i in flat:
                p[BLANK] = 0.6
                p[1:] = 0.4 / (VOCAB_SIZE - 1)
            else:
                k = CHARS.index(ch) + 1
                if i in confuse:
                    wrong, pw = confuse[i]
                    p[CHARS.index(wrong) + 1] = pw
                    p[k] = p_peak - pw if p_peak - pw > 0 else 0.3
                else:
                    p[k] = p_peak
            rows.append(p / p.sum())
        gap = np.full(VOCAB_SIZE, eps)
        gap[BLANK] = 0.95
        rows.append(gap / gap.sum())
    return np.log(np.stack(rows))
