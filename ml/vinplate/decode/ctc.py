"""CTC decoding: greedy, grammar-constrained prefix beam search, forced alignment, exact
sequence likelihood.

All functions take ``logp`` = log-softmax output of the recogniser, shape [T, V]
(V = vocab.VOCAB_SIZE, index 0 = CTC blank).

The grammar-constrained beam search is the "reconstruction" mechanism, and it is worth
being exact about what that means: when a character is smudged or glared out, the network
returns a flat distribution there. A constrained search can only choose among characters
the grammar permits at that position (a digit slot cannot become "O"; a state code must be
a real state code). That resolves *ambiguity*. It cannot recover information that is not in
the pixels, so every such character is reported with its (low) evidence and flagged as
inferred - callers must treat an inferred identifier as a lead to confirm, not a reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vinplate.vocab import BLANK, CHARS, VOCAB_SIZE

NEG = -1e30
_C2I = {c: i + 1 for i, c in enumerate(CHARS)}


def log_softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def greedy(logp: np.ndarray) -> tuple[str, list[float]]:
    """Best-path decode. Returns text and per-character probability."""
    idx = logp.argmax(axis=1)
    prob = np.exp(logp.max(axis=1))
    out, conf, prev = [], [], BLANK
    for t, i in enumerate(idx):
        if i != BLANK and i != prev:
            out.append(CHARS[i - 1])
            conf.append(float(prob[t]))
        elif i != BLANK and i == prev and conf:
            conf[-1] = max(conf[-1], float(prob[t]))
        prev = i
    return "".join(out), conf


def ctc_loglik(logp: np.ndarray, labels: list[int]) -> float:
    """log P(labels | logp), summed over all CTC alignments (forward algorithm)."""
    T = logp.shape[0]
    L = len(labels)
    if L == 0:
        return float(logp[:, BLANK].sum())
    if T < L:
        return NEG
    S = 2 * L + 1
    ext = np.full(S, BLANK, dtype=np.int64)
    ext[1::2] = labels
    # transitions allowed from s-2 only when ext[s] != blank and ext[s] != ext[s-2]
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, BLANK]
    alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        a1 = np.concatenate(([NEG], alpha[:-1]))
        a2 = np.concatenate(([NEG, NEG], alpha[:-2]))
        a2 = np.where(skip, a2, NEG)
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + logp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def ctc_align(logp: np.ndarray, labels: list[int]) -> list[tuple[int, int, float]]:
    """Viterbi forced alignment. Returns, per label, (first_frame, last_frame, max_prob)."""
    T = logp.shape[0]
    L = len(labels)
    if L == 0 or T < L:
        return [(0, 0, 0.0)] * L
    S = 2 * L + 1
    ext = np.full(S, BLANK, dtype=np.int64)
    ext[1::2] = labels
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    score = np.full((T, S), NEG)
    back = np.zeros((T, S), dtype=np.int8)
    score[0, 0] = logp[0, BLANK]
    score[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        s0 = score[t - 1]
        s1 = np.concatenate(([NEG], s0[:-1]))
        s2 = np.where(skip, np.concatenate(([NEG, NEG], s0[:-2])), NEG)
        stack = np.stack([s0, s1, s2])
        arg = stack.argmax(axis=0)
        score[t] = stack[arg, np.arange(S)] + logp[t, ext]
        back[t] = arg
    s = S - 1 if score[-1, S - 1] >= score[-1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    out = []
    for k in range(L):
        frames = np.nonzero(path == 2 * k + 1)[0]
        if len(frames) == 0:
            out.append((0, 0, 0.0))
            continue
        probs = np.exp(logp[frames, labels[k]])
        out.append((int(frames[0]), int(frames[-1]), float(probs.max())))
    return out


@dataclass
class Hypothesis:
    text: str
    ctc_logp: float
    prior: float = 0.0
    valid: bool = True
    score: float = 0.0
    char_conf: list[float] = field(default_factory=list)


def _lse(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def prefix_beam_search(
    logp: np.ndarray,
    grammar=None,
    beam: int = 32,
    top_k: int = 12,
    min_logp: float = -14.0,
    prior_weight: float = 1.0,
    blank_skip: float = 0.995,
) -> list[Hypothesis]:
    """CTC prefix beam search; with a grammar, only grammar-legal prefixes are extended.

    ``blank_skip``: on a time step where the blank probability is at least this, no character is
    emitted (the beams only absorb the blank / continue a repeat). A CTC spike train is blank on
    ~85-90% of its steps, so this is most of the speed-up, and it can only drop paths worth <0.5%.

    Returns hypotheses best-first by ``ctc_logp + prior_weight * grammar prior``. With a
    grammar, only complete (accepting) strings are returned; if none survives the beam the
    list is empty and the caller should fall back to ``greedy`` and mark the read invalid.
    """
    T = logp.shape[0]
    log_blank_skip = float(np.log(blank_skip))
    gs0 = grammar.start() if grammar is not None else None
    beams: dict[str, list] = {"": [0.0, NEG, gs0]}  # prefix -> [p_blank, p_nonblank, gstate]
    for t in range(T):
        lp = logp[t]
        if lp[BLANK] >= log_blank_skip:
            cand: list[int] = []
        else:
            order = np.argsort(-lp[1:])[:top_k] + 1
            cand = [int(c) for c in order if lp[c] > min_logp]
        nxt: dict[str, list] = {}

        def slot(prefix: str, gstate):
            s = nxt.get(prefix)
            if s is None:
                s = nxt[prefix] = [NEG, NEG, gstate]
            return s

        for prefix, (pb, pnb, gs) in beams.items():
            ptot = _lse(pb, pnb)
            s = slot(prefix, gs)
            s[0] = _lse(s[0], ptot + float(lp[BLANK]))
            if prefix:
                s[1] = _lse(s[1], pnb + float(lp[_C2I[prefix[-1]]]))
            for c in cand:
                ch = CHARS[c - 1]
                ns = None
                if grammar is not None:
                    ns = grammar.step(gs, ch)
                    if ns is None:
                        continue
                add = (pb if (prefix and prefix[-1] == ch) else ptot) + float(lp[c])
                s2 = slot(prefix + ch, ns)
                s2[1] = _lse(s2[1], add)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(kv[1][0], kv[1][1]))[:beam])

    hyps: list[Hypothesis] = []
    for prefix, (pb, pnb, gs) in beams.items():
        if not prefix:
            continue
        if grammar is not None and not grammar.accepting(gs):
            continue
        prior = 0.0
        if grammar is not None:
            p = grammar.log_prior(prefix)
            if p is None:
                continue
            prior = p
        ll = _lse(pb, pnb)
        hyps.append(Hypothesis(prefix, ll, prior, True, ll + prior_weight * prior))
    hyps.sort(key=lambda h: -h.score)
    return hyps


def char_confidence(logp: np.ndarray, text: str) -> list[float]:
    """Per-character evidence: peak posterior of that label in its aligned frames."""
    return [a[2] for a in ctc_align(logp, [_C2I[c] for c in text])]


def alternatives(logp: np.ndarray, text: str, k: int = 3) -> list[list[tuple[str, float]]]:
    """For each character, the k best *other* characters the frames in its span favoured."""
    labels = [_C2I[c] for c in text]
    spans = ctc_align(logp, labels)
    out = []
    for lab, (a, b, _) in zip(labels, spans):
        peak = np.exp(logp[a : b + 1, 1:]).max(axis=0)  # over non-blank classes
        peak[lab - 1] = -1.0
        peak[len(CHARS) - 1] = -1.0  # '?' is a statement of ignorance, not a substitute
        top = np.argsort(-peak)[:k]
        out.append([(CHARS[i], float(peak[i])) for i in top])
    return out


def suspected_gaps(spans: list[tuple[int, int, float]], T: int, factor: float = 1.9,
                   edge_factor: float = 2.3) -> list[int]:
    """Where a character may have been *dropped*.

    CTC cannot tell "a washed-out character" from "no character": both are blank frames.
    But characters on a plate / VIN label sit at a roughly even pitch, so a gap much wider
    than the typical pitch is evidence of a missing one. Returns insertion indices: ``k``
    means "between character k-1 and k" (0 = before the first, len(text) = after the last).

    ``factor`` is 1.9, not lower, because plates print deliberately WIDER gaps between their
    groups (state | RTO | series | number): measured on PP-OCR posteriors of correct plate reads a
    group gap is 1.5-1.75 pitches (the recogniser has only ~4 frames per character, so pitch is
    quantised to +-0.25), while one missing character gives >= 2.0. At 1.7 most correctly read
    plates were flagged "possible dropped character". (Looking at the blank probability inside the gap
    does not help: PP-OCR smears each character over neighbouring frames.)
    """
    n = len(spans)
    if n < 4:
        return []
    starts = np.array([s[0] for s in spans], dtype=float)
    pitch = float(np.median(np.diff(starts)))
    if pitch <= 0:
        return []
    out = [k + 1 for k, d in enumerate(np.diff(starts)) if d > factor * pitch]
    if starts[0] > edge_factor * pitch:
        out.append(0)
    if (T - 1 - spans[-1][1]) > edge_factor * pitch:
        out.append(n)
    return out


def posterior_over(hyps: list[Hypothesis]) -> list[float]:
    """Beam-normalised posterior of each hypothesis (an approximation: mass outside the
    beam is ignored, which makes the result mildly over-confident)."""
    if not hyps:
        return []
    s = np.array([h.score for h in hyps])
    s = s - s.max()
    p = np.exp(s)
    return list(p / p.sum())


__all__ = [
    "Hypothesis", "VOCAB_SIZE", "alternatives", "char_confidence", "ctc_align", "ctc_loglik",
    "greedy", "log_softmax", "posterior_over", "prefix_beam_search", "suspected_gaps",
]
