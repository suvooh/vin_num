"""From recogniser posteriors to a *reading*: constrained decode, candidate rescoring,
multi-frame fusion, and an honest account of how much of the answer was inferred.

Design rules (they matter for an identity / fraud workflow):
  * never return a bare string - always the evidence behind each character;
  * a character the pixels did not support is reported as ``inferred``, never silently kept;
  * a character that may have been *dropped* is reported too (``suspected_gaps``): CTC makes
    a washed-out character indistinguishable from an absent one, and a legal-looking
    shorter plate is the most dangerous error this stage can make;
  * ``grammar_cost`` says how far the constraint had to bend the network's own reading;
    a large cost means the grammar manufactured a plausible identifier out of noise;
  * ``status`` is the one field a caller should branch on: accept / review / reject;
  * ``confidence`` is an uncalibrated heuristic until ``eval/calibrate.py`` has been fitted
    on a held-out labelled set (``calibrated`` records which).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vinplate.decode.ctc import (
    Hypothesis,
    alternatives,
    ctc_align,
    ctc_loglik,
    greedy,
    posterior_over,
    prefix_beam_search,
    suspected_gaps,
)
from vinplate.vocab import ALPHABET, UNK, encode_text

REVIEW_COST = 6.0     # nats the grammar may bend the raw reading before a human should look
REJECT_COST = 12.0    # beyond this the "reading" is the grammar talking, not the image


@dataclass
class Reading:
    text: str
    valid: bool                      # accepted by the grammar
    status: str                      # "accept" | "review" | "reject"
    confidence: float                # min(sequence posterior, geometric-mean char evidence)
    seq_posterior: float
    char_conf: list[float]
    inferred: list[int]              # positions whose pixel evidence was weak
    suspected_gaps: list[int]        # insertion points where a character may be missing
    grammar_cost: float              # nats the constraint bent the raw reading by
    greedy_text: str                 # what the network alone said
    n_best: list[tuple[str, float]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    calibrated: bool = False

    @property
    def needs_review(self) -> bool:
        return self.status != "accept"

    @property
    def masked(self) -> str:
        """``text`` with every inferred character replaced by '?' and a '?' inserted at each
        suspected gap. This - not ``text`` - is what to display or match against a registry
        whenever ``status`` is not "accept": it states what is *unknown* instead of showing
        the decoder's guess as if it were read."""
        chars = ["?" if i in self.inferred else c for i, c in enumerate(self.text)]
        for k in sorted(self.suspected_gaps, reverse=True):
            chars.insert(k, "?")
        return "".join(chars)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["needs_review"] = self.needs_review
        d["masked"] = self.masked
        return d


# ----------------------------------------------------------------- candidate generation
def _weakest(char_conf, max_pos, conf_below):
    idx = sorted(range(len(char_conf)), key=lambda i: char_conf[i])[:max_pos]
    return [i for i in idx if char_conf[i] < conf_below]


def _substitutions(logp, text, char_conf, grammar, max_pos=6, alts=3, conf_below=0.9,
                   flat_below=0.25):
    """Single-character substitutions at the weakest positions, using the alternatives the
    network itself favoured. If exactly ONE character is unreadable every legal character is
    tried there, which is what lets a mandatory check digit pin it down. With two or more
    unreadable characters nothing is enumerated: the space explodes and an inferred pair is
    not something to present as a reading."""
    weak = _weakest(char_conf, max_pos, conf_below)
    if not weak:
        return []
    out: list[str] = []
    flat = [i for i in weak if char_conf[i] < flat_below]
    if len(flat) == 1:
        i = flat[0]
        for ch in ALPHABET:
            cand = text[:i] + ch + text[i + 1 :]
            if cand != text and (grammar is None or grammar.is_valid(cand)):
                out.append(cand)
    alt = alternatives(logp, text, k=alts)
    for i in weak:
        for ch, _ in alt[i]:
            cand = text[:i] + ch + text[i + 1 :]
            if cand not in out and (grammar is None or grammar.is_valid(cand)):
                out.append(cand)
    return out


def _insertions(text, gaps, grammar):
    """If exactly one gap looks like a dropped character, try every legal character there."""
    if len(gaps) != 1:
        return []
    k = gaps[0]
    out = []
    for ch in ALPHABET:
        cand = text[:k] + ch + text[k:]
        if grammar is None or grammar.is_valid(cand):
            out.append(cand)
    return out


def _fill_unknown(best: Hypothesis, pool: list[Hypothesis], grammar, prior_weight: float) -> Hypothesis:
    """Exactly one '?': let the grammar/checksum narrow it, under a UNIFORM prior over the
    alphabet for what the character could be (the pixels said nothing). A filled reading
    therefore only beats the '?' reading when the prior - in practice a mandatory check
    digit - is worth more than log(36) nats; otherwise the '?' stays and the candidate fills
    are kept in the N-best so a reviewer can see them. Never enumerates two or more."""
    if grammar is None or best.text.count(UNK) != 1:
        return best
    i = best.text.index(UNK)
    uniform = -float(np.log(len(ALPHABET)))
    have = {h.text for h in pool}
    for ch in ALPHABET:
        cand = best.text[:i] + ch + best.text[i + 1 :]
        if cand in have or not grammar.is_valid(cand):
            continue
        pool.append(_score(cand, best.ctc_logp + uniform, grammar, prior_weight))
    pool.sort(key=lambda h: -h.score)
    return pool[0]


def _score(text: str, ll: float, grammar, prior_weight: float) -> Hypothesis:
    prior = 0.0
    if grammar is not None:
        p = grammar.log_prior(text)
        prior = p if p is not None else -50.0
    return Hypothesis(text, ll, prior, True, ll + prior_weight * prior)


def _analyse(logp, text):
    spans = ctc_align(logp, encode_text(text))
    return [s[2] for s in spans], suspected_gaps(spans, logp.shape[0])


# -------------------------------------------------------------------------- the verdict
def _finish(best: Hypothesis, pool: list[Hypothesis], char_conf, gaps, greedy_text, cost, *,
            infer_below, max_inferred, min_seq, valid) -> Reading:
    post = posterior_over(pool)
    seq = float(post[pool.index(best)]) if pool else 0.0
    inferred = [i for i, c in enumerate(char_conf) if c < infer_below or best.text[i] == UNK]
    ev = [0.0 if best.text[i] == UNK else c for i, c in enumerate(char_conf)]
    gm = float(np.exp(np.mean(np.log(np.clip(ev, 1e-6, 1.0))))) if ev else 0.0
    reasons: list[str] = []
    review = False
    reject = False
    if not valid:
        reject = True
        reasons.append("no legal layout fits this read")
    if cost > REJECT_COST:
        reject = True
        reasons.append(f"grammar bent the raw reading by {cost:.0f} nats: output is mostly inferred")
    elif cost > REVIEW_COST:
        review = True
        reasons.append(f"grammar bent the raw reading by {cost:.1f} nats")
    if seq < min_seq:
        review = True
        reasons.append(f"ambiguous: top reading holds only {seq:.0%} of the probability")
    if UNK in best.text:
        # A '?' that survived to the output is simply not read. The inferred-character budget
        # exists for a character a mandatory checksum pinned down, never for one still unknown.
        review = True
        reasons.append(f"{best.text.count(UNK)} character(s) unreadable and not recoverable")
    if len(inferred) > max_inferred:
        review = True
        reasons.append(f"{len(inferred)} characters rest on weak pixel evidence")
    if gaps:
        review = True
        reasons.append(f"possible dropped character at position(s) {gaps}")
    if best.text != greedy_text and valid and not reject:
        reasons.append("grammar/priors changed the network's raw reading")
    status = "reject" if reject else ("review" if review else "accept")
    return Reading(
        text=best.text, valid=valid, status=status, confidence=min(seq, gm),
        seq_posterior=seq, char_conf=[round(float(c), 3) for c in char_conf],
        inferred=inferred, suspected_gaps=gaps, grammar_cost=round(float(cost), 2),
        greedy_text=greedy_text,
        n_best=[(h.text, round(float(p), 4))
                for h, p in sorted(zip(pool, post), key=lambda x: -x[1])[:5]],
        reasons=reasons,
    )


def _no_legal(logp_list, infer_below, max_inferred, min_seq) -> Reading:
    g = [greedy(lp)[0] for lp in logp_list]
    text = max(set(g), key=g.count) if g else ""
    h = Hypothesis(text, -1e9, 0.0, False, 0.0)
    return _finish(h, [h], [0.0] * len(text), [], text, 99.0, infer_below=infer_below,
                   max_inferred=max_inferred, min_seq=min_seq, valid=False)


def read_line(
    logp: np.ndarray,
    grammar=None,
    *,
    beam: int = 32,
    prior_weight: float = 1.0,
    infer_below: float = 0.5,
    max_inferred: int = 1,
    min_seq: float = 0.85,
) -> Reading:
    g_text, _ = greedy(logp)
    hyps = prefix_beam_search(logp, grammar, beam=beam, prior_weight=prior_weight)
    if not hyps:
        return _no_legal([logp], infer_below, max_inferred, min_seq)

    pool = {h.text: h for h in hyps}

    def add(cands):
        for cand in cands:
            if cand not in pool:
                pool[cand] = _score(cand, ctc_loglik(logp, encode_text(cand)), grammar, prior_weight)

    for h in hyps[:2]:
        cc, gaps = _analyse(logp, h.text)
        add(_substitutions(logp, h.text, cc, grammar))
        if h is hyps[0]:
            add(_insertions(h.text, gaps, grammar))
    ranked = sorted(pool.values(), key=lambda h: -h.score)
    best = _fill_unknown(ranked[0], ranked, grammar, prior_weight)
    cc, gaps = _analyse(logp, best.text)
    return _finish(best, ranked, cc, gaps, g_text, _cost(logp, g_text, best), infer_below=infer_below,
                   max_inferred=max_inferred, min_seq=min_seq, valid=True)


def _cost(logp, g_text, best: Hypothesis) -> float:
    if not g_text:
        return 0.0
    return max(0.0, ctc_loglik(logp, encode_text(g_text)) - best.ctc_logp)


# ------------------------------------------------------------------------- multi-frame
def entropy_fuse(logps: list[np.ndarray], weights: list[float] | None = None,
                 tau: float = 0.5) -> np.ndarray:
    """Uncertainty-weighted average of time-aligned posteriors.

    Rectified plate crops share a canonical width, so the same character lands on the same
    time steps in every frame. At each step, frames that are unsure (a glared character:
    high entropy) are down-weighted exponentially against frames that are sure. Frames of
    different length are resampled to the longest (nearest step).
    """
    T = max(lp.shape[0] for lp in logps)
    ws = weights or [1.0] * len(logps)
    num = np.zeros((T, logps[0].shape[1]))
    den = np.zeros((T, 1))
    for lp, wf in zip(logps, ws):
        if lp.shape[0] != T:
            lp = lp[np.minimum((np.arange(T) * lp.shape[0] / T).astype(int), lp.shape[0] - 1)]
        p = np.exp(lp)
        h = -(p * lp).sum(axis=1, keepdims=True)
        w = wf * np.exp(-h / tau)
        num += w * p
        den += w
    return np.log(num / den + 1e-12)


def fuse_frames(
    logps: list[np.ndarray],
    grammar=None,
    *,
    weights: list[float] | None = None,
    beam: int = 16,
    prior_weight: float = 1.0,
    per_frame_cap: float = 12.0,
    max_candidates: int = 24,
    infer_below: float = 0.5,
    max_inferred: int = 1,
    min_seq: float = 0.85,
) -> Reading:
    """Read one physical identifier from several frames (burst / video).

    Glare and smudges move as the phone moves, so different frames lose different
    characters. Two complementary mechanisms:
      1. *posterior fusion*: uncertainty-weighted averaging of the time-aligned frames, which
         recovers a character that is clear in at least one frame even when every other
         frame dropped it;
      2. *candidate rescoring*: candidates from every frame and from the fused posterior are
         scored under ALL frames (product of experts), each frame's penalty capped so one
         wholly washed-out frame cannot veto the rest.
    """
    w = weights or [1.0] * len(logps)
    fused = entropy_fuse(logps, w)
    g_text, _ = greedy(fused)
    pool_texts: dict[str, None] = {}

    def add(cands):
        for c in cands:
            pool_texts[c] = None

    for src in [fused, *logps]:
        hyps = prefix_beam_search(src, grammar, beam=beam, prior_weight=prior_weight)[:4]
        add(h.text for h in hyps)
        if hyps:
            cc, gaps = _analyse(src, hyps[0].text)
            add(_substitutions(src, hyps[0].text, cc, grammar))
            add(_insertions(hyps[0].text, gaps, grammar))
    if not pool_texts:
        return _no_legal(logps, infer_below, max_inferred, min_seq)

    ref = [ctc_loglik(lp, encode_text(t)) if (t := greedy(lp)[0]) else -1e9 for lp in logps]
    if len(pool_texts) > max_candidates:
        # cheap pre-rank under the fused posterior alone; only the best get the per-frame rescoring
        pre = sorted(pool_texts, key=lambda t: -_score(t, ctc_loglik(fused, encode_text(t)), grammar, prior_weight).score)
        pool_texts = dict.fromkeys(pre[:max_candidates])
    pool: list[Hypothesis] = []
    for text in pool_texts:
        lab = encode_text(text)
        ll = sum(wf * max(ctc_loglik(lp, lab), r - per_frame_cap) for lp, wf, r in zip(logps, w, ref))
        pool.append(_score(text, ll, grammar, prior_weight))
    pool.sort(key=lambda h: -h.score)
    best = _fill_unknown(pool[0], pool, grammar, prior_weight)

    # Evidence per character = strongest support in the fused posterior or any single frame.
    per_frame = [_analyse(lp, best.text)[0] for lp in logps]
    cc_f, gaps = _analyse(fused, best.text)
    cc = np.maximum(np.max(per_frame, axis=0), cc_f).tolist()
    # cost: how far the constraint moved us from the fused posterior's own best reading
    cost = max(0.0, ctc_loglik(fused, encode_text(g_text)) - ctc_loglik(fused, encode_text(best.text))) if g_text else 0.0
    return _finish(best, pool, cc, gaps, g_text, cost, infer_below=infer_below,
                   max_inferred=max_inferred, min_seq=min_seq, valid=True)
