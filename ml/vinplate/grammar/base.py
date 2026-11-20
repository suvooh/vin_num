"""Tiny template grammar: an NFA simulated incrementally, plus scoring/segmentation.

Why not regex? The decoder needs *incremental* queries ("can this prefix still become a
plate?") at every CTC step for every beam hypothesis, and the same grammar must be
exportable as a plain table (see ``to_dfa``) for the native mobile decoder. Both fall
out of a small explicit NFA; Python's ``re`` offers neither.

A grammar is a set of templates; a template is a sequence of tokens:
  Cls(name, chars, lo, hi)   - between lo and hi characters, each drawn from ``chars``
  Choice(name, options)      - exactly one of ``options`` ("" makes it optional)

Hard validity comes from the NFA. *Soft* preferences (modern vs legacy layouts, rare
formats) are expressed as log-priors on complete strings via ``Template.score``, and are
applied when ranking the decoder's final N-best, never to forbid anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator

from vinplate.vocab import ALPHABET, UNK

# NFA element: (template index, token index, progress) - progress is a count for Cls and
# the consumed prefix (str) for Choice.
Element = tuple[int, int, "int | str"]
State = frozenset


@dataclass(frozen=True)
class Cls:
    name: str
    chars: str
    lo: int = 1
    hi: int = 1


@dataclass(frozen=True)
class Choice:
    name: str
    options: tuple[str, ...]


Token = "Cls | Choice"


@dataclass
class Template:
    name: str
    tokens: tuple
    log_prior: float = 0.0
    # parts (token name -> text) -> extra log-prior (e.g. "legacy short number" penalty)
    score: Callable[[dict[str, str]], float] = field(default=lambda parts: 0.0)


def _p0(tok) -> "int | str":
    return 0 if isinstance(tok, Cls) else ""


def _can_finish(tok, p) -> bool:
    return p >= tok.lo if isinstance(tok, Cls) else p in tok.options


def _consume(tok, p, ch: str):
    """Progress after consuming ``ch`` in ``tok`` at progress ``p``, or None."""
    if isinstance(tok, Cls):
        return p + 1 if (ch in tok.chars and p < tok.hi) else None
    q = p + ch
    return q if any(o.startswith(q) for o in tok.options) else None


class Grammar:
    def __init__(self, name: str, templates: list[Template]):
        self.name = name
        self.templates = templates
        self._start: State = frozenset(
            (t, 0, _p0(tpl.tokens[0])) for t, tpl in enumerate(templates)
        )
        self._cache: dict[tuple[State, str], State | None] = {}

    # ------------------------------------------------------------ incremental NFA
    def start(self) -> State:
        return self._start

    def _targets(self, t: int, i: int, p) -> Iterator[tuple[int, "int | str"]]:
        toks = self.templates[t].tokens
        yield i, p
        while _can_finish(toks[i], p) and i + 1 < len(toks):
            i += 1
            p = _p0(toks[i])
            yield i, p

    def step(self, state: State, ch: str) -> State | None:
        """Advance by one character. ``?`` (unreadable) is a wildcard: it advances every way
        any legal character could, so a prefix containing it stays alive as long as *some*
        completion is legal."""
        key = (state, ch)
        if key in self._cache:
            return self._cache[key]
        if ch == UNK:
            out_w: set = set()
            for c in ALPHABET:
                n = self.step(state, c)
                if n:
                    out_w |= n
            res = frozenset(out_w) if out_w else None
            self._cache[key] = res
            return res
        out = set()
        for t, i, p in state:
            toks = self.templates[t].tokens
            for j, q in self._targets(t, i, p):
                np_ = _consume(toks[j], q, ch)
                if np_ is not None:
                    out.add((t, j, np_))
        res = frozenset(out) if out else None
        self._cache[key] = res
        return res

    def accepting(self, state: State) -> bool:
        for t, i, p in state:
            toks = self.templates[t].tokens
            if not _can_finish(toks[i], p):
                continue
            if all(_can_finish(tk, _p0(tk)) for tk in toks[i + 1 :]):
                return True
        return False

    def is_valid(self, text: str) -> bool:
        s = self.start()
        for ch in text:
            s = self.step(s, ch)
            if s is None:
                return False
        return self.accepting(s)

    # ------------------------------------------------------- segmentation + prior
    def segmentations(self, text: str) -> list[tuple[Template, dict[str, str]]]:
        found: list[tuple[Template, dict[str, str]]] = []
        for tpl in self.templates:
            for parts in _segment(tpl.tokens, text):
                found.append((tpl, parts))
        return found

    def best_parse(self, text: str) -> tuple[Template, dict[str, str], float] | None:
        best = None
        for tpl, parts in self.segmentations(text):
            lp = tpl.log_prior + tpl.score(parts)
            if best is None or lp > best[2]:
                best = (tpl, parts, lp)
        return best

    def log_prior(self, text: str) -> float | None:
        """Soft prior of a complete string; None if the grammar rejects it."""
        b = self.best_parse(text)
        return None if b is None else b[2]

    # --------------------------------------------------------- portable export
    def to_dfa(self, alphabet: str) -> dict:
        """Determinise into a plain table the native (Kotlin/Swift/C++) decoder can load."""
        ids: dict[State, int] = {self._start: 0}
        order = [self._start]
        trans: dict[int, dict[str, int]] = {}
        q = deque([self._start])
        while q:
            s = q.popleft()
            row = {}
            for ch in alphabet:
                n = self.step(s, ch)
                if n is None:
                    continue
                if n not in ids:
                    ids[n] = len(ids)
                    order.append(n)
                    q.append(n)
                row[ch] = ids[n]
            trans[ids[s]] = row
        return {
            "name": self.name,
            "alphabet": alphabet,
            "start": 0,
            "accepting": [ids[s] for s in order if self.accepting(s)],
            "transitions": {str(k): v for k, v in sorted(trans.items())},
        }


def _segment(tokens: tuple, text: str, i: int = 0, pos: int = 0) -> Iterator[dict[str, str]]:
    if i == len(tokens):
        if pos == len(text):
            yield {}
        return
    tok = tokens[i]
    if isinstance(tok, Cls):
        n = 0
        while pos + n < len(text) and n < tok.hi and (text[pos + n] in tok.chars or text[pos + n] == UNK):
            n += 1
        for take in range(n, tok.lo - 1, -1):
            for rest in _segment(tokens, text, i + 1, pos + take):
                yield {tok.name: text[pos : pos + take], **rest}
    else:
        for opt in tok.options:
            seg = text[pos : pos + len(opt)]
            if len(seg) == len(opt) and all(a == b or a == UNK for a, b in zip(seg, opt)):
                for rest in _segment(tokens, text, i + 1, pos + len(opt)):
                    yield {tok.name: seg, **rest}
