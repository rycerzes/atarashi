#!/usr/bin/env python3
"""Token-sequence license matcher with matched-span and coverage scoring.

The core of the native cascade (Phase 1): identify which reference license text
is present in an input by aligning normalized token sequences and scoring by how
much of the reference is covered. This is the algorithm class production scanners
(ScanCode, askalono) use — bag-of-words similarity was shown to plateau on this
task. Candidate references are found via a shingle inverted index so a query is
only aligned against references it shares n-grams with.

Two paths:
  * ``exact`` — O(1) hash lookup when the whole input is a known license text.
  * ``match`` — span alignment with coverage in [0, 1] for embedded notices/text.

This is a correct first implementation; the alignment can later be swapped for an
Aho-Corasick / suffix-automaton core for speed without changing the contract.

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher as _DiffMatcher

from atarashi.libs.normalize import normalize, tokens


@dataclass(frozen=True)
class SpanMatch:
    """One reference matched within the query token stream."""

    shortname: str
    score: float  # coverage of the reference, in [0, 1]
    matched_tokens: int
    ref_tokens: int
    start: int  # matched span start token index in the query (inclusive)
    end: int  # matched span end token index in the query (exclusive)


def _shingles(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class LicenseMatcher:
    """Match input text against a fixed set of reference license texts."""

    def __init__(self, references: Iterable[tuple[str, str]],
                 shingle: int = 4, min_tokens: int = 4,
                 required: Iterable[tuple[str, list[str]]] | None = None):
        self.shingle = shingle
        self.min_tokens = min_tokens
        # A license may have several reference units (full text, header/notice, …),
        # so units are stored by index and mapped back to a shortname.
        self._unit_name: list[str] = []
        self._unit_tokens: list[list[str]] = []
        self._index: dict[tuple[str, ...], set[int]] = {}
        self._exact: dict[str, str] = {}
        # Key phrases that MUST appear in the input for a reference to match — the
        # lever that keeps generic boilerplate from matching and separates close
        # variants (e.g. an "Affero"/version clause). Empty => no gating.
        self._required: dict[str, list[list[str]]] = {
            name: [tokens(p) for p in phrases if tokens(p)]
            for name, phrases in (required or [])
        }
        for name, text in references:
            toks = tokens(text)
            if not toks:
                continue
            self._exact.setdefault(normalize(text), name)
            uid = len(self._unit_name)
            self._unit_name.append(name)
            self._unit_tokens.append(toks)
            for shingle_key in _shingles(toks, shingle):
                self._index.setdefault(shingle_key, set()).add(uid)

    def _has_required(self, name: str, q_joined: str) -> bool:
        phrases = self._required.get(name)
        if not phrases:
            return True
        return all(f" {' '.join(p)} " in q_joined for p in phrases)

    def exact(self, query: str) -> str | None:
        """Return a shortname when the whole normalized input equals a reference."""
        return self._exact.get(normalize(query))

    def _candidates(self, q_tokens: list[str]) -> set[int]:
        candidates: set[int] = set()
        for shingle_key in _shingles(q_tokens, self.shingle):
            candidates |= self._index.get(shingle_key, set())
        return candidates

    def match(self, query: str, min_score: float = 0.5, top_k: int = 5) -> list[SpanMatch]:
        """Return the best reference matches within ``query``, ranked by coverage.

        A license with several units (full text + header/notice) is reported once,
        keeping its best-covering unit.
        """
        q_tokens = tokens(query)
        if len(q_tokens) < self.shingle:
            return []
        q_joined = f" {' '.join(q_tokens)} "
        best: dict[str, SpanMatch] = {}
        for uid in self._candidates(q_tokens):
            name = self._unit_name[uid]
            if not self._has_required(name, q_joined):
                continue
            ref = self._unit_tokens[uid]
            blocks = [b for b in _DiffMatcher(None, ref, q_tokens, autojunk=False)
                      .get_matching_blocks() if b.size > 0]
            matched = sum(b.size for b in blocks)
            if matched < self.min_tokens:
                continue
            score = matched / len(ref)
            if score < min_score:
                continue
            start = min(b.b for b in blocks)
            end = max(b.b + b.size for b in blocks)
            cand = SpanMatch(name, score, matched, len(ref), start, end)
            prev = best.get(name)
            if prev is None or (cand.score, cand.matched_tokens) > (prev.score, prev.matched_tokens):
                best[name] = cand
        results = sorted(best.values(), key=lambda m: (m.score, m.matched_tokens), reverse=True)
        return results[:top_k]
