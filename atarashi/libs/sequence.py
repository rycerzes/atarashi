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

from collections.abc import Mapping
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

    def __init__(self, references: Mapping[str, str], shingle: int = 4, min_tokens: int = 4):
        self.shingle = shingle
        self.min_tokens = min_tokens
        self._ref_tokens: dict[str, list[str]] = {}
        self._index: dict[tuple[str, ...], set[str]] = {}
        self._exact: dict[str, str] = {}
        for name, text in references.items():
            toks = tokens(text)
            if not toks:
                continue
            self._exact[normalize(text)] = name
            self._ref_tokens[name] = toks
            for shingle_key in _shingles(toks, shingle):
                self._index.setdefault(shingle_key, set()).add(name)

    def exact(self, query: str) -> str | None:
        """Return a shortname when the whole normalized input equals a reference."""
        return self._exact.get(normalize(query))

    def _candidates(self, q_tokens: list[str]) -> set[str]:
        candidates: set[str] = set()
        for shingle_key in _shingles(q_tokens, self.shingle):
            candidates |= self._index.get(shingle_key, set())
        return candidates

    def match(self, query: str, min_score: float = 0.5, top_k: int = 5) -> list[SpanMatch]:
        """Return the best reference matches within ``query``, ranked by coverage."""
        q_tokens = tokens(query)
        if len(q_tokens) < self.shingle:
            return []
        results: list[SpanMatch] = []
        for name in self._candidates(q_tokens):
            ref = self._ref_tokens[name]
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
            results.append(SpanMatch(name, score, matched, len(ref), start, end))
        results.sort(key=lambda m: (m.score, m.matched_tokens), reverse=True)
        return results[:top_k]
