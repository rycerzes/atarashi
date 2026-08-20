#!/usr/bin/env python3
"""Token-sequence license matcher with matched-span and coverage scoring.

The core of the native cascade: identify which reference license text is present in
an input by aligning normalized token sequences and scoring by how much of the
reference is covered. This is the algorithm class production scanners (ScanCode,
askalono) use — bag-of-words similarity was shown to plateau on this task.

Two paths:
  * ``exact`` — O(1) hash lookup when the whole input is a known license text.
  * ``match`` — span alignment with coverage in [0, 1] for embedded notices/text.

Alignment is seed-and-extend over shared shingles, not a general diff. A shared
n-gram is a *seed*; seeds on the same diagonal (equal ``query_pos - ref_pos``) are
consecutive pieces of one contiguous run, so walking the reference once and grouping
seeds by diagonal recovers every common run of >= ``shingle`` tokens in roughly
O(len(reference)). The previous implementation ran a full ``difflib.SequenceMatcher``
per candidate, which is O(n*m) and dominated scan time by an order of magnitude.

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from atarashi.libs.normalize import normalize, tokens


# A match must contain at least one contiguous run of this many tokens — a
# distinctive license phrase — so scattered common-word overlap does not match.
DEFAULT_MIN_RUN = 8

# Shingles are packed positionally into one int key. The base must be fixed before
# indexing starts — deriving it from vocabulary size would re-key every unit as the
# vocabulary grew — so it is a constant comfortably above any license vocabulary.
_BASE = 1 << 21

# Only the strongest candidates are aligned. Candidates are ranked by how many
# distinct shingles they share with the query, which upper-bounds the run length
# they can produce, so a unit far down that ranking cannot win. Generous by design:
# alignment is now cheap enough that the cap is a safety rail, not a tuning knob.
DEFAULT_MAX_CANDIDATES = 400


@dataclass(frozen=True)
class SpanMatch:
    """One reference matched within the query token stream."""

    shortname: str
    score: float  # coverage of the reference (matched / ref_tokens), in [0, 1]
    longest_run: int  # longest contiguous matched token run — the ranking signal
    matched_tokens: int
    ref_tokens: int
    start: int  # matched span start token index in the query (inclusive)
    end: int  # matched span end token index in the query (exclusive)


class LicenseMatcher:
    """Match input text against a fixed set of reference license texts."""

    def __init__(self, references: Iterable[tuple[str, str]],
                 shingle: int = 4,
                 required: Iterable[tuple[str, list[str]]] | None = None,
                 max_candidates: int = DEFAULT_MAX_CANDIDATES):
        self.shingle = shingle
        self.max_candidates = max_candidates
        # Tokens are interned to ints once, so alignment compares machine integers
        # rather than strings and shingles pack into a single int key.
        self._vocab: dict[str, int] = {}
        # A license may have several reference units (full text, header/notice, …),
        # so units are stored by index and mapped back to a shortname.
        self._unit_name: list[str] = []
        self._unit_ids: list[list[int]] = []
        # Shingle keys per unit, cached at index time. Recomputing them per candidate
        # was the single largest remaining cost once alignment stopped being O(n*m).
        self._unit_keys: list[list[int]] = []
        self._index: dict[int, set[int]] = {}
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
            ids = [self._intern(t) for t in toks]
            keys = self._keys(ids)
            uid = len(self._unit_name)
            self._unit_name.append(name)
            self._unit_ids.append(ids)
            self._unit_keys.append(keys)
            for key in set(keys):
                self._index.setdefault(key, set()).add(uid)

    def _intern(self, token: str) -> int:
        ident = self._vocab.get(token)
        if ident is None:
            ident = len(self._vocab)
            if ident >= _BASE:
                raise ValueError(f"vocabulary exceeded {_BASE} terms; raise _BASE")
            self._vocab[token] = ident
        return ident

    def _keys(self, ids: list[int]) -> list[int]:
        """Pack each ``shingle``-gram into one int key, positionally.

        Positional encoding in base ``_BASE``, so distinct n-grams get distinct
        keys — exact, not a hash, so there are no collisions to guard against.
        """
        n, out = self.shingle, []
        for i in range(len(ids) - n + 1):
            key = 0
            for j in range(i, i + n):
                key = key * _BASE + ids[j] + 1
            out.append(key)
        return out

    def _query_ids(self, toks: list[str]) -> list[int]:
        """Query tokens as ids; unseen tokens get -1, which no reference contains."""
        return [self._vocab.get(t, -1) for t in toks]

    def _has_required(self, name: str, q_joined: str) -> bool:
        phrases = self._required.get(name)
        if not phrases:
            return True
        return all(f" {' '.join(p)} " in q_joined for p in phrases)

    def exact(self, query: str) -> str | None:
        """Return a shortname when the whole normalized input equals a reference."""
        return self._exact.get(normalize(query))

    def _runs(self, ref_keys: list[int], qpos: dict[int, list[int]]):
        """Every common run of >= ``shingle`` tokens, as (ref_start, q_start, length).

        Walks the reference once. A seed at reference position ``i`` matching query
        position ``j`` lies on diagonal ``j - i``; a run is a maximal stretch of
        consecutive seeds on one diagonal, so the previous step's diagonals are all
        that must be carried forward.
        """
        n = self.shingle
        prev: dict[int, tuple[int, int, int]] = {}
        runs: list[tuple[int, int, int]] = []
        for i, key in enumerate(ref_keys):
            cur: dict[int, tuple[int, int, int]] = {}
            for j in qpos.get(key, ()):
                d = j - i
                seed = prev.get(d)
                cur[d] = (seed[0], seed[1], seed[2] + 1) if seed else (i, j, 1)
            for d, run in prev.items():
                if d not in cur:
                    runs.append(run)
            prev = cur
        runs.extend(prev.values())
        # A run of c consecutive shingles spans c + n - 1 tokens.
        return [(si, sj, c + n - 1) for si, sj, c in runs]

    def match(self, query: str, min_run: int = DEFAULT_MIN_RUN, top_k: int = 5) -> list[SpanMatch]:
        """Return the best reference matches within ``query``, ranked by the longest
        contiguous matched run.

        Ranking by the longest run (not coverage) rewards a distinctive license
        phrase over scattered common-word overlap, and is robust to references of
        different lengths. A license with several units (full text + header/notice)
        is reported once, keeping its strongest unit.

        ``score`` is the fraction of *distinct* reference tokens covered by some run
        of at least ``shingle`` tokens. Isolated common words do not contribute, so
        it reads lower than a general-diff coverage would, and means something
        sharper: how much of the reference literally appears in the query.
        """
        q_tokens = tokens(query)
        if len(q_tokens) < self.shingle:
            return []
        q_ids = self._query_ids(q_tokens)
        q_joined = f" {' '.join(q_tokens)} "

        qpos: dict[int, list[int]] = {}
        counts: dict[int, int] = {}
        for j, key in enumerate(self._keys(q_ids)):
            qpos.setdefault(key, []).append(j)
        for key in qpos:
            for uid in self._index.get(key, ()):
                counts[uid] = counts.get(uid, 0) + 1
        if not counts:
            return []
        # Shared-shingle count bounds the achievable run, so the tail cannot win.
        candidates = sorted(counts, key=counts.get, reverse=True)[:self.max_candidates]

        best: dict[str, SpanMatch] = {}
        for uid in candidates:
            name = self._unit_name[uid]
            if not self._has_required(name, q_joined):
                continue
            ref = self._unit_ids[uid]
            runs = self._runs(self._unit_keys[uid], qpos)
            if not runs:
                continue
            longest = max(size for _, _, size in runs)
            if longest < min_run:
                continue
            chained = _chain(runs)
            covered = sum(size for _, _, size in chained)
            start = min(sj for _, sj, _ in chained)
            end = max(sj + size for _, sj, size in chained)
            cand = SpanMatch(name, covered / len(ref), longest, covered, len(ref), start, end)
            prev = best.get(name)
            if prev is None or (cand.longest_run, cand.score) > (prev.longest_run, prev.score):
                best[name] = cand
        # Longest contiguous run first (distinctive phrase); ties broken by coverage
        # so the reference the query most fully fills (e.g. MIT over an MIT-superset
        # like Xnet/X11) wins over a looser superset match.
        results = sorted(best.values(), key=lambda m: (m.longest_run, m.score), reverse=True)
        return results[:top_k]


def _chain(runs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Select runs that overlap in neither the reference nor the query.

    Coverage must mean "how much of the reference is present in the query", which
    requires each query token to be spent once. Without that, a repetitive reference
    inflates: one query phrase matches at many reference offsets, each on its own
    diagonal, and summing them reports far more of the reference as present than is.

    Longest-first is the same greedy order ``difflib`` uses, so the selection stays
    close to the alignment this replaced. Run counts are small, so the quadratic
    overlap check costs nothing.
    """
    taken: list[tuple[int, int, int]] = []
    for run in sorted(runs, key=lambda r: -r[2]):
        si, sj, size = run
        if any(si < tsi + tsize and tsi < si + size or
               sj < tsj + tsize and tsj < sj + size
               for tsi, tsj, tsize in taken):
            continue
        taken.append(run)
    return taken
