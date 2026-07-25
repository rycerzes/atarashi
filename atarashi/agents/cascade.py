#!/usr/bin/env python3
"""Cascade agent: SPDX-tag detection first, else abstain (Phase 0).

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.agents.atarashiAgent import AtarashiAgent
from atarashi.libs.decision import apply_abstention
from atarashi.libs.sequence import DEFAULT_MIN_RUN, LicenseMatcher
from atarashi.spdx.resolver import detect_and_resolve


class Cascade(AtarashiAgent):
    """License identification cascade, precision first.

    Stages, each handling what the cheaper one could not, then abstaining:
      1. author-declared ``SPDX-License-Identifier`` (highest precision);
      2. exact normalized full-text match (input *is* a known license);
      3. token-sequence coverage match (license text embedded in the input);
      4. UNKNOWN — no confident match, rather than a low-confidence guess.
    """

    def __init__(self, licenseList, verbose=0, min_run=DEFAULT_MIN_RUN):
        super().__init__(licenseList, verbose)
        self.min_run = min_run
        self.matcher = LicenseMatcher(self._reference_units())

    def _reference_units(self):
        """Full license text plus header/notice text as separate matchable units."""
        has_header = "processed_header" in self.licenseList.columns
        for _, row in self.licenseList.iterrows():
            name = str(row["shortname"])
            yield (name, str(row["processed_text"]))
            if has_header:
                header = row["processed_header"]
                if isinstance(header, str) and header.strip():
                    yield (name, header)

    def scan(self, filePath):
        """Scan ``filePath`` and return ranked result dicts (or one UNKNOWN)."""
        with open(filePath, errors="replace") as in_file:
            raw = in_file.read()

        spdx = detect_and_resolve(raw, self.licenseList["shortname"])
        if spdx:
            return spdx

        exact = self.matcher.exact(raw)
        if exact:
            return [{"shortname": exact, "sim_type": "ExactFullText",
                     "sim_score": 1.0, "description": ""}]

        hits = self.matcher.match(raw, min_run=self.min_run)
        if hits:
            return [{"shortname": h.shortname, "sim_type": "SequenceCoverage",
                     "sim_score": round(h.score, 4),
                     "description": f"matched tokens {h.start}:{h.end}"} for h in hits]

        return apply_abstention([])
