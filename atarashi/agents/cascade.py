#!/usr/bin/env python3
"""Cascade agent: SPDX-tag detection first, else abstain (Phase 0).

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.agents.atarashiAgent import AtarashiAgent
from atarashi.libs.decision import DEFAULT_MIN_SCORE, apply_abstention
from atarashi.spdx.resolver import detect_and_resolve


class Cascade(AtarashiAgent):
    """License identification cascade.

    Phase 0: resolve an author-declared ``SPDX-License-Identifier`` (the highest-
    precision signal in real source files); if none maps to a known license,
    abstain (UNKNOWN) rather than emit a low-confidence guess. Later phases insert
    exact-hash and sequence matching before the abstention step.
    """

    def __init__(self, licenseList, verbose=0, threshold=DEFAULT_MIN_SCORE):
        super().__init__(licenseList, verbose)
        self.threshold = threshold

    def scan(self, filePath):
        """Scan ``filePath`` and return result dicts (SPDX matches, or one UNKNOWN)."""
        with open(filePath, errors="replace") as in_file:
            raw = in_file.read()
        spdx = detect_and_resolve(raw, self.licenseList["shortname"])
        if spdx:
            return spdx
        return apply_abstention([], self.threshold)
