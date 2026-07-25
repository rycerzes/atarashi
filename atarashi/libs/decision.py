#!/usr/bin/env python3
"""Abstention: return UNKNOWN instead of a low-confidence guess.

A license scanner that always emits its best guess trades precision for recall on
inputs that carry no identifiable license (e.g. a bare copyright header). This
returns an explicit UNKNOWN when the strongest match does not clear a confidence
threshold, so callers can prefer precision. Assumes scores are in [0, 1] (the new
cascade agent's contract: SPDX/exact = 1.0, sequence coverage in [0, 1]).

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

from collections.abc import Sequence

UNKNOWN_SHORTNAME = "UNKNOWN"
DEFAULT_MIN_SCORE = 0.5


def unknown_result(top_score: float = 0.0) -> dict:
    return {
        "shortname": UNKNOWN_SHORTNAME,
        "sim_type": "Abstain",
        "sim_score": top_score,
        "description": "no confident license match",
    }


def apply_abstention(results: Sequence[dict], threshold: float = DEFAULT_MIN_SCORE) -> list[dict]:
    """Return ``results`` if the top match clears ``threshold``; else a single UNKNOWN.

    An empty result set abstains. The original ranking is preserved otherwise.
    """
    top = max((r.get("sim_score", 0.0) for r in results), default=0.0)
    if not results or top < threshold:
        return [unknown_result(top)]
    return list(results)
