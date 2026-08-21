#!/usr/bin/env python3
"""Learned candidate scoring, replacing the hand-ordered ranking key.

The cascade retrieves well and orders badly: on an independent corpus, 62 of 77 wrong
answers already held the correct license inside the top five candidates, 28 of them at
rank two. Hand-tuning that ordering was attempted five separate ways and every attempt
lost more than it gained, which is what motivated learning the combination instead.

The model scores each candidate from the match evidence alone — run lengths, coverage,
gap structure, distance to the leader. It never reads license text. Models that do are
demonstrably unsafe here: on high-overlap/different-meaning pairs they score under 40%
and below chance when overlap disagrees with the label, which is exactly the
GPL-2-vs-GPL-3 case.

Scope, measured: this helps for licenses the model has training data for and does
nothing for ones it has not seen (leave-one-family-out is -0.012). It is therefore
optional and **off unless an artifact is present**, and the engine falls back to the
hand-tuned key whenever the artifact is missing or unreadable.

``featurize`` is imported by the training code so the two cannot drift; a mismatch
between training and inference features would be silent and would poison the scores.

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

import math
from pathlib import Path

DEFAULT_RANKER = Path(__file__).resolve().parents[1] / "data" / "ranker.joblib"

# Fixed order — the model's columns are positional.
FEATURE_NAMES = [
    "longest_run", "log_ref_tokens", "ref_coverage", "query_coverage",
    "matched_tokens", "shingle_ratio", "run_count",
    "ref_gap", "query_gap", "ref_head", "ref_tail", "substitution",
    "required_ok", "run_margin", "cov_margin",
]


def featurize(matches) -> list[list[float]]:
    """Feature rows for one query's ranked candidates.

    ``run_margin`` and ``cov_margin`` are distances to the leader. They are the
    signal a per-candidate confidence bar can never see, and they turn out to be the
    strongest features the model has — the rank *position* itself is deliberately not
    a feature, because a model given it simply reproduces the ordering it was meant
    to improve.
    """
    if not matches:
        return []
    lead = matches[0]
    return [[
        float(m.longest_run),
        math.log1p(m.ref_tokens),
        float(m.score),
        float(m.query_coverage),
        float(m.matched_tokens),
        float(m.shingle_ratio),
        float(m.run_count),
        float(m.ref_gap),
        float(m.query_gap),
        float(m.ref_head),
        float(m.ref_tail),
        float(m.substitution),
        1.0 if m.required_ok else 0.0,
        float(lead.longest_run - m.longest_run),
        float(lead.score - m.score),
    ] for m in matches]


# The artifact is immutable for the life of a process and every agent construction
# would otherwise re-read it from disk — measurably slow, and noisy besides.
_CACHE: dict[str, object] = {}


def load_ranker(path: Path | None = None):
    """The trained scorer, or None when absent — the caller keeps its own ordering."""
    target = Path(path or DEFAULT_RANKER)
    key = str(target)
    if key in _CACHE:
        return _CACHE[key]
    bundle = None
    if target.exists():
        try:
            import warnings

            import joblib

            with warnings.catch_warnings():
                # joblib's unpickler trips a NumPy 2.5 deprecation once per stored
                # array — hundreds of lines on first scan, from a dependency, about
                # nothing the caller can act on.
                warnings.simplefilter("ignore", DeprecationWarning)
                loaded = joblib.load(target)
            if {"model", "scaler"} <= set(loaded):
                bundle = loaded
        except Exception:
            bundle = None
    _CACHE[key] = bundle
    return bundle


def rerank(matches, bundle):
    """Re-order ``matches`` by learned score, most likely first.

    Returns the input untouched on any failure. A ranking model that raises must not
    take the scan down with it — the hand-tuned order is a working fallback.
    """
    if not bundle or len(matches) < 2:
        return matches
    try:
        rows = featurize(matches)
        scores = bundle["model"].predict_proba(bundle["scaler"].transform(rows))[:, 1]
    except Exception:
        return matches
    return [m for _, m in sorted(zip(scores, matches),
                                 key=lambda pair: -pair[0])]
