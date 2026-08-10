#!/usr/bin/env python3
"""Tests for the abstention / UNKNOWN decision.

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.libs.decision import UNKNOWN_SHORTNAME, apply_abstention


def _r(score, name="MIT"):
    return {"shortname": name, "sim_type": "x", "sim_score": score, "description": ""}


def test_confident_results_pass_through_unchanged():
    results = [_r(1.0), _r(0.3)]
    assert apply_abstention(results, threshold=0.5) == results


def test_all_weak_abstains():
    out = apply_abstention([_r(0.2), _r(0.1)], threshold=0.5)
    assert len(out) == 1 and out[0]["shortname"] == UNKNOWN_SHORTNAME


def test_empty_abstains():
    out = apply_abstention([], threshold=0.5)
    assert out[0]["shortname"] == UNKNOWN_SHORTNAME
    assert out[0]["sim_score"] == 0.0


def test_spdx_full_confidence_never_abstains():
    assert apply_abstention([_r(1.0)], threshold=0.5)[0]["shortname"] == "MIT"


def test_threshold_boundary_is_inclusive():
    assert apply_abstention([_r(0.5)], threshold=0.5)[0]["shortname"] == "MIT"


# --- span-match acceptance ---------------------------------------------------

from atarashi.libs.decision import (DEFAULT_MIN_COVERAGE, DEFAULT_STRONG_RUN,
                                    is_confident)


def test_long_run_accepted_even_at_negligible_coverage():
    """A verbatim notice covers ~1% of a long license body but identifies it."""
    assert is_confident(coverage=0.01, longest_run=DEFAULT_STRONG_RUN)


def test_high_coverage_accepted_even_with_short_runs():
    """A whole short license, matched in fragments, is still the license."""
    assert is_confident(coverage=DEFAULT_MIN_COVERAGE, longest_run=1)


def test_short_run_and_low_coverage_rejected():
    assert not is_confident(coverage=0.05, longest_run=10)


def test_thresholds_are_inclusive():
    assert is_confident(coverage=0.0, longest_run=DEFAULT_STRONG_RUN)
    assert is_confident(coverage=DEFAULT_MIN_COVERAGE, longest_run=0)


def test_thresholds_are_overridable():
    assert is_confident(0.05, 10, strong_run=10)
    assert is_confident(0.05, 1, min_coverage=0.05)
    assert not is_confident(0.4, 19, strong_run=20, min_coverage=0.5)
