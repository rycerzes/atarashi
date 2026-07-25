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
