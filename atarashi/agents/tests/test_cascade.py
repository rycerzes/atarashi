#!/usr/bin/env python3
"""Tests for the Phase-0 cascade agent (SPDX-first, else UNKNOWN).

SPDX-License-Identifier: GPL-2.0-only
"""
import pandas as pd

from atarashi.agents.cascade import Cascade

LICENSES = pd.DataFrame(
    {"shortname": ["MIT", "Apache-2.0", "GPL-2.0"], "processed_text": ["", "", ""]}
)


def _scan(tmp_path, content):
    f = tmp_path / "src.py"
    f.write_text(content)
    return Cascade(LICENSES).scan(str(f))


def test_spdx_tag_resolved(tmp_path):
    out = _scan(tmp_path, "# SPDX-License-Identifier: MIT\nprint(1)\n")
    assert out[0]["shortname"] == "MIT"
    assert out[0]["sim_type"] == "SPDXIdentifier"
    assert out[0]["sim_score"] == 1.0


def test_no_license_abstains(tmp_path):
    out = _scan(tmp_path, "# Copyright 2020 ACME Corp\nprint(1)\n")
    assert out[0]["shortname"] == "UNKNOWN"


def test_unknown_spdx_id_abstains(tmp_path):
    out = _scan(tmp_path, "# SPDX-License-Identifier: Nonexistent-9.9\n")
    assert out[0]["shortname"] == "UNKNOWN"
