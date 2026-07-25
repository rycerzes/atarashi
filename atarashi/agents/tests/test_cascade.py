#!/usr/bin/env python3
"""Tests for the Phase-0 cascade agent (SPDX-first, else UNKNOWN).

SPDX-License-Identifier: GPL-2.0-only
"""
import pandas as pd

from atarashi.agents.cascade import Cascade

MIT_TEXT = ("Permission is hereby granted, free of charge, to any person obtaining a "
            "copy of this software and associated documentation files, to deal in the "
            "Software without restriction.")
APACHE_HEADER = ("Licensed under the Apache License Version 2.0 you may not use this "
                 "file except in compliance with the License you may obtain a copy")
LICENSES = pd.DataFrame({
    "shortname": ["MIT", "Apache-2.0", "GPL-2.0"],
    "processed_text": [MIT_TEXT, "Apache License Version 2.0 full body text here", ""],
    "processed_header": ["", APACHE_HEADER, ""],
})


def _scan(tmp_path, content):
    f = tmp_path / "src.py"
    f.write_text(content)
    return Cascade(LICENSES).scan(str(f))


def test_spdx_tag_resolved(tmp_path):
    out = _scan(tmp_path, "# SPDX-License-Identifier: MIT\nprint(1)\n")
    assert out[0]["shortname"] == "MIT"
    assert out[0]["sim_type"] == "SPDXIdentifier"
    assert out[0]["sim_score"] == 1.0


def test_exact_full_text(tmp_path):
    out = _scan(tmp_path, MIT_TEXT)
    assert out[0]["shortname"] == "MIT"
    assert out[0]["sim_type"] == "ExactFullText"


def test_embedded_license_text_sequence_match(tmp_path):
    out = _scan(tmp_path, "/*\n * Copyright 2020 Acme\n * " + MIT_TEXT + "\n */\nint main(){}")
    assert out[0]["shortname"] == "MIT"
    assert out[0]["sim_type"] == "SequenceCoverage"
    assert out[0]["sim_score"] >= 0.9


def test_no_license_abstains(tmp_path):
    out = _scan(tmp_path, "# Copyright 2020 ACME Corp\nprint(1)\n")
    assert out[0]["shortname"] == "UNKNOWN"


def test_unknown_spdx_id_abstains(tmp_path):
    out = _scan(tmp_path, "# SPDX-License-Identifier: Nonexistent-9.9\n")
    assert out[0]["shortname"] == "UNKNOWN"


def test_notice_header_matches_via_header_unit(tmp_path):
    out = _scan(tmp_path, "# " + APACHE_HEADER + "\nimport os\n")
    assert out[0]["shortname"] == "Apache-2.0"
    assert out[0]["sim_type"] == "SequenceCoverage"
