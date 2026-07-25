#!/usr/bin/env python3
"""Tests for SPDX id -> shortname resolution and the upgraded spdx_identifer.

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.libs.initialmatch import spdx_identifer
from atarashi.spdx.resolver import detect_and_resolve, resolve
from atarashi.spdx.detector import detect

SHORTNAMES = ["MIT", "Apache-2.0", "GPL-2.0", "BSD-3-Clause", "LGPL-2.1"]


def test_resolve_exact():
    r = resolve(detect("SPDX-License-Identifier: MIT"), SHORTNAMES)
    assert r == [{"shortname": "MIT", "sim_type": "SPDXIdentifier",
                  "sim_score": 1.0, "description": ""}]


def test_resolve_case_insensitive_to_canonical_shortname():
    # detected id differs in case; result uses the canonical shortname casing
    assert resolve(detect("SPDX-License-Identifier: apache-2.0"), SHORTNAMES)[0]["shortname"] == "Apache-2.0"


def test_resolve_or_later_plus_falls_back():
    assert resolve(detect("SPDX-License-Identifier: GPL-2.0+"), SHORTNAMES)[0]["shortname"] == "GPL-2.0"


def test_unknown_id_dropped():
    assert resolve(detect("SPDX-License-Identifier: Nonexistent-9.9"), SHORTNAMES) == []


def test_compound_resolves_known_drops_unknown():
    out = resolve(detect("SPDX-License-Identifier: MIT OR Nonexistent-9.9"), SHORTNAMES)
    assert [d["shortname"] for d in out] == ["MIT"]


def test_dedupes_repeated_license():
    text = "SPDX-License-Identifier: MIT\nSPDX-License-Identifier: MIT"
    assert len(detect_and_resolve(text, SHORTNAMES)) == 1


def test_spdx_identifer_backward_compatible_schema():
    out = spdx_identifer("// SPDX-License-Identifier: BSD-3-Clause", SHORTNAMES)
    assert out == [{"shortname": "BSD-3-Clause", "sim_type": "SPDXIdentifier",
                    "sim_score": 1.0, "description": ""}]


def test_spdx_identifer_ignores_bare_license_line():
    # regression: the legacy scan matched "license:" lines; the tag path must not
    assert spdx_identifer("license: MIT is used here", SHORTNAMES) == []
