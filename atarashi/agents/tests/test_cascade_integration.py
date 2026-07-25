#!/usr/bin/env python3
"""End-to-end cascade checks against the shipped 382-license list.

Exercises the real reference data (not synthetic texts): SPDX tags, exact full
text, embedded license text, and abstention. The shipped ``licenseList.csv`` has
no header column, so short notices (e.g. the Apache header) correctly abstain
until the notice layer is populated by ``preprocess``.

SPDX-License-Identifier: GPL-2.0-only
"""
import os
import tempfile

import pandas as pd
import pytest

from atarashi.agents.cascade import Cascade

CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "licenses", "licenseList.csv")


@pytest.fixture(scope="module")
def agent():
    if not os.path.exists(CSV):
        pytest.skip("licenseList.csv not present")
    df = pd.read_csv(CSV).fillna("").rename(columns={"text": "processed_text"})
    df = df[["shortname", "processed_text"]]
    if "MIT" not in set(df["shortname"]):
        pytest.skip("MIT not in shipped license list")
    return Cascade(df), df


def _scan(agent, content):
    scanner, _ = agent
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, content.encode())
    os.close(fd)
    try:
        return scanner.scan(path)
    finally:
        os.remove(path)


def test_spdx_tag_on_real_list(agent):
    top = _scan(agent, "# SPDX-License-Identifier: MIT\nprint(1)\n")[0]
    assert top["shortname"] == "MIT"
    assert top["sim_type"] == "SPDXIdentifier"


def test_exact_full_text_on_real_list(agent):
    _, df = agent
    mit = df.loc[df["shortname"] == "MIT", "processed_text"].iloc[0]
    assert _scan(agent, mit)[0]["shortname"] == "MIT"


def test_embedded_license_text_on_real_list(agent):
    _, df = agent
    mit = df.loc[df["shortname"] == "MIT", "processed_text"].iloc[0]
    top = _scan(agent, "/* Copyright 2020 Acme\n" + mit + "\n*/\nint main(){}\n")[0]
    assert top["shortname"] == "MIT"
    assert top["sim_type"] == "SequenceCoverage"


def test_no_license_abstains_on_real_list(agent):
    assert _scan(agent, "# Copyright 2020 Acme Corp\nimport os\n")[0]["shortname"] == "UNKNOWN"
