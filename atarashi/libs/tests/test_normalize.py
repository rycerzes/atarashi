#!/usr/bin/env python3
"""Tests for SPDX-style normalization.

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.libs.normalize import normalize, tokens


def test_case_and_whitespace_collapsed():
    assert normalize("Permission   is\nHEREBY  granted") == "permission is hereby granted"


def test_punctuation_neutralized():
    assert normalize('"MIT", (the License);') == "mit the license"


def test_curly_quotes_and_dashes_folded():
    # smart quotes and en/em dashes must not create distinct tokens
    assert normalize("“BSD” — style") == normalize('"BSD" - style')


def test_tokens_split():
    assert tokens("Apache License, Version 2.0") == ["apache", "license", "version", "2", "0"]


def test_empty_is_empty():
    assert tokens("   \n  ") == []
    assert normalize("") == ""
