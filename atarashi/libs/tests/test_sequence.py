#!/usr/bin/env python3
"""Tests for the token-sequence license matcher.

SPDX-License-Identifier: GPL-2.0-only
"""
from atarashi.libs.sequence import LicenseMatcher

MIT = ("Permission is hereby granted, free of charge, to any person obtaining a copy "
       "of this software and associated documentation files, to deal in the Software "
       "without restriction.")
BSD = ("Redistribution and use in source and binary forms, with or without "
       "modification, are permitted provided that the following conditions are met.")
REFS = {"MIT": MIT, "BSD-3-Clause": BSD}


def test_exact_full_text():
    m = LicenseMatcher(REFS)
    assert m.exact(MIT) == "MIT"
    assert m.exact("  PERMISSION is hereby granted, free of charge...") is None


def test_embedded_notice_matches_with_span():
    m = LicenseMatcher(REFS)
    doc = "/*\n * Copyright 2020 Acme\n * " + MIT + "\n */\nint main(){}"
    hits = m.match(doc, min_score=0.8)
    assert hits and hits[0].shortname == "MIT"
    assert hits[0].score >= 0.9
    # matched span points into the query token stream
    assert 0 <= hits[0].start < hits[0].end


def test_picks_higher_coverage_reference():
    m = LicenseMatcher(REFS)
    assert m.match(BSD, min_score=0.8)[0].shortname == "BSD-3-Clause"


def test_no_license_text_returns_empty():
    m = LicenseMatcher(REFS)
    assert m.match("int main() { return 0; } // just code, no license", min_score=0.5) == []


def test_partial_presence_scores_below_full():
    m = LicenseMatcher(REFS)
    half = "Permission is hereby granted, free of charge, to any person"
    hits = m.match(half, min_score=0.1)
    assert hits and hits[0].shortname == "MIT"
    assert hits[0].score < 1.0


def test_min_score_filters_weak_matches():
    m = LicenseMatcher(REFS)
    half = "Permission is hereby granted, free of charge, to any person"
    assert m.match(half, min_score=0.95) == []
