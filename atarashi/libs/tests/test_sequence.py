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
REFS = [("MIT", MIT), ("BSD-3-Clause", BSD)]


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


AGPL = ("This program is free software you can redistribute it under the terms of the "
        "GNU Affero General Public License as published by the Free Software Foundation.")
GPL = ("This program is free software you can redistribute it under the terms of the "
       "GNU General Public License as published by the Free Software Foundation.")


def test_required_phrase_gates_out_wrong_variant():
    # both share most text; the required "affero" phrase separates them
    refs = [("AGPL-3.0", AGPL), ("GPL-3.0", GPL)]
    required = [("AGPL-3.0", ["affero general public license"])]
    m = LicenseMatcher(refs, required=required)
    # a GPL notice (no "affero") must not match AGPL even though coverage is high
    hits = m.match(GPL, min_score=0.5)
    names = [h.shortname for h in hits]
    assert "AGPL-3.0" not in names
    assert "GPL-3.0" in names


def test_required_phrase_present_allows_match():
    refs = [("AGPL-3.0", AGPL)]
    m = LicenseMatcher(refs, required=[("AGPL-3.0", ["affero general public license"])])
    assert m.match(AGPL, min_score=0.5)[0].shortname == "AGPL-3.0"


def test_multiple_units_per_license_header_matches():
    # a license carries both a full text and a short header/notice unit
    full = ("Apache License Version 2.0 January 2004 terms and conditions for use "
            "reproduction and distribution as defined by sections below")
    header = ("Licensed under the Apache License Version 2.0 you may not use this file "
              "except in compliance with the License")
    m = LicenseMatcher([("Apache-2.0", full), ("Apache-2.0", header)])
    doc = "# " + header + "\nimport os\n"
    hits = m.match(doc, min_score=0.8)
    # reported once, resolved via the header unit
    assert [h.shortname for h in hits] == ["Apache-2.0"]
    assert hits[0].score >= 0.9
