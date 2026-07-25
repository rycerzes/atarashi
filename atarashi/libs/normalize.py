#!/usr/bin/env python3
"""SPDX-style text normalization and tokenization for license matching.

Normalization follows the spirit of the SPDX license matching guidelines: fold
case, neutralize punctuation/quote/bullet variation, and collapse whitespace, so
that cosmetically different renderings of the same license text produce the same
token sequence. Used by the exact-hash and sequence matchers.

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

import re

# Curly quotes / dashes -> ASCII so quotes and hyphens don't split matches.
_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})
_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return a case-folded, punctuation-stripped, whitespace-collapsed form."""
    text = text.translate(_TRANSLATE).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Normalized whitespace-delimited tokens of ``text``."""
    normalized = normalize(text)
    return normalized.split() if normalized else []
