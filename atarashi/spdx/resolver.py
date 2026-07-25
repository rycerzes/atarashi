#!/usr/bin/env python3
"""Resolve detected SPDX license ids to FOSSology license shortnames.

Bridges the SPDX detector to Atarashi's result contract: a resolved tag is a
high-precision match (``sim_score`` 1.0, ``sim_type`` ``SPDXIdentifier``). Ids that
do not correspond to a known shortname are dropped, so the caller abstains rather
than inventing a license.

SPDX-License-Identifier: GPL-2.0-only
"""
from __future__ import annotations

from collections.abc import Iterable

from atarashi.spdx.detector import SpdxMatch, detect


def _index(shortnames: Iterable[str]) -> dict[str, str]:
    return {s.lower(): s for s in shortnames if isinstance(s, str)}


def _lookup(license_id: str, index: dict[str, str]) -> str | None:
    key = license_id.lower()
    if key in index:
        return index[key]
    if key.endswith("+") and key[:-1] in index:  # `GPL-2.0+` -> `GPL-2.0`
        return index[key[:-1]]
    return None


def resolve(matches: Iterable[SpdxMatch], shortnames: Iterable[str]) -> list[dict]:
    """Map SPDX matches to result dicts for ids present in ``shortnames``."""
    index = _index(shortnames)
    results: list[dict] = []
    seen: set[str] = set()
    for match in matches:
        for license_id in match.licenses:
            shortname = _lookup(license_id, index)
            if shortname is None or shortname in seen:
                continue
            seen.add(shortname)
            results.append({
                "shortname": shortname,
                "sim_type": "SPDXIdentifier",
                "sim_score": 1.0,
                "description": "",
            })
    return results


def detect_and_resolve(data: str, shortnames: Iterable[str]) -> list[dict]:
    """Detect SPDX tags in ``data`` and resolve them against ``shortnames``."""
    return resolve(detect(data), shortnames)
