"""Structural tests for app.claims.list_claims().

They check Michael's finished CLAIMS, so they cannot pass while the
placeholder example claim is still in place: ``written_claims`` raises
NotImplementedError until every ``[MICHAEL`` string is gone. With
xfail(strict) the module flips to XPASS(strict) failures the moment the
claims are written, which is the signal to delete the ``pytestmark`` line.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from app.api import list_factors, list_periods
from app.claims import MATCH_LEVELS, Claim, list_claims

pytestmark = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=">>> MICHAEL: remove once CLAIMS is written")


def claim_strings(claim: Claim) -> list[str]:
    return [getattr(claim, field.name) for field in fields(Claim)]


def written_claims() -> list[Claim]:
    claims = list_claims()
    if any(s.startswith("[MICHAEL") for c in claims for s in claim_strings(c)):
        raise NotImplementedError("MICHAEL: CLAIMS still contains the placeholder example")
    return claims


def test_no_placeholder_text_remains():
    for claim in written_claims():
        for text in claim_strings(claim):
            assert not text.startswith("[MICHAEL"), f"{claim.id}: {text!r}"


def test_at_least_four_claims():
    assert len(written_claims()) >= 4


def test_ids_unique_and_url_safe():
    ids = [claim.id for claim in written_claims()]
    assert len(ids) == len(set(ids))
    for claim_id in ids:
        assert claim_id == claim_id.strip().lower()
        assert all(ch.isalnum() or ch in "-_" for ch in claim_id), claim_id


def test_factor_keys_are_in_the_menu():
    valid = {factor.key for factor in list_factors()}
    for claim in written_claims():
        assert claim.factor_key in valid, claim.id


def test_period_keys_are_in_the_menu():
    valid = {period.key for period in list_periods()}
    for claim in written_claims():
        assert claim.default_period_key in valid, claim.id


def test_match_levels_are_valid():
    for claim in written_claims():
        assert claim.match_level in MATCH_LEVELS, claim.id
