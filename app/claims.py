"""Claims a young investor might see online, mapped to the factor Fluke tested.

Content (every string, every mapping, every match level) is Michael's
research judgment. This module only validates the structure.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from app.api import list_factors, list_periods

MATCH_LEVELS = ("direct", "related")


@dataclass(frozen=True)
class Claim:
    id: str                      # url-safe slug, unique
    question: str                # the claim as a person would say it
    short_label: str             # 2-4 words for a card title
    factor_key: str              # must be a key from list_factors()
    match_level: str             # "direct" or "related"
    what_fluke_tested: str       # the actual strategy definition, plain words
    what_it_does_not_test: str   # how the claim differs from the test
    default_period_key: str      # the period the guided flow opens on


CLAIMS: tuple[Claim, ...] = (
    # >>> MICHAEL: write 5 to 8 claims. Keep the structure of this example,
    # replace every bracketed string. Pick each default_period_key so that at
    # least one claim opens on a convincing-looking equity curve that FAILs.
    Claim(
        id="example-claim",
        question="[MICHAEL: the claim, as someone online would phrase it]",
        short_label="[MICHAEL: 2-4 words]",
        factor_key="momentum",
        match_level="related",
        what_fluke_tested="[MICHAEL: what the factor actually does, plain words]",
        what_it_does_not_test="[MICHAEL: how the online claim differs from this test]",
        default_period_key="2005_2009",
    ),
    # <<< MICHAEL
)


def _check(condition: bool, claim: Claim, problem: str) -> None:
    if not condition:
        raise RuntimeError(f"app/claims.py: claim {claim.id!r} {problem}")


def list_claims() -> list[Claim]:
    """The claim menu, validated against the factor and period menus.

    Raises RuntimeError naming the first bad claim if an id is repeated or
    empty, a field is not a non-empty string, ``factor_key`` is not in
    ``list_factors()``, ``default_period_key`` is not in ``list_periods()``,
    or ``match_level`` is not one of ``MATCH_LEVELS``.
    """
    factor_keys = {factor.key for factor in list_factors()}
    period_keys = {period.key for period in list_periods()}
    seen_ids: set[str] = set()

    for claim in CLAIMS:
        for field in fields(Claim):
            value = getattr(claim, field.name)
            _check(
                isinstance(value, str) and value.strip() != "",
                claim,
                f"field {field.name!r} must be a non-empty string, got {value!r}",
            )
        _check(claim.id not in seen_ids, claim, "has a duplicate id")
        seen_ids.add(claim.id)
        _check(
            claim.factor_key in factor_keys,
            claim,
            f"has unknown factor_key {claim.factor_key!r}; valid: {sorted(factor_keys)}",
        )
        _check(
            claim.default_period_key in period_keys,
            claim,
            f"has unknown default_period_key {claim.default_period_key!r}; "
            f"valid: {sorted(period_keys)}",
        )
        _check(
            claim.match_level in MATCH_LEVELS,
            claim,
            f"has match_level {claim.match_level!r}; valid: {MATCH_LEVELS}",
        )

    return list(CLAIMS)
