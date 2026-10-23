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
    Claim(
        id="buy-the-dip",
        question="Should I buy stocks right after they drop?",
        short_label="Buy the Dip",
        factor_key="st_reversal",
        match_level="related",
        what_fluke_tested="Once a month, Fluke buys last week's losers of the 148 big US companies and short-sells last week's winners at the same time.",
        what_it_does_not_test="Fluke short-sells, which most dip-buyers never do. The 148 stocks were picked in 2026, so companies that went bankrupt after a dip are missing, which makes dip-buying seem better than it is.",
        default_period_key="2020_2024",
    ),
    Claim(
        id="ride-the-winners",
        question="Should I buy stocks that have been going up?",
        short_label="Ride the Winners",
        factor_key="momentum",
        match_level="related",
        what_fluke_tested="Whether the stocks in the 148 big US companies that went up the most over the last 12 months (excluding the most recent month) continue to go up over the next month, and whether the stocks that performed the worst over the last 12 months continue to go down over the next month.",
        what_it_does_not_test="Fluke short-sells, which most winner-riders never do. The 148 stocks were picked in 2026, so companies that later failed are missing.",
        default_period_key="2015_2019",
    ),
    Claim(
        id="low-volatility",
        question="Should I buy stocks that don't move much?",
        short_label="Low Volatility",
        factor_key="realized_vol",
        match_level="related",
        what_fluke_tested="Once a month, buys the stocks with the lowest price swings over the past 3 months and short-sells the stocks with the highest price swings over the past 3 months.",
        what_it_does_not_test="The 2026 list of 148 big US companies includes many volatile companies like Nvidia that later soared, and excludes many volatile ones that failed, making shorting volatile stocks look worse.",
        default_period_key="2005_2024",
    ),
    Claim(
        id="small-illiquid-stocks",
        question="Will small, overlooked stocks outperform?",
        short_label="Small Illiquid Stocks",
        factor_key="amihud_illiquidity",
        match_level="related",
        what_fluke_tested="One per month, Fluke buys the stocks whose prices move the most per dollar traded (the hardest trade) and short-sells the easiest to trade.",
        what_it_does_not_test="Hard to trade companies that failed aren't in the list, which makes this factor look better than it is. Since the universe is made up of 148 of the biggest US companies, none of them are truly small or hard to trade.",
        default_period_key="2005_2009",
    ),
    Claim(
        id="back-to-average",
        question="Will stocks far from the past month's average price return to that price?",
        short_label="Back to Average",
        factor_key="price_to_ma",
        match_level="related",
        what_fluke_tested="Once a month, Fluke buys the stocks that are most stretched below the past month's average and sells the stocks most stretched above the past month's average.",
        what_it_does_not_test="The 148 stocks were picked in 2026, so companies that went bankrupt after a dip are missing. Fluke also short-sells, which most back-to-average investors never do.",
        default_period_key="2015_2019",
    ),
    Claim(
        id="follow-the-volume",
        question="Will the most traded stocks outperform the least traded stocks?",
        short_label="Follow the Volume",
        factor_key="dollar_volume",
        match_level="related",
        what_fluke_tested="Once a month, Fluke buys the stocks with the highest dollar volume and sells the stocks with the lowest dollar volume.",
        what_it_does_not_test="High trading volume isn't necessarily the same as hype. The 148 stocks were picked in 2026, so companies that went bankrupt after a dip are missing. Fluke also short-sells, which most volume-followers never do.",
        default_period_key="2005_2009",
    ),
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
