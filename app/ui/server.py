from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.api import (
    ResultNotComputedError,
    get_analysis,
    list_factors,
    list_periods,
    verdict_for,
)


# Creates the FastAPI app.
# This is the bridge between the React frontend and Michael's Python API.
app = FastAPI(
    title="QFRP Web API",
    version="2.0.0",
)


# These are the React addresses used while we are developing the app locally.
# They need permission to request data from the FastAPI server.
local_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


# Later, when the React frontend is deployed, its URL can be stored
# in FRONTEND_ORIGIN instead of being hard-coded here.
frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "",
).strip().rstrip("/")


# Start with the local development addresses.
allowed_origins = local_origins.copy()


# If a deployed frontend URL exists, allow that website too.
if frontend_origin:
    allowed_origins.append(frontend_origin)


# CORS allows the React frontend to communicate with this API
# even though React and FastAPI run on different addresses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# This describes the information React sends when it asks
# the backend for a verdict.
class VerdictRequest(BaseModel):
    factor_key: str
    period_key: str

    # This is how many unique factor and period combinations
    # the user has viewed during the current session.
    n_configs_viewed: int = Field(ge=1)

    # Python treats True and False like numbers in some cases.
    # The contract says booleans should not count as valid values for k,
    # so this makes sure they are rejected.
    @field_validator(
        "n_configs_viewed",
        mode="before",
    )
    @classmethod
    def reject_boolean_count(
        cls,
        value,
    ):
        if isinstance(value, bool):
            raise ValueError(
                "n_configs_viewed must be an integer of at least 1."
            )

        return value


# This helper converts the pandas equity curve into normal dictionaries.
# React cannot directly read a pandas DataFrame.
def equity_curve_to_list(result):
    return [
        {
            "date": index.date().isoformat(),
            "strategy": float(row["strategy"]),
        }
        for index, row in result.equity_curve.iterrows()
    ]


# Michael's API returns Python objects like dataclasses and DataFrames.
# This function changes the AnalysisResult into normal data that
# FastAPI can send to React as JSON.
def analysis_to_dict(result):
    equity_curve = equity_curve_to_list(result)

    return {
        # Basic information about the selected test.
        "factor_key": result.factor_key,
        "factor_label": result.factor_label,
        "period_label": result.period_label,
        "start_date": result.start_date,
        "end_date": result.end_date,

        # Performance after normal 1x trading costs.
        "sharpe_1x": result.sharpe_1x,
        "total_return": result.total_return,
        "annualized_vol": result.annualized_vol,
        "max_drawdown": result.max_drawdown,
        "hit_rate": result.hit_rate,
        "turnover": result.turnover,

        # The strategy's performance over time.
        # The old benchmark column was removed in contract v2.
        "equity_curve": equity_curve,

        # Gross Sharpe is used for the statistical significance test.
        # This must be compared with null_sharpes_gross.
        "sharpe_gross": result.sharpe_gross,
        "null_sharpes_gross": result.null_sharpes_gross,

        # Information about the 2,000 shuffled backtests.
        "n_permutations": result.n_permutations,
        "n_dropped": result.n_dropped,

        # Raw and adjusted significance values.
        "p_value": result.p_value,
        "adjusted_p_value": result.adjusted_p_value,
        "k": result.k,

        # These test whether the strategy performed significantly
        # worse than the shuffled versions.
        "reverse_p_value": result.reverse_p_value,
        "adjusted_reverse_p_value":
            result.adjusted_reverse_p_value,

        # This lets the app compare how much the shuffled
        # portfolios traded with the real strategy.
        "null_turnover": result.null_turnover,

        # Sharpe values when trading costs are tested at 1x, 2x, and 3x.
        "cost_stress": [
            asdict(point)
            for point in result.cost_stress
        ],

        # Shows how many stocks had usable factor scores each year.
        "names_per_year": result.names_per_year,

        # These strings come directly from Michael's API.
        # The contract says they should be displayed without rewriting them.
        "limitations": result.limitations,
    }


# This handles errors returned by get_analysis().
# The contract gives different HTTP codes for different problems.
def analysis_error_to_http(exc):
    # ResultNotComputedError is a subclass of KeyError.
    # Checking its exact class name lets us identify it before
    # treating other KeyErrors as normal bad factor or period keys.
    if isinstance(exc, ResultNotComputedError):
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    # A normal KeyError means the factor or period does not exist,
    # or the factor is intentionally excluded.
    if isinstance(exc, KeyError):
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    # The contract says an invalid value of k should return 422.
    if isinstance(
        exc,
        (ValueError, TypeError),
    ):
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Anything unexpected should still be raised so we can see
    # the real problem instead of hiding a programming bug.
    raise exc


# A very small test endpoint.
# If this returns {"status": "ok"}, the FastAPI server is running.
@app.get("/api/health")
def health():
    return {
        "status": "ok",
    }


# Sends the 10 available factor choices to React.
# React uses these to build the factor dropdown instead of
# hard-coding the factor names in the frontend.
@app.get("/api/factors")
def factors():
    return [
        asdict(factor)
        for factor in list_factors()
    ]


# Sends the 5 available time periods to React.
# React uses these values to build the date-range dropdown.
@app.get("/api/periods")
def periods():
    return [
        asdict(period)
        for period in list_periods()
    ]


# This is the main analysis endpoint.
# React sends the chosen factor, period, and session counter.
@app.get("/api/analysis")
def analysis(
    factor_key: str,
    period_key: str,

    # The counter must always be at least 1.
    # FastAPI automatically returns a 422 error if it is below 1.
    n_configs_viewed: int = Query(
        ...,
        ge=1,
    ),
):
    try:
        # Michael's API calls this value "k".
        # It is the number of unique configurations the user has viewed.
        result = get_analysis(
            factor_key,
            period_key,
            k=n_configs_viewed,
        )

    except (
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        analysis_error_to_http(exc)

    # Convert the Python result into JSON-friendly data
    # before sending it back to React.
    return analysis_to_dict(result)


# React calls this endpoint after loading an analysis.
# It asks Michael's backend to create the plain-English verdict.
@app.post("/api/verdict")
def verdict(
    request: VerdictRequest,
):
    try:
        # Get the exact same analysis again using the same session count.
        # This keeps the adjusted p-value in the analysis and verdict matched.
        result = get_analysis(
            request.factor_key,
            request.period_key,
            k=request.n_configs_viewed,
        )

        # Michael's verdict function decides whether the result is
        # PASS, FAIL, or INCONCLUSIVE.
        result_verdict = verdict_for(
            result,
            request.n_configs_viewed,
        )

    except (
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        analysis_error_to_http(exc)

    # The Verdict is a dataclass, so asdict converts it into
    # normal JSON-friendly data for React.
    #
    # This also includes the new reversed field, which lets
    # the frontend style a "worse than random" FAIL differently.
    return asdict(
        result_verdict
    )