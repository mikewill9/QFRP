from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api import get_analysis, list_factors, list_periods, verdict_for

#This next part is creating the FastAPI app that will act as the bridge
#It bridgges the React Frontend framework to the existing Python backend the MW created
app = FastAPI(
    title="QFRP Web API",
    version="1.0.0",
)


# Local React development addresses.
#These are the frontend addresses used while developing locally.
local_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Later, Render will read the deployed Vercel address
# from an environment variable.
frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "",
).strip().rstrip("/")

allowed_origins = local_origins.copy()

if frontend_origin:
    allowed_origins.append(frontend_origin)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class VerdictRequest(BaseModel):
    factor_key: str
    period_key: str
    n_configs_viewed: int = Field(ge=1)


def analysis_to_dict(result):
    equity_curve = [
        {
            "date": index.date().isoformat(),
            "strategy": float(row["strategy"]),
            "benchmark": float(row["benchmark"]),
        }
        for index, row in result.equity_curve.iterrows()
    ]

    return {
        "factor_key": result.factor_key,
        "factor_label": result.factor_label,
        "period_label": result.period_label,
        "start_date": result.start_date,
        "end_date": result.end_date,

        "sharpe": result.sharpe,
        "total_return": result.total_return,
        "annualized_vol": result.annualized_vol,
        "max_drawdown": result.max_drawdown,
        "hit_rate": result.hit_rate,
        "n_trades": result.n_trades,

        "equity_curve": equity_curve,

        "p_value": result.p_value,
        "null_sharpes": result.null_sharpes,
        "n_permutations": result.n_permutations,

        "cost_stress": [
            asdict(point)
            for point in result.cost_stress
        ],

        "limitations": result.limitations,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/api/factors")
def factors():
    return [
        asdict(factor)
        for factor in list_factors()
    ]


@app.get("/api/periods")
def periods():
    return [
        asdict(period)
        for period in list_periods()
    ]


@app.get("/api/analysis")
def analysis(
    factor_key: str,
    period_key: str,
):
    try:
        result = get_analysis(
            factor_key,
            period_key,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return analysis_to_dict(result)


@app.post("/api/verdict")
def verdict(request: VerdictRequest):
    try:
        result = get_analysis(
            request.factor_key,
            request.period_key,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    result_verdict = verdict_for(
        result,
        request.n_configs_viewed,
    )
#Convert the cerdict dataclass into a regular dictionary
#so React can display the label, p-values, and explanation
    return asdict(result_verdict)