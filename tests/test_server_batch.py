"""HTTP tests for /api/claims and /api/batch that do not depend on Michael's work.

These pass today and keep passing after get_batch_analysis and CLAIMS are
written: the 501 case forces NotImplementedError through a monkeypatch so
it keeps testing the mapping, not the current state of the stub.
"""

from __future__ import annotations

import pytest
from app.ui import server
from fastapi.testclient import TestClient

client = TestClient(server.app)

VALID_BODY = {
    "configs": [{"factor_key": "momentum", "period_key": "2020_2024"}],
    "n_configs_viewed": 1,
}


def test_claims_returns_a_list_of_claim_dicts():
    response = client.get("/api/claims")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body
    expected_keys = {
        "id",
        "question",
        "short_label",
        "factor_key",
        "match_level",
        "what_fluke_tested",
        "what_it_does_not_test",
        "default_period_key",
    }
    for claim in body:
        assert set(claim) == expected_keys


def test_batch_rejects_empty_configs():
    response = client.post("/api/batch", json={"configs": [], "n_configs_viewed": 1})
    assert response.status_code == 422


def test_batch_rejects_more_than_ten_configs():
    configs = [{"factor_key": "momentum", "period_key": "2020_2024"}] * 11
    response = client.post("/api/batch", json={"configs": configs, "n_configs_viewed": 11})
    assert response.status_code == 422


def test_batch_rejects_missing_or_zero_count():
    response = client.post("/api/batch", json={"configs": VALID_BODY["configs"]})
    assert response.status_code == 422
    response = client.post(
        "/api/batch", json={"configs": VALID_BODY["configs"], "n_configs_viewed": 0}
    )
    assert response.status_code == 422


def test_batch_rejects_boolean_count():
    response = client.post(
        "/api/batch", json={"configs": VALID_BODY["configs"], "n_configs_viewed": True}
    )
    assert response.status_code == 422


def test_batch_returns_501_while_unimplemented(monkeypatch):
    def not_done(configs, k):
        raise NotImplementedError("MICHAEL: get_batch_analysis")

    monkeypatch.setattr(server, "get_batch_analysis", not_done)
    response = client.post("/api/batch", json=VALID_BODY)
    assert response.status_code == 501
    assert response.json()["detail"] == "get_batch_analysis is not implemented yet (Michael)"


def test_batch_maps_analysis_errors(monkeypatch):
    def unknown(configs, k):
        raise KeyError("unknown factor_key 'nope'")

    monkeypatch.setattr(server, "get_batch_analysis", unknown)
    response = client.post("/api/batch", json=VALID_BODY)
    assert response.status_code == 404
    # The message itself, not str(KeyError) with an extra layer of quotes.
    assert response.json()["detail"] == "unknown factor_key 'nope'"

    def too_few(configs, k):
        raise ValueError("k must be >= the number of unique configs")

    monkeypatch.setattr(server, "get_batch_analysis", too_few)
    assert client.post("/api/batch", json=VALID_BODY).status_code == 422


def test_batch_response_shape(monkeypatch):
    """The route's own work: thresholds, k, and one result per unique config."""
    from app.api import FAIL_ABOVE, PASS_BELOW

    def fake_batch(configs, k):
        return [
            (server.get_analysis(factor, period, k), server.verdict_for(server.get_analysis(factor, period, k), k))
            for factor, period in configs
        ]

    monkeypatch.setattr(server, "get_batch_analysis", fake_batch)
    body = {
        "configs": [
            {"factor_key": "momentum", "period_key": "2020_2024"},
            {"factor_key": "momentum", "period_key": "2020_2024"},  # duplicate
            {"factor_key": "st_reversal", "period_key": "2020_2024"},
        ],
        "n_configs_viewed": 2,
    }
    response = client.post("/api/batch", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["k"] == 2
    assert data["pass_below"] == PASS_BELOW
    assert data["fail_above"] == FAIL_ABOVE
    assert [(r["factor_key"], r["period_key"]) for r in data["results"]] == [
        ("momentum", "2020_2024"),
        ("st_reversal", "2020_2024"),
    ]
    for item in data["results"]:
        assert item["analysis"]["k"] == 2
        assert item["verdict"]["n_configs_viewed"] == 2
        assert "equity_curve" in item["analysis"]
        assert "explanation" in item["verdict"]


@pytest.mark.parametrize("path", ["/api/health", "/api/factors", "/api/periods"])
def test_existing_routes_still_work(path):
    assert client.get(path).status_code == 200
