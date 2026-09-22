import { useEffect, useMemo, useState } from "react";

import Plot from "react-plotly.js";
import { apiFetch } from "../api.js";

const VIEWED_KEY = "qfrp-viewed-configs";

function loadViewedConfigs() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(VIEWED_KEY) ?? "[]");

    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function formatPercent(value) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function metricTone(value) {
  if (value > 0) return "positive";
  if (value < 0) return "negative";

  return "neutral";
}

function AnalyzerPage() {
  const [factors, setFactors] = useState([]);
  const [periods, setPeriods] = useState([]);

  const [factorKey, setFactorKey] = useState("");

  const [periodKey, setPeriodKey] = useState("");

  const [result, setResult] = useState(null);

  const [verdict, setVerdict] = useState(null);

  const [viewedConfigs, setViewedConfigs] = useState(loadViewedConfigs);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadMenus() {
      try {
        const [factorData, periodData] = await Promise.all([
          apiFetch("/api/factors"),
          apiFetch("/api/periods"),
        ]);

        if (!cancelled) {
          setFactors(factorData);
          setPeriods(periodData);

          setFactorKey(factorData[0]?.key ?? "");

          setPeriodKey(periodData[0]?.key ?? "");
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      }
    }

    loadMenus();

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedFactor = useMemo(
    () => factors.find((factor) => factor.key === factorKey),
    [factors, factorKey],
  );

  async function runAnalysis(event) {
    event.preventDefault();

    if (!factorKey || !periodKey) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const comboKey = `${factorKey}::${periodKey}`;

      const nextViewed = viewedConfigs.includes(comboKey)
        ? viewedConfigs
        : [...viewedConfigs, comboKey];

      const analysisData = await apiFetch(
        `/api/analysis?factor_key=${encodeURIComponent(
          factorKey,
        )}&period_key=${encodeURIComponent(periodKey)}`,
      );

      const verdictData = await apiFetch("/api/verdict", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          factor_key: factorKey,

          period_key: periodKey,

          n_configs_viewed: nextViewed.length,
        }),
      });

      sessionStorage.setItem(VIEWED_KEY, JSON.stringify(nextViewed));

      setViewedConfigs(nextViewed);

      setResult(analysisData);

      setVerdict(verdictData);
    } catch (runError) {
      setError(runError.message);
    } finally {
      setLoading(false);
    }
  }

  const placeholderData = result?.limitations?.some((item) =>
    item.startsWith("PLACEHOLDER"),
  );

  return (
    <div className="analyzer-page page-width">
      <section className="page-heading">
        <div>
          <div className="eyebrow">FACTOR REALITY CHECK</div>

          <h1>Analyzer</h1>

          <p>
            Compare a factor's observed result with its permutation-based null
            distribution.
          </p>
        </div>

        <div
          className="session-pill"
          title="Unique factor-period combinations viewed in this browser tab"
        >
          <strong>{viewedConfigs.length}</strong>

          <span>configurations viewed</span>
        </div>
      </section>

      <div className="analyzer-layout">
        <aside className="control-panel">
          <h2>Test configuration</h2>

          <form onSubmit={runAnalysis}>
            <label htmlFor="factor">Factor</label>

            <select
              id="factor"
              value={factorKey}
              onChange={(event) => setFactorKey(event.target.value)}
            >
              {factors.map((factor) => (
                <option key={factor.key} value={factor.key}>
                  {factor.label}
                </option>
              ))}
            </select>

            {selectedFactor && (
              <p className="helper-text">{selectedFactor.description}</p>
            )}

            <label htmlFor="period">Date range</label>

            <select
              id="period"
              value={periodKey}
              onChange={(event) => setPeriodKey(event.target.value)}
            >
              {periods.map((period) => (
                <option key={period.key} value={period.key}>
                  {period.label}
                </option>
              ))}
            </select>

            <button
              className="primary-button full-width"
              type="submit"
              disabled={loading || !factorKey || !periodKey}
            >
              {loading ? "Loading analysis…" : "Run analysis"}
            </button>
          </form>

          <div className="counter-explainer">
            <h3>Why the counter matters</h3>

            <p>
              Testing more configurations raises the chance that one appears
              strong by luck. The verdict accounts for how many unique
              combinations you have viewed.
            </p>
          </div>
        </aside>

        <section className="results-column" aria-live="polite">
          {error && <div className="error-banner">{error}</div>}

          {!result && !error && (
            <div className="empty-state">
              <span className="empty-icon">↗</span>

              <h2>Choose a configuration to begin</h2>

              <p>
                Your performance metrics, permutation test, cost sensitivity,
                and verdict will appear here.
              </p>
            </div>
          )}

          {result && verdict && (
            <>
              <div className="result-title-row">
                <div>
                  <span className="result-kicker">{result.period_label}</span>

                  <h2>{result.factor_label}</h2>
                </div>

                {placeholderData && (
                  <span className="demo-badge">DEMO DATA</span>
                )}
              </div>

              <div className="metric-grid">
                <Metric
                  label="Sharpe"
                  value={formatNumber(result.sharpe)}
                  tone={metricTone(result.sharpe)}
                />

                <Metric
                  label="Total return"
                  value={formatPercent(result.total_return)}
                  tone={metricTone(result.total_return)}
                />

                <Metric
                  label="Max drawdown"
                  value={formatPercent(result.max_drawdown)}
                  tone="negative"
                />

                <Metric
                  label="Hit rate"
                  value={formatPercent(result.hit_rate)}
                />

                <Metric label="Trades" value={String(result.n_trades)} />
              </div>

              <div className="chart-card">
                <div className="card-heading">
                  <h3>Equity curve</h3>

                  <p>
                    Strategy performance with the equal-weight long-only
                    universe shown for context.
                  </p>
                </div>

                <Plot
                  data={[
                    {
                      x: result.equity_curve.map((point) => point.date),

                      y: result.equity_curve.map((point) => point.strategy),

                      type: "scatter",

                      mode: "lines",

                      name: "Strategy",

                      line: {
                        width: 3,
                      },
                    },

                    {
                      x: result.equity_curve.map((point) => point.date),

                      y: result.equity_curve.map((point) => point.benchmark),

                      type: "scatter",

                      mode: "lines",

                      name: "Equal-weight context",

                      line: {
                        width: 2,
                      },
                    },
                  ]}
                  layout={baseChartLayout({
                    yaxis: {
                      title: "Growth of $1",

                      gridcolor: "#e5e7eb",
                    },

                    xaxis: {
                      gridcolor: "#f3f4f6",
                    },
                  })}
                  config={{
                    displayModeBar: false,

                    responsive: true,
                  }}
                  useResizeHandler
                  className="plot"
                />
              </div>

              <div className="chart-card">
                <div className="card-heading split-heading">
                  <div>
                    <h3>Permutation test</h3>

                    <p>
                      {result.n_permutations.toLocaleString()} reshuffles of the
                      same factor, compared with the observed Sharpe.
                    </p>
                  </div>

                  <div className="p-value-box">
                    <span>Raw p-value</span>

                    <strong>{formatPercent(result.p_value)}</strong>
                  </div>
                </div>

                <Plot
                  data={[
                    {
                      x: result.null_sharpes,

                      type: "histogram",

                      name: "Reshuffled Sharpe",

                      opacity: 0.9,
                    },
                  ]}
                  layout={baseChartLayout({
                    xaxis: {
                      title: "Sharpe ratio",

                      gridcolor: "#f3f4f6",
                    },

                    yaxis: {
                      title: "Count",

                      gridcolor: "#e5e7eb",
                    },

                    shapes: [
                      {
                        type: "line",

                        x0: result.sharpe,

                        x1: result.sharpe,

                        y0: 0,

                        y1: 1,

                        yref: "paper",

                        line: {
                          width: 3,

                          dash: "dash",
                        },
                      },
                    ],

                    annotations: [
                      {
                        x: result.sharpe,

                        y: 1,

                        yref: "paper",

                        text: `Observed Sharpe: ${formatNumber(result.sharpe)}`,

                        showarrow: true,

                        arrowhead: 2,

                        ax: 45,

                        ay: 35,
                      },
                    ],
                  })}
                  config={{
                    displayModeBar: false,

                    responsive: true,
                  }}
                  useResizeHandler
                  className="plot"
                />
              </div>

              <div className="two-column-grid">
                <div className="chart-card">
                  <div className="card-heading">
                    <h3>Cost stress</h3>

                    <p>
                      How the result changes as the assumed trading cost is
                      multiplied.
                    </p>
                  </div>

                  <Plot
                    data={[
                      {
                        x: result.cost_stress.map(
                          (point) => `${point.multiplier}×`,
                        ),

                        y: result.cost_stress.map((point) => point.sharpe),

                        type: "bar",

                        name: "Sharpe",
                      },
                    ]}
                    layout={baseChartLayout({
                      xaxis: {
                        title: "Cost multiplier",
                      },

                      yaxis: {
                        title: "Sharpe ratio",

                        gridcolor: "#e5e7eb",

                        zeroline: true,
                      },
                    })}
                    config={{
                      displayModeBar: false,

                      responsive: true,
                    }}
                    useResizeHandler
                    className="plot compact-plot"
                  />
                </div>

                <div
                  className={`verdict-card verdict-${verdict.label.toLowerCase()}`}
                >
                  <div className="verdict-label">{verdict.label}</div>

                  <h3>{verdict.headline}</h3>

                  <p>{verdict.explanation}</p>

                  <div className="verdict-stats">
                    <div>
                      <span>Raw p</span>

                      <strong>{formatPercent(verdict.raw_p_value)}</strong>
                    </div>

                    <div>
                      <span>Adjusted p</span>

                      <strong>{formatPercent(verdict.adjusted_p_value)}</strong>
                    </div>

                    <div>
                      <span>Configs viewed</span>

                      <strong>{verdict.n_configs_viewed}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div className="limitations-card">
                <h3>Limitations</h3>

                <ul>
                  {result.limitations.map((limitation, index) => (
                    <li key={`${index}-${limitation}`}>{limitation}</li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className="metric-card">
      <span>{label}</span>

      <strong className={`metric-${tone}`}>{value}</strong>
    </div>
  );
}

function baseChartLayout(overrides = {}) {
  return {
    autosize: true,

    height: 360,

    margin: {
      l: 55,
      r: 20,
      t: 20,
      b: 55,
    },

    paper_bgcolor: "#ffffff",

    plot_bgcolor: "#ffffff",

    font: {
      family: "Inter, ui-sans-serif, system-ui, sans-serif",

      color: "#374151",
    },

    hovermode: "x unified",

    showlegend: true,

    legend: {
      orientation: "h",

      x: 0,

      y: 1.12,
    },

    ...overrides,
  };
}

export default AnalyzerPage;
