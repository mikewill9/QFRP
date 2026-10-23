import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  addViewedConfigs,
  buildShareUrl,
  fetchBatch,
  fetchFactors,
  fetchPeriods,
  formatNumber,
  formatPercent,
  formatPValue,
  makeConfigKey,
  readSharedParams,
} from "../analysisHelpers.js";
import ComparisonTable from "../components/ComparisonTable.jsx";
import ConfigCounter from "../components/ConfigCounter.jsx";
import DualEquityChart from "../components/DualEquityChart.jsx";
import LoadingProgress from "../components/LoadingProgress.jsx";
import PeriodConsistency from "../components/PeriodConsistency.jsx";
import PermutationHistogram from "../components/PermutationHistogram.jsx";
import ShareLink from "../components/ShareLink.jsx";
import SharedBanner from "../components/SharedBanner.jsx";
import AnalyzerPage from "./AnalyzerPage.jsx";

const TABS = [
  { id: "compare", label: "Compare" },
  { id: "periods", label: "Across periods" },
  { id: "advanced", label: "Advanced" },
];

function tabFromParam(value) {
  return TABS.some((tab) => tab.id === value) ? value : "compare";
}

// Two factors, same period. Each is tested against its OWN null; this is
// never a test of A vs B and the copy must not say one beat the other.
function CompareTab({ factors, periods, shared }) {
  const [factorA, setFactorA] = useState(shared.a || factors[0]?.key || "");
  const [factorB, setFactorB] = useState(
    shared.b || factors[1]?.key || factors[0]?.key || "",
  );
  const [periodKey, setPeriodKey] = useState(shared.period || periods[0]?.key || "");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isSharedView, setIsSharedView] = useState(false);

  const sharedReady =
    shared.shared && shared.k && shared.a && shared.b && shared.period;

  useEffect(() => {
    if (!sharedReady) {
      return undefined;
    }

    let cancelled = false;

    async function openShared() {
      try {
        const data = await fetchBatch(
          [
            { factor_key: shared.a, period_key: shared.period },
            { factor_key: shared.b, period_key: shared.period },
          ],
          shared.k,
        );

        if (!cancelled) {
          setResult({
            k: data.k,
            passBelow: data.pass_below,
            a: data.results[0],
            b: data.results[1],
            previousK: null,
          });
          setIsSharedView(true);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError.message);
          setIsSharedView(true);
        }
      }
    }

    openShared();

    return () => {
      cancelled = true;
    };
  }, [sharedReady, shared.a, shared.b, shared.period, shared.k]);

  const sameFactor = factorA && factorA === factorB;

  async function compare(event) {
    event.preventDefault();

    if (!factorA || !factorB || !periodKey || sameFactor) {
      return;
    }

    setLoading(true);
    setError("");
    setIsSharedView(false);

    try {
      const configs = [
        { factor_key: factorA, period_key: periodKey },
        { factor_key: factorB, period_key: periodKey },
      ];
      const previousK = result?.k ?? null;
      const viewed = addViewedConfigs(
        configs.map((config) =>
          makeConfigKey(config.factor_key, config.period_key),
        ),
      );
      const data = await fetchBatch(configs, viewed.length);

      setResult({
        k: data.k,
        passBelow: data.pass_below,
        a: data.results[0],
        b: data.results[1],
        previousK,
      });
    } catch (fetchError) {
      setError(fetchError.message);
    } finally {
      setLoading(false);
    }
  }

  const a = result?.a ?? null;
  const b = result?.b ?? null;

  // Plain comparisons of two numbers for the two sentences BP writes. Not a
  // statistical test: "higher return" is a fact about the history, nothing
  // more, and "lower adjusted p" is each factor's own test.
  const higherReturn =
    a && b
      ? a.analysis.total_return >= b.analysis.total_return
        ? a
        : b
      : null;
  const lowerAdjustedP =
    a && b
      ? a.analysis.adjusted_p_value <= b.analysis.adjusted_p_value
        ? a
        : b
      : null;

  const shareUrl = result
    ? buildShareUrl("/compare", {
        a: a.factor_key,
        b: b.factor_key,
        period: a.period_key,
        k: result.k,
        shared: 1,
      })
    : "";

  return (
    <div className="lab-tab">
      <form className="lab-controls" onSubmit={compare}>
        <label>
          <span>Factor A</span>

          <select value={factorA} onChange={(event) => setFactorA(event.target.value)}>
            {factors.map((factor) => (
              <option key={factor.key} value={factor.key}>
                {factor.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Factor B</span>

          <select value={factorB} onChange={(event) => setFactorB(event.target.value)}>
            {factors.map((factor) => (
              <option key={factor.key} value={factor.key}>
                {factor.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Period</span>

          <select value={periodKey} onChange={(event) => setPeriodKey(event.target.value)}>
            {periods.map((period) => (
              <option key={period.key} value={period.key}>
                {period.label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          className="primary-button"
          disabled={loading || sameFactor || !factorA || !factorB || !periodKey}
        >
          Compare
        </button>
      </form>

      {sameFactor && (
        <p className="helper-text">Choose two different factors to compare.</p>
      )}

      {isSharedView && result && <SharedBanner k={result.k} />}

      {loading && <LoadingProgress />}

      {error && <div className="error-banner">{error}</div>}

      {a && b && (
        <div className="lab-results">
          <ConfigCounter
            k={result.k}
            passBelow={result.passBelow}
            previousK={result.previousK}
          />

          <div className="chart-card">
            <ComparisonTable a={a} b={b} />
          </div>

          <div className="chart-card">
            <DualEquityChart a={a} b={b} />
          </div>

          <div className="two-column-grid">
            {[a, b].map((item) => (
              <div className="chart-card" key={item.factor_key}>
                <div className="card-heading">
                  <h3>{item.analysis.factor_label}</h3>

                  <p>
                    Raw p {formatPValue(item.analysis.p_value)} · adjusted p{" "}
                    {formatPValue(item.analysis.adjusted_p_value)}
                  </p>
                </div>

                <PermutationHistogram result={item.analysis} formatNumber={formatNumber} />
              </div>
            ))}
          </div>

          <section className="what-this-shows">
            <h3>What this shows</h3>

            <p>
              {!Number.isFinite(a.analysis.total_return) || !Number.isFinite(b.analysis.total_return) ? (
                "A historical-return comparison is unavailable for these results."
              ) : a.analysis.total_return === b.analysis.total_return ? (
                <>Both factors had the same total return in {a.analysis.period_label}: {formatPercent(a.analysis.total_return)}.</>
              ) : (
                <>{higherReturn.analysis.factor_label} had the higher historical total return in {a.analysis.period_label}: {formatPercent(higherReturn.analysis.total_return)}.</>
              )}
            </p>

            <p>
              {!Number.isFinite(a.analysis.adjusted_p_value) || !Number.isFinite(b.analysis.adjusted_p_value) ? (
                "An adjusted p-value comparison is unavailable for these results."
              ) : a.analysis.adjusted_p_value === b.analysis.adjusted_p_value ? (
                <>Both factors had the same adjusted p-value: {formatPValue(a.analysis.adjusted_p_value)}.</>
              ) : (
                <>{lowerAdjustedP.analysis.factor_label} had the lower adjusted p-value in its own test: {formatPValue(lowerAdjustedP.analysis.adjusted_p_value)}.</>
              )}
            </p>

            <p className="structural-line">
              Each factor is tested against its own randomized null distribution. This is not a statistical test showing that one factor is better than the other.
            </p>
          </section>

          <div className="two-column-grid">
            {[a, b].map((item) => (
              <div className="limitations-card" key={item.factor_key}>
                <h3>{item.analysis.factor_label}: limitations</h3>

                <ul>
                  {item.analysis.limitations.map((limitation, index) => (
                    <li key={`${index}-${limitation}`}>{limitation}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="share-card">
            <h3>Share this comparison</h3>

            <ShareLink url={shareUrl} />
          </div>
        </div>
      )}
    </div>
  );
}

// One factor across all five periods, one batch fetch, one shared k.
function PeriodsTab({ factors, periods, shared }) {
  const [factorKey, setFactorKey] = useState(shared.factor || factors[0]?.key || "");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isSharedView, setIsSharedView] = useState(false);

  const sharedReady = shared.shared && shared.k && shared.factor && periods.length > 0;

  useEffect(() => {
    if (!sharedReady) {
      return undefined;
    }

    let cancelled = false;

    async function openShared() {
      try {
        const data = await fetchBatch(
          periods.map((period) => ({
            factor_key: shared.factor,
            period_key: period.key,
          })),
          shared.k,
        );

        if (!cancelled) {
          setResult({
            k: data.k,
            passBelow: data.pass_below,
            results: data.results,
            previousK: null,
          });
          setIsSharedView(true);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError.message);
          setIsSharedView(true);
        }
      }
    }

    openShared();

    return () => {
      cancelled = true;
    };
  }, [sharedReady, shared.factor, shared.k, periods]);

  async function run(event) {
    event.preventDefault();

    if (!factorKey || periods.length === 0) {
      return;
    }

    setLoading(true);
    setError("");
    setIsSharedView(false);

    try {
      const configs = periods.map((period) => ({
        factor_key: factorKey,
        period_key: period.key,
      }));
      const previousK = result?.k ?? null;
      const viewed = addViewedConfigs(
        configs.map((config) =>
          makeConfigKey(config.factor_key, config.period_key),
        ),
      );
      const data = await fetchBatch(configs, viewed.length);

      setResult({
        k: data.k,
        passBelow: data.pass_below,
        results: data.results,
        previousK,
      });
    } catch (fetchError) {
      setError(fetchError.message);
    } finally {
      setLoading(false);
    }
  }

  const shareUrl = result
    ? buildShareUrl("/compare", {
        tab: "periods",
        factor: result.results[0]?.factor_key,
        k: result.k,
        shared: 1,
      })
    : "";

  return (
    <div className="lab-tab">
      <form className="lab-controls" onSubmit={run}>
        <label>
          <span>Factor</span>

          <select value={factorKey} onChange={(event) => setFactorKey(event.target.value)}>
            {factors.map((factor) => (
              <option key={factor.key} value={factor.key}>
                {factor.label}
              </option>
            ))}
          </select>
        </label>

        <button type="submit" className="primary-button" disabled={loading || !factorKey}>
          Check all periods
        </button>
      </form>

      {isSharedView && result && <SharedBanner k={result.k} />}

      {loading && <LoadingProgress />}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="lab-results">
          <ConfigCounter
            k={result.k}
            passBelow={result.passBelow}
            previousK={result.previousK}
          />

          <div className="chart-card">
            <div className="card-heading">
              <h3>{result.results[0]?.analysis.factor_label}</h3>

              <p>Compare the same factor across the available periods, with each period tested against its own random reshuffles and every row using the same configuration count.</p>
            </div>

            <PeriodConsistency results={result.results} passBelow={result.passBelow} />
          </div>

          <div className="share-card">
            <h3>Share this period check</h3>

            <ShareLink url={shareUrl} />
          </div>
        </div>
      )}
    </div>
  );
}

function ComparisonLabPage() {
  const [searchParams] = useSearchParams();
  const shared = useMemo(() => readSharedParams(searchParams), [searchParams]);

  const [tab, setTab] = useState(() => tabFromParam(shared.tab));
  const [factors, setFactors] = useState([]);
  const [periods, setPeriods] = useState([]);
  const [menusLoaded, setMenusLoaded] = useState(false);
  const [menuError, setMenuError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadMenus() {
      try {
        const [factorData, periodData] = await Promise.all([
          fetchFactors(),
          fetchPeriods(),
        ]);

        if (!cancelled) {
          setFactors(factorData);
          setPeriods(periodData);
          setMenusLoaded(true);
        }
      } catch (loadError) {
        if (!cancelled) {
          setMenuError(loadError.message);
          setMenusLoaded(true);
        }
      }
    }

    loadMenus();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="lab-page page-width">
      <section className="page-heading">
        <div>
          <div className="eyebrow">ASK YOUR OWN QUESTIONS</div>

          <h1>Comparison Lab</h1>

          <p>Compare historical results, inspect the evidence, and watch how exploring more configurations changes the adjustment.</p>
        </div>
      </section>

      <div className="tab-bar" role="tablist" aria-label="Comparison Lab sections">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={`tab-button${tab === item.id ? " tab-active" : ""}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {menuError && <div className="error-banner">{menuError}</div>}

      {!menusLoaded && !menuError && <LoadingProgress />}

      {menusLoaded && !menuError && tab === "compare" && (
        <CompareTab factors={factors} periods={periods} shared={shared} />
      )}

      {menusLoaded && !menuError && tab === "periods" && (
        <PeriodsTab factors={factors} periods={periods} shared={shared} />
      )}

      {tab === "advanced" && (
        <div className="lab-tab">
          <AnalyzerPage embedded />
        </div>
      )}
    </div>
  );
}

export default ComparisonLabPage;
