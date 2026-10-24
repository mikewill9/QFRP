import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  addViewedConfigs,
  buildShareUrl,
  evidenceKey,
  evidenceLabel,
  fetchBatch,
  fetchClaims,
  fetchPeriods,
  formatNumber,
  formatPercent,
  formatPValue,
  makeConfigKey,
  readSharedParams,
} from "../analysisHelpers.js";
import Checklist from "../components/Checklist.jsx";
import ClaimCard from "../components/ClaimCard.jsx";
import ClaimMatchNotice from "../components/ClaimMatchNotice.jsx";
import ConfigCounter from "../components/ConfigCounter.jsx";
import DualEquityChart from "../components/DualEquityChart.jsx";
import EvidenceSummary from "../components/EvidenceSummary.jsx";
import LoadingProgress from "../components/LoadingProgress.jsx";
import MetricHelp from "../components/MetricHelp.jsx";
import PeriodConsistency from "../components/PeriodConsistency.jsx";
import PermutationHistogram from "../components/PermutationHistogram.jsx";
import ShareLink from "../components/ShareLink.jsx";
import SharedBanner from "../components/SharedBanner.jsx";

// The guided flow, in order. The order and which fields are visible before
// the reveal are fixed (decided); BP edits copy only.
const STEPS = [
  "choose",
  "context",
  "look",
  "predict",
  "reveal",
  "challenge",
  "limits",
  "reflect",
];

const STEP_TITLES = {
  choose: "Choose a question",
  context: "Understand the test",
  look: "Look at the result",
  predict: "Make your prediction",
  reveal: "Reveal the evidence",
  challenge: "Challenge the result",
  limits: "Check the limits",
  reflect: "Reflect",
};

const PREDICTIONS = ["yes", "no", "not_sure"];
const REFLECTIONS = ["more", "less", "same"];

const PREDICTION_LABELS = {
  yes: "Yes",
  no: "No",
  not_sure: "Not sure",
};
const REFLECTION_LABELS = {
  more: "More",
  less: "Less",
  same: "About the same",
};

function StepCard({ step, children }) {
  const index = STEPS.indexOf(step) + 1;

  return (
    <section className={`step-card step-${step}`} aria-labelledby={`step-${step}`}>
      <header className="step-header">
        <span className="step-chip">
          {index} / {STEPS.length}
        </span>

        <h2 id={`step-${step}`}>{STEP_TITLES[step]}</h2>
      </header>

      {children}
    </section>
  );
}

function ClaimExplorerPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const shared = useMemo(() => readSharedParams(searchParams), [searchParams]);

  const [step, setStep] = useState("choose");
  const [claims, setClaims] = useState([]);
  const [periods, setPeriods] = useState([]);
  const [menusLoaded, setMenusLoaded] = useState(false);
  const [menuError, setMenuError] = useState("");

  const [claim, setClaim] = useState(null);
  const [periodKey, setPeriodKey] = useState("");

  // The single-config batch for the look / predict / reveal steps.
  // Sharpe, p-values, histogram and verdict are held here but NOT rendered
  // before the reveal step.
  const [batch, setBatch] = useState(null);
  // The five-period batch for the challenge step.
  const [challenge, setChallenge] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState({ answer: "", note: "" });
  const [reflection, setReflection] = useState("");
  const [isSharedView, setIsSharedView] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadMenus() {
      try {
        const [claimData, periodData] = await Promise.all([
          fetchClaims(),
          fetchPeriods(),
        ]);

        if (!cancelled) {
          setClaims(claimData);
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

  // Shared link: skip to the reveal for that claim/period, fetch with the
  // sender's k, and do NOT touch the recipient's session set.
  const sharedTarget =
    shared.shared && shared.k && menusLoaded
      ? (claims.find((candidate) => candidate.id === shared.claim) ?? null)
      : null;

  useEffect(() => {
    if (!sharedTarget) {
      return undefined;
    }

    let cancelled = false;
    const targetPeriod = shared.period || sharedTarget.default_period_key;

    async function openShared() {
      try {
        const data = await fetchBatch(
          [{ factor_key: sharedTarget.factor_key, period_key: targetPeriod }],
          shared.k,
        );

        if (!cancelled) {
          setBatch({
            k: data.k,
            passBelow: data.pass_below,
            item: data.results[0],
          });
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError.message);
        }
      }

      if (!cancelled) {
        setClaim(sharedTarget);
        setPeriodKey(targetPeriod);
        setIsSharedView(true);
        setStep("reveal");
      }
    }

    openShared();

    return () => {
      cancelled = true;
    };
  }, [sharedTarget, shared.period, shared.k]);

  const sharedPending =
    Boolean(sharedTarget) && step === "choose" && !error;
  const sharedClaimMissing =
    shared.shared && menusLoaded && !menuError && !sharedTarget;

  function reset() {
    setClaim(null);
    setPeriodKey("");
    setBatch(null);
    setChallenge(null);
    setError("");
    setPrediction({ answer: "", note: "" });
    setReflection("");
    setIsSharedView(false);
  }

  function restart() {
    reset();
    setStep("choose");

    if (shared.shared) {
      setSearchParams({}, { replace: true });
    }
  }

  function chooseClaim(selected) {
    reset();
    setClaim(selected);
    setPeriodKey(selected.default_period_key);
    setStep("context");
  }

  // Entering "look": the config joins the session set BEFORE the fetch and
  // k is the size of that set.
  async function startLook() {
    setStep("look");
    setLoading(true);
    setError("");
    setBatch(null);
    setChallenge(null);

    try {
      const config = { factor_key: claim.factor_key, period_key: periodKey };
      const viewed = addViewedConfigs([
        makeConfigKey(config.factor_key, config.period_key),
      ]);
      const data = await fetchBatch([config], viewed.length);

      setBatch({
        k: data.k,
        passBelow: data.pass_below,
        item: data.results[0],
      });
    } catch (fetchError) {
      setError(fetchError.message);
    } finally {
      setLoading(false);
    }
  }

  // "Does it hold up in other periods?": all five periods join the session
  // set at once, then ONE batch fetch at the new k.
  async function runChallenge() {
    setStep("challenge");
    setLoading(true);
    setError("");

    try {
      const configs = periods.map((period) => ({
        factor_key: claim.factor_key,
        period_key: period.key,
      }));
      const viewed = addViewedConfigs(
        configs.map((config) =>
          makeConfigKey(config.factor_key, config.period_key),
        ),
      );
      const data = await fetchBatch(configs, viewed.length);

      setChallenge({
        k: data.k,
        passBelow: data.pass_below,
        results: data.results,
        previousK: batch?.k ?? null,
      });
    } catch (fetchError) {
      setError(fetchError.message);
    } finally {
      setLoading(false);
    }
  }

  const analysis = batch?.item?.analysis ?? null;
  const verdict = batch?.item?.verdict ?? null;

  // The link carries the sender's latest k: after the challenge that is the
  // five-period count, so the recipient's bar matches what the sender saw last.
  const shareUrl =
    claim && batch
      ? buildShareUrl("/claims", {
          claim: claim.id,
          period: periodKey,
          k: challenge?.k ?? batch.k,
          shared: 1,
        })
      : "";

  return (
    <div className="claims-page page-width">
      <section className="page-heading">
        <div>
          <div className="eyebrow">FOLLOW THE EVIDENCE</div>

          <h1>Claim Explorer</h1>

          <p>Choose a supported claim and investigate what the historical results can and cannot tell you.</p>
        </div>
      </section>

      {isSharedView && batch && <SharedBanner k={batch.k} />}

      {menuError && <div className="error-banner">{menuError}</div>}

      {sharedClaimMissing && (
        <div className="error-banner">
          This shared link points to a claim that is no longer available. Choose a claim below to start a new investigation.
        </div>
      )}

      {sharedPending && <LoadingProgress />}

      {step === "choose" && !sharedPending && (
        <StepCard step="choose">
          <p className="step-copy">Choose one of the supported claims below. Each card connects a question you might see online with a specific strategy Fluke can test.</p>

          {!menusLoaded && !menuError && <LoadingProgress />}

          <div className="claim-grid">
            {claims.map((item) => (
              <ClaimCard key={item.id} claim={item} onSelect={chooseClaim} />
            ))}
          </div>
        </StepCard>
      )}

      {step === "context" && claim && (
        <StepCard step="context">
          <blockquote className="claim-question">{claim.question}</blockquote>

          <ClaimMatchNotice claim={claim} />

          <p className="step-copy">First, check how closely the tested strategy matches the claim. Next, inspect its historical results and make a prediction before revealing the statistical evidence.</p>

          <div className="step-actions">
            <button type="button" className="primary-button" onClick={startLook}>
              Start
            </button>

            <button type="button" className="secondary-button" onClick={restart}>
              Back to claims
            </button>
          </div>
        </StepCard>
      )}

      {step === "look" && claim && (
        <StepCard step="look">
          <p className="step-copy">Look at the historical growth, total return, and largest decline. Consider what these results suggest and what questions you still have.</p>

          {loading && <LoadingProgress />}

          {error && <div className="error-banner">{error}</div>}

          {analysis && (
            <>
              <div className="result-title-row">
                <div>
                  <span className="result-kicker">{analysis.period_label}</span>

                  <h3>{analysis.factor_label}</h3>
                </div>
              </div>

              <div className="metric-grid metric-grid-two">
                <div className="metric-card">
                  <span>Total return</span>

                  <strong>{formatPercent(analysis.total_return)}</strong>
                </div>

                <div className="metric-card">
                  <span>Max drawdown</span>

                  <strong>{formatPercent(analysis.max_drawdown)}</strong>
                </div>
              </div>

              <div className="chart-card">
                <DualEquityChart a={batch.item} />
              </div>

              <div className="step-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => setStep("predict")}
                >
                  Make a prediction
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "predict" && (
        <StepCard step="predict">
          <p className="step-copy">Based on the historical result, do you think this strategy will show unusually strong evidence compared with random reshuffles?</p>

          <div className="choice-row" role="group" aria-label="Your prediction">
            {PREDICTIONS.map((answer) => (
              <button
                key={answer}
                type="button"
                className={`choice-button${
                  prediction.answer === answer ? " choice-selected" : ""
                }`}
                aria-pressed={prediction.answer === answer}
                onClick={() => setPrediction({ ...prediction, answer })}
              >
                {PREDICTION_LABELS[answer]}
              </button>
            ))}
          </div>

          <label className="note-field">
            <span>Why do you think so? (optional)</span>

            <textarea
              rows={3}
              value={prediction.note}
              onChange={(event) =>
                setPrediction({ ...prediction, note: event.target.value })
              }
            />
          </label>

          <div className="step-actions">
            <button
              type="button"
              className="primary-button"
              disabled={!prediction.answer}
              onClick={() => setStep("reveal")}
            >
              Reveal the evidence
            </button>
          </div>
        </StepCard>
      )}

      {step === "reveal" && (
        <StepCard step="reveal">
          <p className="step-copy">Fluke checks whether a strategy's result is real or could just be luck. The test compares the observed gross Sharpe with random reshuffles of the same factor. The classification also accounts for how many unique configurations are included in this investigation.</p>

          {sharedPending && <LoadingProgress />}

          {error && <div className="error-banner">{error}</div>}

          {analysis && verdict && (
            <>
              <ConfigCounter k={batch.k} passBelow={batch.passBelow} />

              <div className="chart-card">
                <div className="card-heading">
                  <h3>
                    Observed result versus random reshuffles
                  </h3>

                  <p>
                    The histogram compares {analysis.n_permutations.toLocaleString()} random reshuffles with the observed gross Sharpe.
                  </p>
                </div>

                <PermutationHistogram
                  result={analysis}
                  formatNumber={formatNumber}
                />
              </div>

              <div className={`evidence-badge evidence-${evidenceKey(verdict).toLowerCase()}`}>
                {evidenceLabel(verdict)}
              </div>

              <EvidenceSummary analysis={analysis} verdict={verdict} />

              <details className="test-details">
                <summary>View statistical details</summary>

                <dl className="details-list">
                  <div>
                    <dt>
                      Raw p-value <MetricHelp metric="pValue" />
                    </dt>

                    <dd>{formatPValue(verdict.raw_p_value)}</dd>
                  </div>

                  <div>
                    <dt>
                      Adjusted p-value <MetricHelp metric="adjustedP" />
                    </dt>

                    <dd>{formatPValue(verdict.adjusted_p_value)}</dd>
                  </div>

                  <div>
                    <dt>k</dt>

                    <dd>{verdict.n_configs_viewed}</dd>
                  </div>

                  <div>
                    <dt>Internal label</dt>

                    <dd>
                      {verdict.label}
                      {verdict.reversed ? " (reversed)" : ""}
                    </dd>
                  </div>
                </dl>
              </details>

              <div className="step-actions">
                <button type="button" className="primary-button" onClick={runChallenge}>
                  Does it hold up in other periods?
                </button>

                <button type="button" className="secondary-button" onClick={restart}>
                  Start over
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "challenge" && (
        <StepCard step="challenge">
          <p className="step-copy">Check the same factor across all available periods. Previously unseen factor-period combinations increase the count, and every row below uses the updated adjustment.</p>

          {loading && <LoadingProgress />}

          {error && <div className="error-banner">{error}</div>}

          {challenge && (
            <>
              <ConfigCounter
                k={challenge.k}
                passBelow={challenge.passBelow}
                previousK={challenge.previousK}
              />

              <PeriodConsistency
                results={challenge.results}
                passBelow={challenge.passBelow}
              />

              <div className="step-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => setStep("limits")}
                >
                  Check costs and limits
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "limits" && analysis && claim && (
        <StepCard step="limits">
          <p className="step-copy">The result depends on the data and assumptions used in the test. Review trading costs and the limitations before deciding what the evidence means for the claim.</p>

          <div className="chart-card">
            <div className="card-heading">
              <h3>Historical cost sensitivity</h3>

              <p>The 2x and 3x cost scenarios are what-if checks on the same history; they do not produce new statistical verdicts.</p>
            </div>

            <div className="table-scroll">
              <table className="cost-table">
                <thead>
                  <tr>
                    <th scope="col">Cost multiplier</th>

                    <th scope="col">Sharpe</th>
                  </tr>
                </thead>

                <tbody>
                  {analysis.cost_stress.map((point) => (
                    <tr key={point.multiplier}>
                      <th scope="row">{point.multiplier}x</th>

                      <td>{formatNumber(point.sharpe)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="limitations-card">
            <h3>Limitations of this test</h3>

            <ul>
              {analysis.limitations.map((limitation, index) => (
                <li key={`${index}-${limitation}`}>{limitation}</li>
              ))}
            </ul>
          </div>

          <div className="claim-gap-card">
            <h3>What the claim says that this test does not cover</h3>

            <p>{claim.what_it_does_not_test}</p>
          </div>

          <div className="step-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep("reflect")}
            >
              Reflect on the evidence
            </button>
          </div>
        </StepCard>
      )}

      {step === "reflect" && claim && (
        <StepCard step="reflect">
          <p className="step-copy">After reviewing the evidence and limitations, are you more convinced, less convinced, or about as convinced by the claim as before?</p>

          <div className="choice-row" role="group" aria-label="Your reflection">
            {REFLECTIONS.map((answer) => (
              <button
                key={answer}
                type="button"
                className={`choice-button${
                  reflection === answer ? " choice-selected" : ""
                }`}
                aria-pressed={reflection === answer}
                onClick={() => setReflection(answer)}
              >
                {REFLECTION_LABELS[answer]}
              </button>
            ))}
          </div>

          {reflection && (
            <p className="reflection-summary">
              {prediction.answer
                ? `Your first answer was “${PREDICTION_LABELS[prediction.answer]}.” `
                : "No prediction was recorded in this shared view. "}
              Your confidence after reviewing the evidence: {REFLECTION_LABELS[reflection].toLowerCase()}.
            </p>
          )}

          <Checklist />

          {shareUrl && (
            <div className="share-card">
              <h3>Share this investigation</h3>

              <ShareLink url={shareUrl} />
            </div>
          )}

          <div className="step-actions">
            <Link to="/compare" className="primary-button">
              Ask your own questions in the Comparison Lab
            </Link>

            <button type="button" className="secondary-button" onClick={restart}>
              Start over
            </button>
          </div>
        </StepCard>
      )}
    </div>
  );
}

export default ClaimExplorerPage;
