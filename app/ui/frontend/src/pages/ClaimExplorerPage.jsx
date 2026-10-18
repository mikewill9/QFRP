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

// >>> BP: one short title per step, shown in each step card's header
const STEP_TITLES = {
  choose: "[BP: step title: pick a claim]",
  context: "[BP: step title: what Fluke tested]",
  look: "[BP: step title: look at the result]",
  predict: "[BP: step title: make a prediction]",
  reveal: "[BP: step title: the randomization test]",
  challenge: "[BP: step title: other periods]",
  limits: "[BP: step title: costs and limits]",
  reflect: "[BP: step title: reflect and share]",
};

const PREDICTIONS = ["yes", "no", "not_sure"];
const REFLECTIONS = ["more", "less", "same"];

// >>> BP: the answer labels for the predict and reflect steps
const PREDICTION_LABELS = {
  yes: "[BP: Yes]",
  no: "[BP: No]",
  not_sure: "[BP: Not sure]",
};
const REFLECTION_LABELS = {
  more: "[BP: More]",
  less: "[BP: Less]",
  same: "[BP: About the same]",
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
          {/* >>> BP: page eyebrow, title and intro line <<< */}
          <div className="eyebrow">[BP: CLAIM EXPLORER EYEBROW]</div>

          <h1>[BP: page title]</h1>

          <p>[BP: one line: what this guided investigation does]</p>
        </div>
      </section>

      {isSharedView && batch && <SharedBanner k={batch.k} />}

      {menuError && <div className="error-banner">{menuError}</div>}

      {sharedClaimMissing && (
        <div className="error-banner">
          {/* >>> BP: one line: this shared link points at a claim that no longer exists <<< */}
          [BP: one line: this shared link points at a claim that no longer
          exists; pick one below]
        </div>
      )}

      {sharedPending && <LoadingProgress />}

      {step === "choose" && !sharedPending && (
        <StepCard step="choose">
          {/* >>> BP: choose-step body copy <<< */}
          <p className="step-copy">[BP: one or two sentences: pick a claim you have seen online]</p>

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

          {/* >>> BP: context-step body copy <<< */}
          <p className="step-copy">[BP: one or two sentences: here is what we will do next]</p>

          <div className="step-actions">
            <button type="button" className="primary-button" onClick={startLook}>
              Start
            </button>

            <button type="button" className="secondary-button" onClick={restart}>
              [BP: back to claims]
            </button>
          </div>
        </StepCard>
      )}

      {step === "look" && claim && (
        <StepCard step="look">
          {/* >>> BP: look-step body copy (no verdict language here; the user has not predicted yet) <<< */}
          <p className="step-copy">[BP: one or two sentences: here is the strategy's historical result; no verdict yet]</p>

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
                  [BP: continue]
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "predict" && (
        <StepCard step="predict">
          {/* >>> BP: predict-step question and helper copy <<< */}
          <p className="step-copy">[BP: the question: does this look convincing?]</p>

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
            <span>[BP: optional note label, e.g. why?]</span>

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
              [BP: reveal the test]
            </button>
          </div>
        </StepCard>
      )}

      {step === "reveal" && (
        <StepCard step="reveal">
          {/* >>> BP: reveal-step body copy <<< */}
          <p className="step-copy">[BP: one or two sentences: here is how the result compares with random reshuffles]</p>

          {sharedPending && <LoadingProgress />}

          {error && <div className="error-banner">{error}</div>}

          {analysis && verdict && (
            <>
              <ConfigCounter k={batch.k} passBelow={batch.passBelow} />

              <div className="chart-card">
                <div className="card-heading">
                  <h3>
                    {/* >>> BP: histogram heading and one-sentence caption <<< */}
                    [BP: histogram heading]
                  </h3>

                  <p>
                    [BP: one sentence explaining what the histogram shows;{" "}
                    {analysis.n_permutations.toLocaleString()} reshuffles]
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
                <summary>[BP: details toggle label]</summary>

                <dl className="details-list">
                  <div>
                    <dt>Raw p-value</dt>

                    <dd>{formatPValue(verdict.raw_p_value)}</dd>
                  </div>

                  <div>
                    <dt>Adjusted p-value</dt>

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
                  [BP: start over]
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "challenge" && (
        <StepCard step="challenge">
          {/* >>> BP: challenge-step body copy: the counter just jumped <<< */}
          <p className="step-copy">[BP: one or two sentences: you just looked at five configurations, watch the bar move]</p>

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
                  [BP: continue]
                </button>
              </div>
            </>
          )}
        </StepCard>
      )}

      {step === "limits" && analysis && claim && (
        <StepCard step="limits">
          {/* >>> BP: limits-step body copy <<< */}
          <p className="step-copy">[BP: one or two sentences: what the test can still miss]</p>

          <div className="chart-card">
            <div className="card-heading">
              <h3>[BP: cost table heading: historical cost sensitivity]</h3>

              <p>[BP: one sentence: 2x and 3x are what-ifs on the same history, not new verdicts]</p>
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
            <h3>[BP: limitations heading]</h3>

            <ul>
              {analysis.limitations.map((limitation, index) => (
                <li key={`${index}-${limitation}`}>{limitation}</li>
              ))}
            </ul>
          </div>

          <div className="claim-gap-card">
            <h3>[BP: heading: what this claim says that the test does not cover]</h3>

            <p>{claim.what_it_does_not_test}</p>
          </div>

          <div className="step-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => setStep("reflect")}
            >
              [BP: continue]
            </button>
          </div>
        </StepCard>
      )}

      {step === "reflect" && claim && (
        <StepCard step="reflect">
          {/* >>> BP: reflect-step body copy and the "you said X, now Y" sentence <<< */}
          <p className="step-copy">[BP: the question: more or less convinced than before?]</p>

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
              [BP: one sentence: you said{" "}
              {PREDICTION_LABELS[prediction.answer] ?? "[BP: no prediction on a shared link]"}, now{" "}
              {REFLECTION_LABELS[reflection]}]
            </p>
          )}

          <Checklist />

          {shareUrl && (
            <div className="share-card">
              <h3>[BP: share heading]</h3>

              <ShareLink url={shareUrl} />
            </div>
          )}

          <div className="step-actions">
            <Link to="/compare" className="primary-button">
              [BP: one line handing off: now ask your own questions and watch
              the bar]
            </Link>

            <button type="button" className="secondary-button" onClick={restart}>
              [BP: start over]
            </button>
          </div>
        </StepCard>
      )}
    </div>
  );
}

export default ClaimExplorerPage;
