import { evidenceKey, evidenceLabel } from "../analysisHelpers.js";

// Three fixed blocks. "What it may mean" is the backend's explanation,
// verbatim: the contract says it is displayed without rewriting.
function EvidenceSummary({ analysis, verdict }) {
  const tone = evidenceKey(verdict).toLowerCase();

  return (
    <section className={`evidence-summary evidence-${tone}`}>
      <div className="evidence-block">
        <h3>What we found</h3>

        <p className="evidence-label">{evidenceLabel(verdict)}</p>

        <p className="evidence-context">
          {analysis.factor_label} · {analysis.period_label}
        </p>
      </div>

      <div className="evidence-block">
        <h3>What it may mean</h3>

        <p>{verdict.explanation}</p>
      </div>

      <div className="evidence-block">
        <h3>What we can't conclude</h3>

        <p>This test does not show that the strategy will perform the same way in the future, prove that it is a better investment, or provide a buy or sell recommendation.</p>
      </div>
    </section>
  );
}

export default EvidenceSummary;
