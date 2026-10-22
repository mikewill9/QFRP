import { formatPValue } from "../analysisHelpers.js";

// THE component that makes the session counter visible. Rendered wherever k
// changes. `passBelow` comes from the batch response; the one division below
// is the only statistic the frontend computes, and it is for display only.
// `previousK` is optional: when it differs from `k`, the jump is highlighted.
function ConfigCounter({ k, passBelow, previousK = null }) {
  const bar = k > 0 && passBelow != null ? passBelow / k : null;
  const jumped = previousK != null && previousK !== k;

  return (
    <aside
      className={`config-counter${jumped ? " config-counter-jumped" : ""}`}
      aria-live="polite"
    >
      <div className="config-counter-numbers">
        {jumped && (
          <span className="config-counter-previous" aria-label="previous count">
            {previousK}
          </span>
        )}

        <strong className="config-counter-k">{k}</strong>

        <span className="config-counter-unit">
          configurations in this investigation
        </span>
      </div>

      <div className="config-counter-bar">
        <span>Raw p-value cutoff for positive evidence</span>

        <strong>{bar == null ? "N/A" : formatPValue(bar)}</strong>
      </div>

      <p className="config-counter-copy">
        This view accounts for {k} unique factor-period combinations. Looking at more configurations gives more chances to find an unusual result by luck, so positive evidence must clear a lower raw p-value cutoff.
      </p>
    </aside>
  );
}

export default ConfigCounter;
