import Plot from "react-plotly.js";
import {
  evidenceLabel,
  formatNumber,
  formatPercent,
  formatPValue,
} from "../analysisHelpers.js";

// The period whose dates contain every other period's dates is the "full
// period" (2005-2024). It overlaps the four five-year periods, so it is
// shown apart from them, never as a fifth independent check.
function splitFullPeriod(results) {
  const full = results.find((candidate) =>
    results.every(
      (other) =>
        other === candidate ||
        (candidate.analysis.start_date <= other.analysis.start_date &&
          candidate.analysis.end_date >= other.analysis.end_date),
    ),
  );

  if (!full || results.length === 1) {
    return { full: null, parts: results };
  }

  return { full, parts: results.filter((item) => item !== full) };
}

function PeriodRow({ item, isFull }) {
  return (
    <tr className={isFull ? "period-row-full" : undefined}>
      <th scope="row">
        {item.analysis.period_label}

        {isFull && <span className="period-full-tag">full period</span>}
      </th>

      <td>{formatPercent(item.analysis.total_return)}</td>

      <td>{formatNumber(item.analysis.sharpe_1x)}</td>

      <td>{formatPValue(item.analysis.adjusted_p_value)}</td>

      <td>{evidenceLabel(item.verdict)}</td>
    </tr>
  );
}

function PeriodConsistency({ results }) {
  const { full, parts } = splitFullPeriod(results);
  const k = results[0]?.analysis.k;

  return (
    <section className="period-consistency">
      <div className="table-scroll">
        <table className="period-table">
          <thead>
            <tr>
              <th scope="col">Period</th>

              <th scope="col">Total return</th>

              <th scope="col">Sharpe (1x costs)</th>

              <th scope="col">Adjusted p-value</th>

              <th scope="col">Evidence</th>
            </tr>
          </thead>

          <tbody>
            {parts.map((item) => (
              <PeriodRow key={item.period_key} item={item} isFull={false} />
            ))}
          </tbody>

          {full && (
            <tbody className="period-table-full">
              <PeriodRow item={full} isFull />
            </tbody>
          )}
        </table>
      </div>

      {full && (
        <p className="period-full-note">
          The full period overlaps the shorter periods, so these are sensitivity checks, not independent confirmations.
        </p>
      )}

      <p className="period-k-note">
        Every row is adjusted for the same {k} configurations. The raw p-value cutoff is shown in the configuration counter.
      </p>

      <Plot
        data={[
          {
            x: parts.map((item) => item.analysis.period_label),
            y: parts.map((item) => item.analysis.sharpe_1x),
            type: "bar",
            name: "Five-year periods",
          },
          ...(full
            ? [
                {
                  x: [`${full.analysis.period_label} (full period)`],
                  y: [full.analysis.sharpe_1x],
                  type: "bar",
                  name: "Full period",
                  marker: { color: "#94a3b8" },
                },
              ]
            : []),
        ]}
        layout={{
          autosize: true,
          height: 320,
          margin: { l: 55, r: 20, t: 20, b: 70 },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          font: {
            family: "Inter, ui-sans-serif, system-ui, sans-serif",
            color: "#374151",
          },
          showlegend: true,
          dragmode: false,
          legend: { orientation: "h", x: 0, y: 1.15 },
          xaxis: { title: "Period", fixedrange: true },
          yaxis: {
            title: "Sharpe (1x costs)",
            gridcolor: "#e5e7eb",
            zeroline: true,
            fixedrange: true,
          },
        }}
        config={{ displayModeBar: false, responsive: true, scrollZoom: false }}
        useResizeHandler
        className="plot compact-plot"
      />
    </section>
  );
}

export default PeriodConsistency;
