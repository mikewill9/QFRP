import {
  evidenceLabel,
  formatNumber,
  formatPercent,
  formatPValue,
} from "../analysisHelpers.js";
import MetricHelp from "./MetricHelp.jsx";

// Two factors, same period, side by side. Contract v2 field names only.
// Each factor is tested against its own null; nothing here ranks A vs B.
const ROWS = [
  {
    label: "Total return",
    metric: "totalReturn",
    value: (item) => formatPercent(item.analysis.total_return),
  },
  {
    label: "Sharpe (1x costs)",
    metric: "sharpe",
    value: (item) => formatNumber(item.analysis.sharpe_1x),
  },
  {
    label: "Max drawdown",
    metric: "drawdown",
    value: (item) => formatPercent(item.analysis.max_drawdown),
  },
  {
    label: "Turnover",
    metric: "turnover",
    value: (item) => `${formatNumber(item.analysis.turnover)}x / year`,
  },
  {
    label: "Raw p-value",
    metric: "pValue",
    value: (item) => formatPValue(item.analysis.p_value),
  },
  {
    label: "Adjusted p-value",
    metric: "adjustedP",
    value: (item) => formatPValue(item.analysis.adjusted_p_value),
  },
  {
    label: "Evidence",
    metric: null,
    value: (item) => evidenceLabel(item.verdict),
  },
];

function ComparisonTable({ a, b }) {
  return (
    <div className="table-scroll">
      <table className="comparison-table">
        <thead>
          <tr>
            <th scope="col">Metric</th>

            <th scope="col">{a.analysis.factor_label}</th>

            <th scope="col">{b.analysis.factor_label}</th>
          </tr>
        </thead>

        <tbody>
          {ROWS.map((row) => (
            <tr key={row.label}>
              <th scope="row">
                <span className="metric-name">
                  {row.label}

                  {row.metric && <MetricHelp metric={row.metric} />}
                </span>
              </th>

              <td>{row.value(a)}</td>

              <td>{row.value(b)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ComparisonTable;
