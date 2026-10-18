import Plot from "react-plotly.js";

// One Plotly chart with one or two `equity_curve` traces. No benchmark:
// it was removed in contract v2. `b` is optional so the same chart serves
// the Claim Explorer's single-strategy "look" step.
function DualEquityChart({ a, b = null }) {
  const traces = [a, b]
    .filter(Boolean)
    .map((item) => ({
      x: item.analysis.equity_curve.map((point) => point.date),
      y: item.analysis.equity_curve.map((point) => point.strategy),
      type: "scatter",
      mode: "lines",
      name: item.analysis.factor_label,
      line: { width: 3 },
    }));

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 360,
        margin: { l: 55, r: 20, t: 20, b: 55 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {
          family: "Inter, ui-sans-serif, system-ui, sans-serif",
          color: "#374151",
        },
        hovermode: "x unified",
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12 },
        yaxis: { title: "Growth of $1", gridcolor: "#e5e7eb" },
        xaxis: { gridcolor: "#f3f4f6" },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="plot"
    />
  );
}

export default DualEquityChart;
