import Plot from "react-plotly.js";
function PermutationHistogram({ result, formatNumber }) {
  return (
    <Plot
      data={[
        {
          // plot the 2000 shuffled gross Sharpe values.
          // These values show what performance looked like when
          // the factor ranking were randomly reshuffled.
          x: result.null_sharpes_gross,
          type: "histogram",
          name: "Reshuffled gross Sharpe",
          opacity: 0.9,
        },
      ]}
      layout={{
        autosize: true,
        height: 360,

        margin: {
          l: 55,
          r: 20,
          t: 20,
          b: 55,
        },

        xaxis: {
          title: "Gross Sharpe ratio",
          gridcolor: "#f3f4f6",
        },

        yaxis: {
          title: "Count",
          gridcolor: "#e5e7eb",
        },

        // Draw the real factor's gross Sharpe as a vertical line/
        // Gross Sharpe is used because the null distribution is also gross.
        shapes: [
          {
            type: "line",
            x0: result.sharpe_gross,
            x1: result.sharpe_gross,
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
            x: result.sharpe_gross,
            y: 1,
            yref: "paper",

            text: `Observed gross Sharpe: ${formatNumber(result.sharpe_gross)}`,

            showarrow: true,
            arrowhead: 2,
            ax: 45,
            ay: 35,
          },
        ],

        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",

        showlegend: true,
      }}
      config={{
        displayModeBar: false,
        responsive: true,
      }}
      useResizeHandler
      className="plot"
    />
  );
}

export default PermutationHistogram;
