import Plot from "react-plotly.js";
function PermutationHistogram({ result, formatNumber }) {
  return (
    <Plot
      data={
        [
          //histogram data will follow ...
        ]
      }
      layout={
        {
          // axes
          // vertical line
          // annotation
        }
      }
      config={{
        displayModeBar: false,
        responsive: true,
      }}
      useResizedHandler
      classname="plot"
    />
  );
}

export default PermutationHistogram;
