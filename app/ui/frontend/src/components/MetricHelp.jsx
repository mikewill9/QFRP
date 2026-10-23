import { useEffect, useRef, useState } from "react";

const DEFINITIONS = {
  sharpe: {
    title: "Sharpe ratio (1x costs)",
    text: "Historical return relative to return variability, after the standard modeled trading costs.",
  },
  pValue: {
    title: "Raw p-value",
    text: "How unusual the observed gross Sharpe is compared with random reshuffles; a smaller value gives more evidence in the positive direction under this test.",
  },
  adjustedP: {
    title: "Adjusted p-value",
    text: "The raw p-value multiplied by the number of configurations examined, capped at 1, to account for multiple testing.",
  },
  drawdown: {
    title: "Max drawdown",
    text: "The largest decline from a previous portfolio peak during the tested period.",
  },
  turnover: {
    title: "Turnover",
    text: "How much the portfolio trades over time, shown here as a yearly multiple.",
  },
  totalReturn: {
    title: "Total return",
    text: "The total gain or loss over the tested period after the standard modeled trading costs.",
  },
};

// A small "?" button that opens a popover with the metric's definition.
function MetricHelp({ metric }) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const definition = DEFINITIONS[metric];

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function closeOnOutsideClick(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setOpen(false);
      }
    }

    function closeOnEscape(event) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);

    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  if (!definition) {
    return null;
  }

  return (
    <span className="metric-help" ref={wrapperRef}>
      <button
        type="button"
        className="metric-help-button"
        aria-label={`What is ${definition.title}?`}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        ?
      </button>

      {open && (
        <span className="metric-help-popover" role="dialog">
          <strong>{definition.title}</strong>

          <span>{definition.text}</span>
        </span>
      )}
    </span>
  );
}

export default MetricHelp;
