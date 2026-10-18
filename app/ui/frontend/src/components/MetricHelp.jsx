import { useEffect, useRef, useState } from "react";

// >>> BP: one plain-English sentence per metric (titles are the metric names already used in the Analyzer)
const DEFINITIONS = {
  sharpe: {
    title: "Sharpe ratio (1x costs)",
    text: "[BP: one plain-English sentence]",
  },
  pValue: {
    title: "Raw p-value",
    text: "[BP: one plain-English sentence]",
  },
  adjustedP: {
    title: "Adjusted p-value",
    text: "[BP: one plain-English sentence]",
  },
  drawdown: {
    title: "Max drawdown",
    text: "[BP: one plain-English sentence]",
  },
  turnover: {
    title: "Turnover",
    text: "[BP: one plain-English sentence]",
  },
  totalReturn: {
    title: "Total return",
    text: "[BP: one plain-English sentence]",
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
