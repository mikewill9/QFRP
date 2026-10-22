// Shown while any fetch is in flight. One marker covers the default text.
function LoadingProgress({ message }) {
  const text = message ?? "Loading the historical results and statistical evidence...";

  return (
    <div className="loading-progress" role="status" aria-live="polite">
      <span className="loading-spinner" aria-hidden="true" />

      <p>{text}</p>
    </div>
  );
}

export default LoadingProgress;
