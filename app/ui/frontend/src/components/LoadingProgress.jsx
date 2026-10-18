// Shown while any fetch is in flight. One marker covers the default text.
function LoadingProgress({ message }) {
  // >>> BP: default loading message (pages may pass a more specific one)
  const text = message ?? "[BP: loading message]";

  return (
    <div className="loading-progress" role="status" aria-live="polite">
      <span className="loading-spinner" aria-hidden="true" />

      <p>{text}</p>
    </div>
  );
}

export default LoadingProgress;
