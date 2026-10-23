import { useEffect, useState } from "react";

// Copy-to-clipboard with an inline "copied" state. No alert().
function ShareLink({ url }) {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!copied) {
      return undefined;
    }

    const timer = setTimeout(() => setCopied(false), 2000);

    return () => clearTimeout(timer);
  }, [copied]);

  async function copy() {
    setFailed(false);

    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
    } catch {
      setFailed(true);
    }
  }

  return (
    <div className="share-link">
      <input
        className="share-link-input"
        type="text"
        readOnly
        value={url}
        onFocus={(event) => event.target.select()}
        aria-label="Shareable link"
      />

      <button type="button" className="secondary-button" onClick={copy}>
        {copied ? "Link copied" : "Share evidence"}
      </button>

      {failed && (
        <span className="share-link-fallback">
          Could not copy automatically. Select the displayed link and copy it.
        </span>
      )}
    </div>
  );
}

export default ShareLink;
