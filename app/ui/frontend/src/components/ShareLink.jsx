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

      {/* >>> BP: the three button/status strings below <<< */}
      <button type="button" className="secondary-button" onClick={copy}>
        {copied ? "[BP: copied state]" : "[BP: copy button label]"}
      </button>

      {failed && (
        <span className="share-link-fallback">
          [BP: one line: copying failed, select the link and copy it yourself]
        </span>
      )}
    </div>
  );
}

export default ShareLink;
