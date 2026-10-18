// Shown when the page was opened from a shared link. The view is adjusted
// for the SENDER's k, not the recipient's session.
function SharedBanner({ k }) {
  return (
    <div className="shared-banner" role="note">
      <strong>{k}</strong>

      {/* >>> BP (Michael approves): one sentence: this view is adjusted for the k configurations the sender had looked at, not yours <<< */}
      <p>
        [BP (Michael approves): one sentence: this view is adjusted for the{" "}
        {k} configurations the sender had looked at, not yours]
      </p>
    </div>
  );
}

export default SharedBanner;
