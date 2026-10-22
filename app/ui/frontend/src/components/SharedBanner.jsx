// Shown when the page was opened from a shared link. The view is adjusted
// for the SENDER's k, not the recipient's session.
function SharedBanner({ k }) {
  return (
    <div className="shared-banner" role="note">
      <strong>{k}</strong>

      <p>
        This shared view uses the adjustment for the {k} configurations in the sender&apos;s investigation. That is the sender&apos;s context, not your own browsing history.
      </p>
    </div>
  );
}

export default SharedBanner;
