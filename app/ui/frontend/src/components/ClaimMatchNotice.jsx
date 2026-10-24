// What Fluke actually tested versus what the online claim says. A "related"
// match is visually prominent: the evidence is about a cousin of the claim.
function ClaimMatchNotice({ claim }) {
  const related = claim.match_level === "related";

  return (
    <aside
      className={`claim-match-notice${related ? " claim-match-related" : ""}`}
    >
      <div className="claim-match-header">
        <span className={`match-badge match-${claim.match_level}`}>
          {related ? "CLOSE, NOT EXACT" : claim.match_level}
        </span>

        {related && (
          <p className="claim-match-warning">
            Fluke tested a similar strategy, not the exact one in this claim. Here's the difference.
          </p>
        )}
      </div>

      <dl className="claim-match-grid">
        <div>
          <dt>What Fluke tested</dt>

          <dd>{claim.what_fluke_tested}</dd>
        </div>

        <div>
          <dt>What it does not test</dt>

          <dd>{claim.what_it_does_not_test}</dd>
        </div>
      </dl>
    </aside>
  );
}

export default ClaimMatchNotice;
