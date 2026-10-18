// One claim in the Claim Explorer's opening grid. Everything shown comes
// from app/claims.py (Michael's content).
function ClaimCard({ claim, onSelect }) {
  return (
    <button
      type="button"
      className={`claim-card claim-card-${claim.match_level}`}
      onClick={() => onSelect(claim)}
    >
      <span className="claim-card-label">{claim.short_label}</span>

      <span className="claim-card-question">{claim.question}</span>

      <span className="claim-card-meta">
        <span className={`match-badge match-${claim.match_level}`}>
          {claim.match_level}
        </span>
      </span>
    </button>
  );
}

export default ClaimCard;
