// The four questions the app teaches. Headings are fixed; BP writes the
// one-sentence prompts under each.
const QUESTIONS = [
  "Could this be luck?",
  "Does it hold up in other periods?",
  "What can the test still miss?",
  "How many things did I look at before I found this?",
];

const PROMPTS = [
  "Compare the observed result with random reshuffles before treating it as evidence.",
  "Check other periods, remembering that overlapping periods are not independent confirmations.",
  "Review trading costs, data limitations, and the gap between the claim and the strategy tested.",
  "Use the configuration count to account for the opportunities you had to find an unusual result.",
];

function Checklist() {
  return (
    <ol className="checklist">
      {QUESTIONS.map((question, index) => (
        <li key={question} className="checklist-item">
          <span className="checklist-number">{index + 1}</span>

          <div>
            <h4>{question}</h4>

            <p>{PROMPTS[index]}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

export default Checklist;
