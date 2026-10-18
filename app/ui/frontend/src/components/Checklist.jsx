// The four questions the app teaches. Headings are fixed; BP writes the
// one-sentence prompts under each.
const QUESTIONS = [
  "Could this be luck?",
  "Does it hold up in other periods?",
  "What can the test still miss?",
  "How many things did I look at before I found this?",
];

function Checklist() {
  return (
    <ol className="checklist">
      {QUESTIONS.map((question, index) => (
        <li key={question} className="checklist-item">
          <span className="checklist-number">{index + 1}</span>

          <div>
            <h4>{question}</h4>

            {/* >>> BP: one sentence under each of the four fixed headings <<< */}
            <p>[BP: one sentence]</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

export default Checklist;
