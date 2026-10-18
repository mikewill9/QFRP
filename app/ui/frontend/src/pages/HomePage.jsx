import { Link } from "react-router-dom";

// Layout machinery kept from the previous home page (hero, three-card grid,
// closing section). Content is three blocks: the problem, who it's for,
// where to start, plus two calls to action.
function HomePage() {
  return (
    <div className="home-page page-width">
      <section className="hero">
        {/* >>> BP: hero eyebrow, headline and intro. If a permutation count is shown anywhere, use analysis.n_permutations from the API rather than a hard-coded number. <<< */}
        <div className="eyebrow">[BP: HERO EYEBROW]</div>

        <h1>[BP: headline: good backtest, or just good luck?]</h1>

        <p className="hero-copy">[BP: two or three sentences: what Fluke is]</p>

        <div className="hero-actions">
          <Link to="/claims" className="primary-button">
            [BP: CTA: explore a claim]
          </Link>

          <Link to="/compare" className="secondary-button">
            [BP: CTA: open the comparison lab]
          </Link>
        </div>
      </section>

      <section className="purpose-grid" id="how-it-works">
        <article className="purpose-card">
          <span className="step-number">01</span>

          {/* >>> BP: block 1, the problem <<< */}
          <h2>[BP: heading: the problem]</h2>

          <p>[BP: two sentences: why a strong-looking backtest can be luck]</p>
        </article>

        <article className="purpose-card">
          <span className="step-number">02</span>

          {/* >>> BP: block 2, who it's for <<< */}
          <h2>[BP: heading: who it's for]</h2>

          <p>[BP: two sentences: young investors, clubs, classrooms]</p>
        </article>

        <article className="purpose-card">
          <span className="step-number">03</span>

          {/* >>> BP: block 3, where to start <<< */}
          <h2>[BP: heading: where to start]</h2>

          <p>[BP: two sentences: start with a claim, then ask your own questions in the lab]</p>
        </article>
      </section>

      <section className="why-section">
        <div>
          {/* >>> BP: closing section eyebrow and heading <<< */}
          <div className="eyebrow">[BP: WHY WE BUILT IT]</div>

          <h2>[BP: one-line heading on the counter idea]</h2>
        </div>

        <p>[BP: two or three sentences: the more you look, the higher the bar]</p>
      </section>
    </div>
  );
}

export default HomePage;
