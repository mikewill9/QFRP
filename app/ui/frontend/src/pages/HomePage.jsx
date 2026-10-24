import { Link } from "react-router-dom";

// Layout machinery kept from the previous home page (hero, three-card grid,
// closing section). Content is three blocks: the problem, who it's for,
// where to start, plus two calls to action.
function HomePage() {
  return (
    <div className="home-page page-width">
      <section className="hero">
        <div className="eyebrow">QUESTION THE BACKTEST</div>

        <h1>A strong-looking strategy can still be a fluke.</h1>

        <p className="hero-copy">Fluke checks whether a strategy's result is real or could just be luck. Fluke helps you investigate supported investing claims by comparing historical strategy results with randomized alternatives, other time periods, trading costs, and known limitations.</p>

        <div className="hero-actions">
          <Link to="/claims" className="primary-button">
            Explore a claim
          </Link>

          <Link to="/compare" className="secondary-button">
            Open the Comparison Lab
          </Link>
        </div>
      </section>

      <section className="purpose-grid" id="how-it-works">
        <article className="purpose-card">
          <span className="step-number">01</span>

          <h2>A convincing chart is only the beginning</h2>

          <p>A historical chart can look persuasive even when chance helped produce the result. Repeated searching and a favorable time period can make a weak idea look strong.</p>
        </article>

        <article className="purpose-card">
          <span className="step-number">02</span>

          <h2>For curious students and investment clubs</h2>

          <p>Fluke is for students and investment clubs who want to examine quantitative investing claims. You can explore the evidence without treating a backtest as a prediction.</p>
        </article>

        <article className="purpose-card">
          <span className="step-number">03</span>

          <h2>Start with a question</h2>

          <p>Choose a supported claim for a guided investigation. Then open the Comparison Lab to compare tested factors and ask your own questions.</p>
        </article>
      </section>

      <section className="why-section">
        <div>
          <div className="eyebrow">LOOK PAST THE CHART</div>

          <h2>Does the evidence hold up when you challenge it?</h2>
        </div>

        <p>The more configurations you inspect, the more chances you have to find an unusual result by luck. Fluke tracks unique factor-period combinations and adjusts the evidence for that search.</p>
      </section>
    </div>
  );
}

export default HomePage;
