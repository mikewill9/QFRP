import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="home-page page-width">
      <section className="hero">
        <div className="eyebrow">
          QUANTITATIVE RESEARCH • STATISTICAL TESTING
        </div>

        <h1>Good backtest, or just good luck?</h1>

        <p className="hero-copy">
          Fluke lets you explore quantitative factors and compare their observed
          performance with 2,000 reshuffles of the same factor. As you test more
          configurations, the app also shows why an apparently strong result can
          become less convincing.
        </p>

        <div className="hero-actions">
          <Link to="/analyzer" className="primary-button">
            Open the analyzer
          </Link>

          <a href="#how-it-works" className="secondary-button">
            How it works
          </a>
        </div>
      </section>

      <section className="purpose-grid" id="how-it-works">
        <article className="purpose-card">
          <span className="step-number">01</span>

          <h2>Choose a factor</h2>

          <p>
            Select one of the fixed factor and date-range combinations prepared
            by the research layer.
          </p>
        </article>

        <article className="purpose-card">
          <span className="step-number">02</span>

          <h2>Compare with chance</h2>

          <p>
            See where the real Sharpe ratio falls against the null distribution
            from reshuffled factor scores.
          </p>
        </article>

        <article className="purpose-card">
          <span className="step-number">03</span>

          <h2>Account for searching</h2>

          <p>
            Your session counter tracks unique configurations viewed so the
            verdict reflects multiple testing.
          </p>
        </article>
      </section>

      <section className="why-section">
        <div>
          <div className="eyebrow">WHY WE BUILT IT</div>

          <h2>
            Backtests can look persuasive even when the underlying signal is
            weak.
          </h2>
        </div>

        <p>
          Trying many ideas increases the chance that one looks impressive by
          accident. Fluke turns that problem into an interactive experience: the
          more configurations you inspect, the more context the app adds to the
          result you are seeing.
        </p>
      </section>
    </div>
  );
}

export default HomePage;
