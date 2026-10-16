import { NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import AnalyzerPage from "./pages/AnalyzerPage.jsx";

function App() {
  return (
    <div className="site-shell">
      <header className="topbar">
        <NavLink to="/" className="brand" aria-label="Fluke home">
          Fluke
        </NavLink>

        <nav className="nav-links" aria-label="Primary navigation">
          <NavLink to="/" end>
            Home
          </NavLink>

          <NavLink to="/analyzer">Analyzer</NavLink>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />

          <Route path="/analyzer" element={<AnalyzerPage />} />
        </Routes>
      </main>

      <footer className="footer">
        <span>Fluke</span>
        <span>Research output. Not investment advice.</span>
      </footer>
    </div>
  );
}

export default App;
