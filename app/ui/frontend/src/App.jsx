import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import ClaimExplorerPage from "./pages/ClaimExplorerPage.jsx";
import ComparisonLabPage from "./pages/ComparisonLabPage.jsx";

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

          <NavLink to="/claims">Explore Claims</NavLink>

          <NavLink to="/compare">Comparison Lab</NavLink>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />

          <Route path="/claims" element={<ClaimExplorerPage />} />

          <Route path="/compare" element={<ComparisonLabPage />} />

          {/* The old Analyzer lives on under Comparison Lab → Advanced. */}
          <Route path="/analyzer" element={<Navigate to="/compare" replace />} />
        </Routes>
      </main>

      <footer className="footer">
        <span>Fluke</span>

        <span>Research and education only. Not investment advice.</span>
      </footer>
    </div>
  );
}

export default App;
