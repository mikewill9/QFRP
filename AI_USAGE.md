# AI Usage

## How we used AI

Michael Williams:
Michael wrote the core statistical functions by hand: the within-date shuffle, the permutation p-value, and the Bonferroni adjustment, plus their tests. Claude Code wrote larger infrastructure: the backtest pipeline module, the permutation runner, and the API layer. Michael directed each task, reviewed every change before committing, and rejected or corrected approaches when needed (for example, catching that the original shuffle test was biased by trading costs). Michael also used Claude in chat for design review, research on significance thresholds, and code skeletons he completed himself.

## Hand-written (Michael)

- permute_within_date + tests
- permutation_pvalue + tests
- bonferroni_adjust + tests

## AI-assisted (Claude Code, reviewed by Michael)

- Backtest pipeline extraction (pipeline.py)
- Permutation runner (run_permutations.py)
- Data extension to 2005, EA removal
- app/api.py, including reversed-verdict logic and roll_spread exclusion
- CONTRACT.md
- Shipping precomputed results
- High-precision rerun tooling: --config flag in run_permutations.py, compare_permutations.py, tests/test_run_permutations_cli.py
- PROJECT_HANDOFF.md update (post-Checkpoint 11 section)
- Pre-submission documentation cleanup and the README libraries list
- Fluke v3 scaffold (branch fluke-v3, the first commit after f7d3aa9 / origin/bp-frontend): component and page skeletons, routing, URL state, CSS, the /api/batch and /api/claims HTTP routes, structural tests and xfail test specs. The statistical function get_batch_analysis, claim mappings and descriptions (app/claims.py), and user-facing sentences were initially left as marked stubs. Michael subsequently implemented get_batch_analysis. The frontend wording was later AI-drafted and applied with ChatGPT/Codex as documented below; the claim content remains unfinished. See SCAFFOLD_TODO.md.
- app/claims.py: fixed typos and wording only; the survivorship sentence and all other claim text are Michael's
- User-test fixes (frontend): disabled Plotly zoom/pan, reworded the "related" claim badge and notice, added the "real or luck" intro sentence, and added MetricHelp to the reveal step's p-value labels (the new wording was supplied in the task, not written by Claude Code)

## Drafted with Claude (chat), edited and approved by Michael

- Verdict text, reversed-verdict text, survivorship note
- LIMITATIONS.md
- Planning and teamwork messages and outlines
- docs/amihud_survivorship.md: outline and some wording drafted with Claude chat; data checks run, claims verified, and final text written by Michael
- Running app directions in README.md

## Methodology decisions (made by Michael after reviewing options with Claude)

- Zero-cost significance test, Bonferroni correction, verdict cutoffs, dropping roll_spread, featuring the survivorship result

Bhavik's Work Documentation

## How we used AI — Bhavik Patel

Bhavik used ChatGPT for initial frontend scaffolding, selected component
implementations, and troubleshooting. He substantially rewrote most of
the supplied styles.css, retaining AI-generated typography, colors, and
some formatting. He also worked out the unique-configuration tracking
and results-context implementation, and handled most of the React page
structure, routing, Git workflow, and Vercel CLI deployment, with
targeted AI guidance when issues arose.

## Written or substantially reworked by Bhavik

- Most of styles.css, rewritten from an AI-generated starting point.
- Unique-configuration tracking: loadViewedConfigs(), sessionStorage
  persistence, and tracking distinct factor-period combinations
  in runAnalysis().
- Passing n_configs_viewed to the backend for statistical adjustment.
- Results context: displaying permutation counts, dropped draws,
  turnover, names_per_year, and limitations.
- React page organization and navigation using App.jsx, BrowserRouter,
  NavLink, and Routes.

## AI-assisted implementation

- Initial AnalyzerPage.jsx scaffolding and API-connected results layout.
- Plotly chart setup: mapping equity_curve and cost_stress data into
  chart traces and configuring shared chart settings through
  baseChartLayout().
- Reversed-FAIL presentation: conditional verdictClass styling and
  displaying reverse p-values when verdict.reversed is true.
- Typography, color choices, and some visual formatting retained
  in styles.css.
- Missing-value fixes for null_turnover and n_dropped.
- vercel.json fallback routing for opening and refreshing /analyzer
  directly.

## AI-assisted troubleshooting and deployment

- Targeted help with Git push rejection and a research-notebook
  merge conflict.
- Explanation of the Vite bundle-size warning and guidance on build
  and lint checks.
- Render setup guidance, including Python-version pinning and
  frontend-origin configuration.
- Targeted Vercel CLI help with project creation, root-directory
  prompts, and environment variables.
- Debugging guidance for empty factor and period dropdowns using
  browser Network and Console information.

## Work performed by Bhavik

- Substantially rewrote most of styles.css from an AI-generated starting
  point, retaining AI-generated typography, colors, and some formatting.

- Implemented or reworked unique-configuration tracking in AnalyzerPage.jsx:
  - Used loadViewedConfigs() to read previously viewed configurations
    from sessionStorage.
  - Identified each configuration using a factorKey::periodKey string.
  - Used viewedConfigs.includes(comboKey) in runAnalysis() to avoid
    counting the same combination more than once.
  - Saved the updated list under qfrp-viewed-configs.
  - Sent nextViewed.length as n_configs_viewed to /api/analysis and
    /api/verdict, leaving statistical calculations to the backend.

- Implemented or reworked the results-context displays:
  - Displayed n_permutations, n_dropped, turnover, and null_turnover
    in the Test details section.
  - Used Object.entries(result.names_per_year) to render yearly counts.
  - Used result.limitations.map() to display the backend's limitations.
  - Applied AI-suggested missing-value handling for null_turnover
    and n_dropped.

- Organized the React page structure and navigation:
  - Used BrowserRouter in main.jsx.
  - Defined the Home and Analyzer routes in App.jsx through Routes
    and Route.
  - Used NavLink for the main navigation and Link for the Home page's
    Open the analyzer button.
  - Connected the shared stylesheet through the styles.css import
    in main.jsx.

- Integrated the frontend with the backend:
  - Connected factor and period selections to API requests.
  - Passed the selected factor, period, and configuration count
    through the analysis workflow.
  - Checked that returned data appeared in the corresponding
    frontend sections.

- Performed local verification:
  - Ran npm run build and npm run lint.
  - Checked the Home and Analyzer pages in the browser.
  - Reported errors and applied targeted fixes with AI assistance.

- Managed the Git workflow:
  - Committed and pushed frontend changes on bp-frontend.
  - Incorporated Michael's master changes into the frontend branch.
  - Used AI guidance to resolve the unintended conflict in
    notebooks/momentum_research.ipynb by retaining Michael's version.

- Configured and deployed the application:
  - Linked the frontend directory to the qfrp-frontend Vercel project.
  - Configured VITE_API_BASE_URL to point to the Render backend.
  - Deployed the frontend using npx vercel --prod.
  - Configured Render's PYTHON_VERSION and FRONTEND_ORIGIN settings.
  - Checked the deployed pages and reported the empty-dropdown issue
    for further troubleshooting.

## Fluke v3 guide implementation — October 21, 2026

At Bhavik's request, ChatGPT/Codex applied the AI-drafted replacement code
from `Fluke_Complete_Guide_With_Code(2).md` to the uploaded fluke-v3
project at commit 34442a5. This covered 13 frontend files and the evidence
label mapping in analysisHelpers.js, completing the 34 BP text tasks.
The UI wording in these changes is AI-drafted, not claimed as hand-written
by Bhavik. Human review of the wording remains pending, including Michael's
review of the five statistical-copy areas identified in SCAFFOLD_TODO.md.

Codex compared the replacements with the existing source, installed
dependencies in a separate test environment, and ran the frontend build,
lint, existing Python tests, and checks against all 50 stored results.
Michael's statistical calculations, precomputed data, and unfinished claim
definitions were preserved. Detailed verification and release blockers are
recorded in IMPLEMENTATION_STATUS.md.
