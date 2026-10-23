# Fluke guide implementation

## Completed

Applied the 13 full frontend replacements and the EVIDENCE_LABELS block from
Fluke_Complete_Guide_With_Code(2).md to fluke-v3, based on commit 34442a5.
All 34 BP text tasks are filled; no `>>> BP` or `[BP` markers remain in
app/ui/frontend/src. Preserved the uploaded .gitignore and package-lock.json
changes. Documented this AI assistance in AI_USAGE.md.

Backend calculations, stored research results, claim definitions, styles,
API wiring, and Advanced Analyzer source remain unchanged.

## Verification

- npm ci: successful using the uploaded lockfile.
- npm run build: successful. Existing large Plotly bundle warning remains
  (about 4.98 MB JavaScript / 1.52 MB gzip); it is not a build failure.
- npm run lint: successful, no errors.
- Python test suite: 580 passed, six expected failures in
  tests/test_claims.py because the backend still contains the example claim.
- All 50 real factor-period results: API responses contain chart data and
  agree on the requested configuration count across analysis and verdict.
- Chromium browser: Home, eight-step Claim Explorer using the existing
  example claim, Comparison Lab, Across Periods, Advanced Analyzer, shared
  claims and comparison links, and legacy /analyzer redirect checked.
- Prediction required before revealing evidence. Configuration count grows
  from 1 to 5 on the challenge, then 6 on a comparison that adds one new
  configuration. Repeating configurations does not increase the count.
- Shared claim preserves sender k without modifying recipient session history.
- Mobile Home checked at 390 px width; no horizontal overflow.
- No JavaScript page errors in tested browser flows.
- Production Render API: health, claims, factors, periods, CORS preflight,
  five-period batch analysis, and analysis/verdict consistency passed.
  Responses allow the existing https://qfrp-frontend.vercel.app origin.
- Python test environment: 3.12.14. Node: 24.19.0.

## Still required before final submission

1. Michael must complete app/claims.py with verified real claims and mappings.
   The existing `example-claim` is still present. The claim-flow UI is tested,
   but its research content is not final.
2. Remove the intentional xfail in tests/test_claims.py once that content is
   finished, and rerun the tests.
3. Review the AI-drafted copy, including Michael's five designated areas:
   EVIDENCE_LABELS, ConfigCounter, PeriodConsistency's overlap note,
   ComparisonLabPage's interpretation, and SharedBanner.
4. Finish deployment and check the actual production pages against the new
   backend. A successful local build is not a production deployment.

## Hosting inspected during implementation

- Existing Render service: QFRP, https://qfrp.onrender.com.
- Changed Render from bp-frontend to fluke-v3. Backend commit 34442a5
  deployed successfully as dep-dbcnic3rjlhs73e3fkp0 (live).
- Existing Vercel project: qfrp-frontend, project
  prj_qPOzrzLc6M1WYBrwBCx9EdrPplAC, team team_UobBKnHKytbQPaMbqEg1O73Y.
- The Vercel integration denied access to the project's bob-and-friends scope
  (403). Its connection needs access to that existing team before deployment.
- GitHub writes were rejected by the connected integration (403). Local Git
  push also requires authentication. Frontend implementation is committed
  locally as 8456b52; it has NOT been pushed to GitHub or deployed to Vercel.
- No replacement project or paid service has been created.
- Revealing masked Render environment settings was rejected by automatic
  approval review. Existing values were preserved; no secrets were revealed.
  The deployed Python version could not be confirmed through that check.

## Using the source ZIP

The ZIP contains project source and stored results, without local credentials,
virtual environments, node_modules, or generated build output. These generated
folders can be recreated. Keep your original folder as a backup. The included frontend-changes.patch contains the complete implementation
relative to the uploaded source. In your original QFRP checkout, use
`git apply --check <path-to-frontend-changes.patch>` before applying it with
`git apply <path-to-frontend-changes.patch>`. This patch excludes your already
present .gitignore and package-lock edits. It does not change backend source.
Review and commit before pushing fluke-v3; do not replace your .git directory.

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -r app/ui/requirements.txt pytest httpx
.\.venv\Scripts\python.exe -m uvicorn app.ui.server:app --reload
```

In a second terminal:

```powershell
cd app\ui\frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. The existing vite.config.js proxies /api
to the backend at 127.0.0.1:8000. For Vercel, VITE_API_BASE_URL must be the
Render origin. FRONTEND_ORIGIN on Render must match the actual frontend origin.
