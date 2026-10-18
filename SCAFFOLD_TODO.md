# Fluke v3 scaffold: what the humans fill in

Scaffold commit: the single commit on fluke-v3 directly after f7d3aa9 (origin/bp-frontend); `git log --oneline f7d3aa9..fluke-v3 | tail -1` prints it  (everything after this commit on fluke-v3 is hand-written)
Find every hole:  grep -rn ">>> MICHAEL\|>>> BP" --include=*.py --include=*.js --include=*.jsx .

## Michael (backend), suggested order
1. app/claims.py:29         write CLAIMS (5-8 claims; one default period should be a convincing FAIL)
2. app/api.py:566           implement get_batch_analysis
3. tests/test_claims.py:18  delete the xfail line
4. tests/test_batch.py:16   delete the xfail line
Then: python3 -m pytest -q  (must be fully green), git push.

## BP (frontend), suggested order
1. src/analysisHelpers.js:93                EVIDENCE_LABELS (Michael approves)
2. src/components/ConfigCounter.jsx:25      counter copy (Michael approves)
3. src/App.jsx:41                           footer line
4. src/pages/HomePage.jsx:10                hero eyebrow, headline and intro. If a permutation count is shown anywhere, use analysis.n_permutations from the API rather than a hard-coded number.
5. src/pages/HomePage.jsx:32                block 1, the problem
6. src/pages/HomePage.jsx:41                block 2, who it's for
7. src/pages/HomePage.jsx:50                block 3, where to start
8. src/pages/HomePage.jsx:59                closing section eyebrow and heading
9. src/pages/ClaimExplorerPage.jsx:42       one short title per step, shown in each step card's header
10. src/pages/ClaimExplorerPage.jsx:57       the answer labels for the predict and reflect steps
11. src/pages/ClaimExplorerPage.jsx:304      page eyebrow, title and intro line
12. src/pages/ClaimExplorerPage.jsx:319      one line: this shared link points at a claim that no longer exists
13. src/pages/ClaimExplorerPage.jsx:329      choose-step body copy
14. src/pages/ClaimExplorerPage.jsx:348      context-step body copy
15. src/pages/ClaimExplorerPage.jsx:365      look-step body copy (no verdict language here; the user has not predicted yet)
16. src/pages/ClaimExplorerPage.jsx:416      predict-step question and helper copy
17. src/pages/ClaimExplorerPage.jsx:462      reveal-step body copy
18. src/pages/ClaimExplorerPage.jsx:476      histogram heading and one-sentence caption
19. src/pages/ClaimExplorerPage.jsx:547      challenge-step body copy: the counter just jumped
20. src/pages/ClaimExplorerPage.jsx:583      limits-step body copy
21. src/pages/ClaimExplorerPage.jsx:646      reflect-step body copy and the "you said X, now Y" sentence
22. src/components/ClaimMatchNotice.jsx:16   one sentence: this is related evidence, not a test of the exact claim
23. src/components/Checklist.jsx:20          one sentence under each of the four fixed headings
24. src/pages/ComparisonLabPage.jsx:211      one line: pick two different factors
25. src/pages/ComparisonLabPage.jsx:257      (Michael approves) two sentences (higher return, lower adjusted p) plus the fixed structural line; the sentences state facts, never that one factor "beat" the other
26. src/pages/ComparisonLabPage.jsx:437      one sentence introducing the across-periods table
27. src/pages/ComparisonLabPage.jsx:499      page eyebrow, title and intro line
28. src/components/PeriodConsistency.jsx:86  (Michael approves) one sentence: the full period overlaps the others so it is not an independent check
29. src/components/PeriodConsistency.jsx:94  one line: every row above is adjusted for the same k (interpolated); the pass bar itself is shown by ConfigCounter
30. src/components/EvidenceSummary.jsx:29    one or two sentences on what a PASS/FAIL here does not tell you
31. src/components/SharedBanner.jsx:8        (Michael approves) one sentence: this view is adjusted for the k configurations the sender had looked at, not yours
32. src/components/ShareLink.jsx:40          the three button/status strings below
33. src/components/MetricHelp.jsx:3          one plain-English sentence per metric (titles are the metric names already used in the Analyzer)
34. src/components/LoadingProgress.jsx:3     default loading message (pages may pass a more specific one)
Then: npm run build && npm run lint, test the flows in section "Run it locally".

## Run it locally
Terminal 1 (repo root):   python3 -m uvicorn app.ui.server:app --reload
Terminal 2 (app/ui/frontend):   npm run dev   → http://localhost:5173
Until Michael's stubs are filled, /api/batch returns 501 and the new pages show an error state. The old Analyzer under /compare → Advanced still works.

## Decisions I made
- A commit cannot contain its own hash, so "Scaffold commit" above names the base commit and the git command that prints the scaffold commit.
- tests/test_claims.py keeps `raises=NotImplementedError` like test_batch.py: a `written_claims()` helper in the test raises NotImplementedError while any claim string starts with `[MICHAEL`, because `list_claims()` itself must keep returning the example claim so that `/api/claims` answers 200 today.
- tests/test_server_batch.py tests the 501 mapping by monkeypatching `get_batch_analysis` to raise NotImplementedError, so the test keeps passing after Michael implements it; it also checks the 404/422 mapping and the response shape the same way.
- `POST /api/batch` deduplicates `configs` (first-seen order) before calling `get_batch_analysis`, so the response has one result per unique config and lines up whatever Michael decides about duplicates inside the function.
- httpx 0.28.1 was already installed in .venv, so app/ui/requirements.txt is unchanged.
- index.html already had `<title>Fluke</title>` and App.jsx already said "Fluke"; the hard-coded permutation count on the home page was "2,000" (not "1,000") and was removed, with the marker telling BP to use analysis.n_permutations.
- Labels the brief states verbatim (nav items, tab names, Start / Compare / "Does it hold up in other periods?", the Checklist headings, "What we found" / "What it may mean" / "What we can't conclude", "What this shows", "full period") are written as given. Metric names and table column headers reuse the wording BP already used in AnalyzerPage.jsx and are not marked. Everything else a user reads is a bracketed placeholder.
- One `>>> BP` marker comment may cover several bracketed strings directly beneath it (same pattern as the EVIDENCE_LABELS example in the brief); each bracketed string is still a hole.
- The "fixed structural line" in Compare (each factor against its own null, not a test of which is better) is a `BP (Michael approves)` placeholder whose marker says the meaning is fixed; BP picks the words.
- ConfigCounter takes an optional `previousK` prop; when it differs from `k` the old count is shown struck through with a short pulse animation, so the jump is obvious on the challenge step and after a second comparison.
- DualEquityChart accepts a missing `b` and draws one trace; the Claim Explorer's look step reuses it for the single strategy curve.
- PeriodConsistency finds the "full period" as the result whose dates contain every other result's dates, instead of hard-coding 2005_2024.
- The claims share link carries the sender's LATEST k (five after the challenge), so the recipient's bar matches what the sender saw last. Shared compare / periods links keep the k in the URL; if it is below the unique config count the backend's 422 message is shown.
- In the Claim Explorer the step chip is numeric ("3 / 8"); step titles and answer labels (Yes / No / Not sure, More / Less / About the same) are placeholders under one marker each.
- LoadingProgress has one default placeholder message; pages do not pass their own, to keep the number of markers small.
- `<AnalyzerPage embedded />` drops the whole page heading, including the old session pill; in the lab the ConfigCounter shows the count instead.
- The brief's prompt file and a `.claude/launch.json` used for local preview are listed in .git/info/exclude and not committed.
- On this machine Vite listens on the IPv6 localhost only; use http://localhost:5173 (not 127.0.0.1) in the browser.
