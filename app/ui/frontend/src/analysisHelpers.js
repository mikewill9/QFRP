// Shared helpers for the Fluke pages: the session config counter, the batch
// and claims endpoints, display formatting, and shareable-link URL state.
//
// No statistics live here. The only arithmetic the frontend is allowed is
// `passBelow / k` for display (see ConfigCounter); everything else comes
// from the backend.

import { apiFetch } from "./api.js";

// MUST stay exactly this string: AnalyzerPage.jsx uses it too, and the
// counter has to be shared across every page in the same browser tab.
export const VIEWED_KEY = "qfrp-viewed-configs";

export function loadViewedConfigs() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(VIEWED_KEY) ?? "[]");

    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// Same format AnalyzerPage uses, so one config viewed anywhere counts once.
export function makeConfigKey(factorKey, periodKey) {
  return `${factorKey}::${periodKey}`;
}

// Adds every key not already in the session set and returns the new,
// deduplicated array. Call this BEFORE the fetch so that `.length` is the
// k every result on the screen is adjusted for.
export function addViewedConfigs(keys) {
  const next = [...loadViewedConfigs()];

  for (const key of keys) {
    if (!next.includes(key)) {
      next.push(key);
    }
  }

  try {
    sessionStorage.setItem(VIEWED_KEY, JSON.stringify(next));
  } catch {
    // Session storage can be unavailable (private mode); the counter then
    // lives only in React state for this page.
  }

  return next;
}

// Shown while Michael's get_batch_analysis stub is empty and /api/batch
// answers 501. Developer-facing: it tells BP the error is not his bug.
export const BATCH_NOT_IMPLEMENTED_MESSAGE =
  "Backend batch endpoint not implemented yet (/api/batch returned 501: " +
  "get_batch_analysis in app/api.py is still a stub for Michael). " +
  "This is not a frontend bug.";

export async function fetchBatch(configs, nConfigsViewed) {
  try {
    return await apiFetch("/api/batch", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        configs,
        n_configs_viewed: nConfigsViewed,
      }),
    });
  } catch (error) {
    if (/not implemented yet/i.test(error.message ?? "")) {
      throw new Error(BATCH_NOT_IMPLEMENTED_MESSAGE, { cause: error });
    }

    throw error;
  }
}

export function fetchClaims() {
  return apiFetch("/api/claims");
}

export function fetchFactors() {
  return apiFetch("/api/factors");
}

export function fetchPeriods() {
  return apiFetch("/api/periods");
}

// >>> BP (Michael approves): the four user-facing labels. Keep four; INCONCLUSIVE must stay distinct from FAIL.
export const EVIDENCE_LABELS = {
  PASS: "[BP: label for PASS]",
  INCONCLUSIVE: "[BP: label for INCONCLUSIVE]",
  FAIL: "[BP: label for FAIL]",
  FAIL_REVERSED: "[BP: label for FAIL when reversed === true]",
};

// One of the four EVIDENCE_LABELS keys, for styling and lookups.
export function evidenceKey(verdict) {
  if (!verdict) {
    return "";
  }

  if (verdict.label === "FAIL" && verdict.reversed) {
    return "FAIL_REVERSED";
  }

  return verdict.label;
}

export function evidenceLabel(verdict) {
  const key = evidenceKey(verdict);

  if (key === "PASS") {
    return EVIDENCE_LABELS.PASS;
  }

  if (key === "INCONCLUSIVE") {
    return EVIDENCE_LABELS.INCONCLUSIVE;
  }

  if (key === "FAIL_REVERSED") {
    return EVIDENCE_LABELS.FAIL_REVERSED;
  }

  return EVIDENCE_LABELS.FAIL;
}

export function formatPercent(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
}

export function formatNumber(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "N/A";
  }

  return Number(value).toFixed(digits);
}

// Display rule set by Michael: p-values as a percent to 3 significant
// figures, and "<0.1%" below 0.001.
export function formatPValue(value) {
  if (value == null || value === "") {
    return "N/A";
  }

  const number = Number(value);

  if (!Number.isFinite(number) || number < 0 || number > 1) {
    return "N/A";
  }

  if (number < 0.001) {
    return "<0.1%";
  }

  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumSignificantDigits: 3,
  }).format(number);
}

// URL state for shareable links. `k` is the sender's config count; the
// recipient's view is adjusted for it (SharedBanner says so).
export function readSharedParams(searchParams) {
  const rawK = Number.parseInt(searchParams.get("k") ?? "", 10);

  return {
    shared: searchParams.get("shared") === "1",
    k: Number.isInteger(rawK) && rawK >= 1 ? rawK : null,
    claim: searchParams.get("claim") ?? "",
    period: searchParams.get("period") ?? "",
    a: searchParams.get("a") ?? "",
    b: searchParams.get("b") ?? "",
    factor: searchParams.get("factor") ?? "",
    tab: searchParams.get("tab") ?? "",
  };
}

export function buildShareUrl(pathname, params) {
  const query = new URLSearchParams();

  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") {
      query.set(key, String(value));
    }
  }

  const origin =
    typeof window === "undefined" ? "" : window.location.origin;

  return `${origin}${pathname}?${query.toString()}`;
}
