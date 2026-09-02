// tuner-routes.ts — pure hash-route parsing for the tuner UI. The URL is
// the single source of truth for which view is open (fleet, launch form, or
// one run's overview / science / evidence). No DOM access here — `TunerApp`
// wires `parseTunerHash(location.hash)` to a signal and calls `tunerHash()`
// to navigate.

export type RunTab = "overview" | "science" | "evidence";

export type TunerRoute =
  | { view: "fleet" }
  | { view: "launch" }
  | { view: "run"; runId: string; tab: RunTab; candidate?: string };

const RUN_TABS: RunTab[] = ["overview", "science", "evidence"];

/** Parse a `window.location.hash` (`"#/tuner/..."`, or `""`). Anything that
 * isn't a recognised tuner route falls back to the fleet. */
export function parseTunerHash(hash: string): TunerRoute {
  const raw = hash.replace(/^#/, "").replace(/^\/+/, "");
  const [path = "", queryString] = raw.split("?", 2);
  const parts = path.split("/").filter(Boolean);
  if (parts[0] !== "tuner") return { view: "fleet" };
  if (parts[1] === "launch") return { view: "launch" };
  if (parts[1] === "run" && parts[2]) {
    const runId = decodeURIComponent(parts[2]);
    const tab = RUN_TABS.includes(parts[3] as RunTab) ? (parts[3] as RunTab) : "overview";
    const candidate = new URLSearchParams(queryString ?? "").get("candidate");
    return candidate ? { view: "run", runId, tab, candidate } : { view: "run", runId, tab };
  }
  return { view: "fleet" };
}

export function tunerHash(route: TunerRoute): string {
  switch (route.view) {
    case "fleet":
      return "#/tuner";
    case "launch":
      return "#/tuner/launch";
    case "run": {
      let base = `#/tuner/run/${encodeURIComponent(route.runId)}`;
      if (route.tab !== "overview") base = `${base}/${route.tab}`;
      return route.candidate ? `${base}?candidate=${encodeURIComponent(route.candidate)}` : base;
    }
  }
}
