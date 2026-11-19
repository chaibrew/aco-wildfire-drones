/* Planner names and colours. Colours are CSS variables (app/globals.css) so they switch with light / dark mode. */

export const STRATEGIES = ["retention", "full_reset", "fast_evaporation", "selective_reset"] as const

export const PLANNER: Record<string, { label: string; short: string; code: string; color: string }> = {
  retention: { label: "Keep pheromone (retention)", short: "Retention", code: "RET", color: "var(--p-retention)" },
  full_reset: { label: "Reset all pheromone (full reset)", short: "Full reset", code: "RST", color: "var(--p-full_reset)" },
  fast_evaporation: { label: "Fast evaporation", short: "Fast evap.", code: "EVP", color: "var(--p-fast_evaporation)" },
  selective_reset: { label: "Reset near changes (selective)", short: "Selective", code: "SEL", color: "var(--p-selective_reset)" },
  greedy: { label: "Greedy baseline", short: "Greedy", code: "GRD", color: "var(--p-greedy)" },
  vns: { label: "VNS baseline", short: "VNS", code: "VNS", color: "var(--p-vns)" },
}
export const SETTING: Record<string, string> = { tuned: "ACO", baseline: "" }
/** The site shows the tuned ACO only (parameters chosen by F-Race / irace); it is labelled plain "ACO". */
export const SHOWN_SETTINGS = ["tuned", "baseline"]

/** Planner key: "greedy", "vns" or "<setting>_<strategy>" (e.g. "tuned_full_reset"). */
export const strategyOf = (key: string) => (PLANNER[key] ? key : key.split("_").slice(1).join("_"))
export const settingOf = (key: string) => (PLANNER[key] ? "baseline" : key.split("_")[0])
export const plannerLabel = (key: string) =>
  PLANNER[key] ? PLANNER[key].label : `ACO · ${PLANNER[strategyOf(key)].label}`
export const plannerShort = (key: string) =>
  PLANNER[key] ? PLANNER[key].short : `ACO · ${PLANNER[strategyOf(key)].short}`
export const plannerColor = (key: string) => PLANNER[strategyOf(key)].color

/** Staleness = time since last visit / revisit period. Points stay grey until they need attention (≥ 0.75), above 1 they are overdue. */
export const STALE_BINS = [
  { max: 0.75, color: "var(--st-0)", label: "fine" },
  { max: 1, color: "var(--st-1)", label: "due soon" },
  { max: 2, color: "var(--st-2)", label: "overdue" },
  { max: Infinity, color: "var(--st-3)", label: "missed (> 2× period)" },
] as const
export const staleColor = (s: number) => STALE_BINS.find((b) => s < b.max)!.color

export const SIDE_COLOR = { a: "var(--side-a)", b: "var(--side-b)" } as const

export function hhmm(sec: number) {
  const m = Math.round(sec / 60)
  return `+${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`
}
export const pct = (f: number) => `${Math.round(f * 100)}%`
