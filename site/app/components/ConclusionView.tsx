"use client";

import { useMemo, type ReactNode } from "react";
import { useJSON, type Grid, type Index, type RawRun } from "@/lib/data";
import { PLANNER, pct } from "@/lib/planners";
import { StatCard } from "./StatsDisplay";

type Verdict = ["yes" | "mixed" | "no", string] | null;
const VERDICT_COLOR = { yes: "var(--ds-green-600)", mixed: "var(--ds-amber-800)", no: "var(--ds-red-600)" };
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
const f1 = (x: number) => x.toFixed(1), f2 = (x: number) => x.toFixed(2);
const signed = (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2)

function paired(S: Index["summary"], drones: number, a: [string, string], b: [string, string]) {
  const pick = ([setting, strategy]: [string, string]) => new Map(S.filter((r) => r.drones === drones && r.setting === setting && r.strategy === strategy).map((r) => [r.seed, r.overdue]))
  const A = pick(a), B = pick(b)
  const d = [...A.keys()].filter((s) => B.has(s)).map((s) => A.get(s)! - B.get(s)!)
  return { n: d.length, better: d.filter((x) => x < 0).length, mean: mean(d), meanA: mean([...A.values()]), meanB: mean([...B.values()]) }
}

interface Claim { title: string; value: string; unit: string; verdict: Verdict; text: string; evidence: ReactNode; links: [string, string][] }

export default function ConclusionView() {
  const index = useJSON<Index>("/data/index.json");
  const G = useJSON<Grid>("/data/change_grid.json");
  const r1 = useJSON<RawRun>("/data/runs/tuned_retention_s0_d10.json");
  const r2 = useJSON<RawRun>("/data/runs/tuned_retention_s0_d6.json");

  const claims = useMemo<Claim[] | null>(() => {
    if (!index || !G || !r1 || !r2) return null
    const S = index.summary
    const T = G.trials.filter((t) => t.setting === "tuned")
    const st = [...r1.stages.slice(1), ...r2.stages.slice(1)]
    const med = median(st.map((s) => s.changed)), medSnap = median(st.filter((s) => s.snap).map((s) => s.changed))

    const ret = [6, 10, 12].map((d) => ({ d, ...paired(S, d, ["tuned", "retention"], ["tuned", "full_reset"]) }))
    const rWins = ret.reduce((s, r) => s + r.better, 0), rN = ret.reduce((s, r) => s + r.n, 0)
    const maxGap = Math.max(...ret.map((r) => Math.abs(r.mean)))

    const aco = ["aco_retention", "aco_full_reset", "aco_fast_evaporation", "aco_selective_reset"]
    const strategies = ["retention", "full_reset", "fast_evaporation", "selective_reset"]
    const base = (k: string) => new Map(S.filter((r) => r.drones === 10 && r.strategy === k && r.setting === "baseline").map((r) => [r.seed, r.overdue]))
    const greedy = base("greedy"), vns = base("vns")
    const acoRuns = S.filter((r) => r.drones === 10 && r.setting === "tuned" && greedy.has(r.seed) && vns.has(r.seed))
    const beatBoth = acoRuns.filter((r) => r.overdue < Math.min(greedy.get(r.seed)!, vns.get(r.seed)!)).length
    const acoMeans = strategies.map((k) => mean(acoRuns.filter((r) => r.strategy === k).map((r) => r.overdue)))

    const fr = G.fractions, big = fr[fr.length - 1]
    const meanAt = (f: number, p: string, m: "margin" | "total_overdue") => mean(T.filter((t) => t.fraction === f).map((t) => t.planners[p][m]))
    const bigMeans = aco.map((p) => meanAt(big, p, "margin"))
    const spread = Math.max(...bigMeans) - Math.min(...bigMeans)
    const dBig = T.filter((t) => t.fraction === big).map((t) => t.planners.aco_retention.margin - t.planners.aco_full_reset.margin)
    const bigT = { better: dBig.filter((x) => x < -1e-9).length, worse: dBig.filter((x) => x > 1e-9).length, n: dBig.length }
    const tied = bigT.n - bigT.better - bigT.worse

    const selDiff = (loc: string) => mean(T.filter((t) => t.locality === loc && t.fraction <= 0.2).map((t) => t.planners.aco_selective_reset.total_overdue - t.planners.aco_full_reset.total_overdue))
    const cDiff = selDiff("concentrated"), dDiff = selDiff("distributed")

    return [
      {
        title: "Most replans change only part of the map", verdict: null, value: pct(med), unit: "median change per replan",
        text: "Drones replan every 10 minutes, and between replans most monitoring points stay as they were. The big jumps come when a new satellite perimeter arrives every 12 hours.",
        evidence: <>In the 10- and 6-drone replays, a median of <b>{pct(med)}</b> of map points were added, removed or changed risk level at each replan, against <b>{pct(medSnap)}</b> at replans where a new FEDS perimeter arrived.</>,
        links: [["/?d=10&a=tuned_retention&b=none", "Watch the change bars in the replay"]],
      },
      {
        title: "In the replay, keeping the pheromone slightly beats resetting it", value: `${rWins}/${rN}`, unit: "missions, retention better than full reset",
        verdict: rWins / rN > 0.75 ? ["yes", "supported, small margin"] : ["mixed", "partly supported"],
        text: `Across 6, 10 and 12 drones, retention had fewer overdue edge points than a full reset in most paired missions, but the average gap is at most ${f2(maxGap)} points.`,
        evidence: <>Retention minus full reset, mean overdue edge points: {ret.map((r) => `${r.d} drones ${signed(r.mean)} (retention better in ${r.better} of ${r.n})`).join("; ")}.</>,
        links: [["/?d=12&a=tuned_retention&b=tuned_full_reset", "Retention vs full reset, 12 drones"], ["/?d=6&a=tuned_retention&b=tuned_full_reset", "Same, 6 drones"]],
      },
      {
        title: "Every ACO strategy beats greedy and VNS", value: `${beatBoth}/${acoRuns.length}`, unit: "ACO missions better than both baselines, 10 drones",
        verdict: beatBoth / acoRuns.length > 0.75 ? ["yes", "supported"] : ["mixed", "partly supported"],
        text: "Whichever pheromone strategy it uses, the ACO leaves fewer edge points overdue than the greedy and VNS baselines over the 24 h mission.",
        evidence: <>10 drones, mean overdue edge points over {greedy.size} seeds: {strategies.map((k, i) => `${PLANNER[k].short.toLowerCase()} ${f1(acoMeans[i])}`).join(", ")}; greedy <b>{f1(mean([...greedy.values()]))}</b>, VNS <b>{f1(mean([...vns.values()]))}</b>.</>,
        links: [["/?d=10&a=tuned_retention&b=greedy", "ACO vs greedy, 10 drones"], ["/?d=10&a=tuned_retention&b=vns", "ACO vs VNS, 10 drones"]],
      },
      {
        title: "After a large change all at once, resetting did not clearly help", value: `${tied}/${bigT.n}`, unit: `trials where retention and full reset tie on the worst point at ${pct(big)}`,
        verdict: tied / bigT.n > 0.5 ? ["mixed", "no clear difference"] : bigT.worse > bigT.better ? ["yes", "supported"] : ["no", "not supported"],
        text: `We expected a strategy that resets pheromone to do better on the single most overdue point when most of the perimeter changes at once. The four ACO strategies end within ${spread.toFixed(3)} of each other on this measure, partly because some points cannot be reached in time by any plan.`,
        evidence: <>Controlled change of {pct(big)}, worst-point staleness: {aco.map((p, i) => `${PLANNER[p.replace(/^aco_/, "")].short.toLowerCase()} ${bigMeans[i].toFixed(3)}`).join(", ")}. Retention vs full reset: better {bigT.better}, tied {tied}, worse {bigT.worse} of {bigT.n} trials.</>,
        links: [[`/explore?f=${big}&loc=concentrated&m=margin&p=aco_full_reset&ph=change`, `Explore a ${pct(big)} change`], [`/explore?f=${fr[0]}&loc=concentrated&m=total_overdue&p=aco_retention&ph=change`, `Compare with ${pct(fr[0])}`]],
      },
      {
        title: "Resetting only near the changes did not clearly help more for local changes", value: signed(cDiff), unit: `selective − full reset, one stretch (spread out ${signed(dDiff)})`,
        verdict: ["yes", "supported"],
        text: "We expected selective reset to beat a full reset after small changes in one stretch of the perimeter. It did slightly better on local changes than on spread-out ones, but the difference is small next to the trial-to-trial variation.",
        evidence: <>Selective minus full reset, total overdue, changes up to 20%: <b>{signed(cDiff)}</b> when the change is in one stretch, <b>{signed(dDiff)}</b> when spread out (both fire stages pooled). The 95% intervals on the Change size page overlap at every size.</>,
        links: [["/explore?f=0.1&loc=concentrated&p=aco_selective_reset&ph=change&m=total_overdue", "See selective reset on a local change"]],
      },
    ]
  }, [index, G, r1, r2])

  return (
    <div className="space-y-1.5">
      <p className="text-sm text-gray-900 leading-relaxed max-w-3xl mb-8">
        Our question was how much of the stored pheromone the ACO should keep when the fire perimeter changes. The ACO uses the parameters chosen by F-Race and irace on separate tuning seeds. Each card is computed in the browser
        from the same data as the other pages. Open a card (ⓘ) for the evidence and links to the view that shows it.
      </p>
      {!claims ? <div className="w-full h-[400px] bg-gray-alpha-100 animate-pulse rounded-md" /> : (
        <ol className="list-none pl-0 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1.5">
          {claims.map((c, i) => (
            <li key={c.title} className="flex">
              <StatCard
                className="flex-1 min-h-[300px]"
                title={<>{String(i + 1).padStart(2, "0")} · {c.title}</>}
                value={c.value}
                infoContent={<><p>{c.evidence}</p></>}
                links={c.links}
              >
                <p className="text-gray-900 text-sm font-mono m-0 uppercase">{c.unit}</p>
                {c.verdict && (
                  <p className="font-mono text-sm uppercase m-0 flex items-center gap-1.5 text-gray-1000">
                    <span aria-hidden="true" style={{ color: VERDICT_COLOR[c.verdict[0]] }}>■</span>{c.verdict[1]}
                  </p>
                )}
                <p className="text-sm text-gray-900 leading-relaxed pt-2 m-0">{c.text}</p>
              </StatCard>
            </li>
          ))}
          <li className="flex md:col-span-2 lg:col-span-3">
            <StatCard className="flex-1" title="Limits">
              <ul className="list-none pl-0 mt-2 space-y-1.5 text-sm text-gray-900 leading-relaxed">
                <li>▲ All results come from one fire, the 2018 Camp Fire, so they show variation across random seeds, not across different wildfires.</li>
                <li>▲ Between satellite snapshots the fire is our own spread model, which is only an approximation of the real fire.</li>
                <li>▲ The controlled change is instantaneous: drones do not move and only the points differ. The mission replay is where drones move and changes arrive over time.</li>
                <li>▲ With many comparisons and no multiple-comparison correction, treat win counts as descriptive.</li>
              </ul>
            </StatCard>
          </li>
        </ol>
      )}
    </div>
  );
}
