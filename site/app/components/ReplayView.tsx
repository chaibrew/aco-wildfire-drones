"use client";

import { memo, useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { decodeRun, dronePos, getJSON, readQuery, stageAt, stagePheromone, stagePoints, useJSON, writeQuery, type Index, type Iso, type RawRun, type Run, type XY } from "@/lib/data";
import { PLANNER, SHOWN_SETTINGS, SIDE_COLOR, STALE_BINS, hhmm, plannerColor, plannerLabel, plannerShort, settingOf, staleColor, strategyOf } from "@/lib/planners";
import { CountLine, MetricRow, StatCard } from "./StatsDisplay";
import { DroneIcon, DroneSprite, FireMap, LegendItem, Ln, Pixel, Sq, heading, polyline, rotorFrame, type Proj, type Tip } from "./FireMap";
import { Field, Seg, Select, Toggle } from "./Controls";

const SPEEDS = [{ value: 2, label: "2 min/s" }, { value: 10, label: "10 min/s" }, { value: 30, label: "30 min/s" }, { value: 60, label: "1 h/s" }, { value: 120, label: "2 h/s" }];
const DAY = 86400;
/** The replay shows one recorded mission (seed 0); the means card still averages all 10 seeds. */
const SEED = 0;
const CENTER: XY = [-0.9, -2];
const SPAN = 27;
const AXIS = { fill: "var(--ds-gray-900)", fontSize: 11, fontFamily: "var(--font-geist-mono)" };

/* Mission clock shared by the maps (redrawn every frame) and the rest of the page (updated ~10 times a second). */
function createClock(t0: number) {
  let t = t0;
  const subs = new Set<() => void>();
  return {
    get: () => t,
    set: (v: number) => { t = v; subs.forEach((f) => f()); },
    subscribe: (f: () => void) => { subs.add(f); return () => { subs.delete(f); }; },
  };
}
type Clock = ReturnType<typeof createClock>;

/** Compact planner name for the key rows. */
const compact = (k: string) => PLANNER[strategyOf(k)].short;
const order = (k: string) => (PLANNER[k] ? 1 : 0) * 10 + Object.keys(PLANNER).indexOf(strategyOf(k));

type PhMode = "on" | "off";
const PH_MODES = [{ value: "on" as const, label: "On" }, { value: "off" as const, label: "Off" }];
const isACO = (run: Run) => run.stages.some((s) => s.pc);

/** Pheromone edges for the current stage, drawn under the points: wider and darker = more pheromone relative to tau0. */
function PheromoneLayer({ run, k, mode, color, proj }: { run: Run; k: number; mode: PhMode; color: string; proj: Proj }) {
  const edges = useMemo(() => (mode === "off" ? null : stagePheromone(run, k)), [run, k, mode]);
  if (!edges?.length) return null;
  const top = Math.max(...edges.map((e) => Math.log(e[2])));
  return (
    <g aria-hidden="true">
      {edges.map(([p, q, r], i) => {
        const f = top > 0 ? Math.log(r) / top : 1;
        const [x1, y1] = proj.p(p), [x2, y2] = proj.p(q);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2.5 + 5 * f} strokeOpacity={0.35 + 0.55 * f} strokeLinecap="round" />;
      })}
    </g>
  );
}

/** Direction of travel at time t: the last movement before t, or the next one while hovering over a point. */
function droneHeading(trip: Run["trips"][number], t: number) {
  const now = dronePos(trip, t);
  if (!now) return 0;
  for (const dt of [-30, -90, 30, 90, -300, 300]) {
    const o = dronePos(trip, t + dt);
    if (!o) continue;
    const dx = dt < 0 ? now[0] - o[0] : o[0] - now[0], dy = dt < 0 ? now[1] - o[1] : o[1] - now[1];
    if (Math.hypot(dx, dy) > 0.02) return heading(dx, dy);
  }
  return 0;
}

const MissionLayer = memo(function MissionLayer({ index, run, fire, color, clock, proj, ph, onTip }: {
  index: Index; run: Run; fire: Iso | null; color: string; clock: Clock; proj: Proj; ph: PhMode; onTip: (t: Tip | null) => void;
}) {
  const t = useSyncExternalStore(clock.subscribe, clock.get, clock.get);
  const h = t / 3600, isoNow = Math.floor(h);
  const k = stageAt(run, t);
  const pts = stagePoints(run, k);
  const feds = index.perimeters.filter((p) => p.h <= h + 1e-6);
  const lastFeds = feds.length ? Math.max(...feds.map((p) => p.h)) : -1;
  const px = 0.42 * proj.k;
  return (
    <g>
      {fire?.filter((d) => d.h < isoNow).map((d, i) => (
        <polyline key={`iso${i}`} points={polyline(proj, d.path)} fill="none" stroke="var(--ds-gray-alpha-400)" strokeWidth={1.2} />
      ))}
      {feds.map((d, i) => (
        <polygon key={`feds${d.h}-${i}`} points={polyline(proj, d.ring)} fill={d.h === lastFeds ? "var(--ds-gray-alpha-100)" : "none"}
          stroke={d.h === lastFeds ? "var(--ds-gray-1000)" : "var(--ds-gray-alpha-600)"} strokeWidth={d.h === lastFeds ? 3 : 1.5} strokeDasharray={d.h === lastFeds ? undefined : "6 6"} />
      ))}
      {fire?.filter((d) => d.h === isoNow).map((d, i) => (
        <polyline key={`front${i}`} points={polyline(proj, d.path)} fill="none" stroke="var(--fire)" strokeWidth={3} />
      ))}
      <PheromoneLayer run={run} k={k} mode={ph} color={color} proj={proj} />
      {pts.map((q, i) => {
        const [x, y] = proj.p(q.p);
        return (
          <Pixel key={i} x={x} y={y} size={q.s > 2 ? px * 1.15 : px} fill={staleColor(q.s)} pulse={q.s > 1}
            onEnter={() => onTip({ x, y, content: <><span style={{ color: staleColor(q.s) }}>■</span> STALENESS {q.s.toFixed(2)} · {q.s > 2 ? "MISSED" : q.s > 1 ? "OVERDUE" : q.s >= 0.75 ? "DUE SOON" : "FINE"}</> })}
            onLeave={() => onTip(null)} />
        );
      })}
      {run.trips.map((tr, i) => {
        const d = dronePos(tr, t);
        if (!d) return null;
        const [x, y] = proj.p(d);
        return <DroneSprite key={`d${i}`} x={x} y={y} size={px * 4.6} angle={droneHeading(tr, t)} frame={rotorFrame(i)} ring={color} />;
      })}
    </g>
  );
});

export default function ReplayView() {
  const params = useMemo(() => readQuery(), []);
  const index = useJSON<Index>("/data/index.json");
  const [drones, setDrones] = useState(+(params.get("d") ?? 10));
  const seed = SEED;
  // older links may name untuned planners; the site shows the tuned ACO only
  const asShown = (k: string) => k.replace(/^untuned_/, "tuned_");
  const [a, setA] = useState(asShown(params.get("a") ?? "tuned_retention"));
  const [b, setB] = useState(asShown(params.get("b") && params.get("b") !== "none" ? params.get("b")! : "tuned_full_reset"));
  const [compare, setCompare] = useState(params.get("b") !== "none");
  const [speed, setSpeed] = useState(10);
  const [playing, setPlaying] = useState(false);
  const clock = useMemo(() => createClock(+(params.get("t") ?? 0) * 3600), [params]);
  const [t, setT] = useState(clock.get());
  const [runs, setRuns] = useState<{ a: Run; b: Run | null; fire: Iso } | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [mobileSide, setMobileSide] = useState<"a" | "b">("a");
  const [ph, setPh] = useState<PhMode>(() => (params.get("ph") === "off" ? "off" : "on"));
  const [tipA, setTipA] = useState<Tip | null>(null);
  const [tipB, setTipB] = useState<Tip | null>(null);

  const keys = useMemo(() => index ? [...new Set(index.runs.filter((r) => r.drones === drones && SHOWN_SETTINGS.includes(r.setting)).map((r) => (r.setting === "baseline" ? r.strategy : `${r.setting}_${r.strategy}`)))].sort((x, y) => order(x) - order(y)) : [], [index, drones]);
  useEffect(() => {
    if (!keys.length) return;
    if (!keys.includes(a)) setA(keys[0]);
    if (!keys.includes(b)) setB(keys.find((k) => k !== a) ?? keys[1]);
  }, [keys, a, b]);

  const runId = useCallback((key: string) => `${settingOf(key)}_${strategyOf(key)}_s${seed}_d${drones}`, [seed, drones]);
  useEffect(() => {
    if (!keys.includes(a) || (compare && !keys.includes(b))) return;
    let live = true;
    Promise.all([getJSON<RawRun>(`/data/runs/${runId(a)}.json`), compare ? getJSON<RawRun>(`/data/runs/${runId(b)}.json`) : null, getJSON<Iso>(`/data/fire/s${seed}.json`)])
      .then(([ra, rb, fire]) => { if (live) setRuns({ a: decodeRun(ra), b: rb ? decodeRun(rb) : null, fire }); });
    return () => { live = false; };
  }, [a, b, compare, seed, drones, keys, runId]);
  useEffect(() => { writeQuery({ d: drones, a, b: compare ? b : "none", ph }); }, [drones, a, b, compare, ph]);

  const seek = useCallback((time: number) => { const v = Math.max(0, Math.min(DAY, time)); clock.set(v); setT(v); }, [clock]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0, last: number | null = null, lastUI = 0;
    const snaps = index ? [...new Set(index.perimeters.map((p) => p.h * 3600))].filter((s) => s > 0) : [];
    const tick = (now: number) => {
      if (last !== null) {
        const prev = clock.get();
        const next = Math.min(DAY, prev + ((now - last) / 1000) * speed * 60);
        clock.set(next);
        const hit = snaps.find((s) => prev < s && next >= s);
        if (hit) { setFlash(`New satellite perimeter at ${hhmm(hit)}`); setTimeout(() => setFlash(null), 2200); }
        if (now - lastUI > 100 || next >= DAY) { setT(next); lastUI = now; }
        if (next >= DAY) { setPlaying(false); return; }
      }
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, index, clock]);

  const toggle = useCallback(() => setPlaying((p) => { if (!p && clock.get() >= DAY) seek(0); return !p; }), [clock, seek]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || (e.target as HTMLElement).closest("button,select,input")) return;
      e.preventDefault();
      toggle();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [toggle]);

  const stA = runs ? runs.a.stages[stageAt(runs.a, t)] : null;
  const stB = runs?.b ? runs.b.stages[stageAt(runs.b, t)] : null;
  const meanOf = (r: Run) => r.stages.reduce((s, x) => s + x.true_over, 0) / r.stages.length;

  if (!index) return <div className="w-full h-[560px] bg-gray-alpha-100 animate-pulse rounded-md" />;

  const groups = SHOWN_SETTINGS.map((g) => ({
    label: g === "baseline" ? "Baselines" : "ACO pheromone strategies",
    options: keys.filter((k) => settingOf(k) === g).map((k) => ({ value: k, label: plannerLabel(k) })),
  })).filter((g) => g.options.length);

  const map = (which: "a" | "b") => {
    const run = which === "a" ? runs?.a : runs?.b;
    const key = which === "a" ? a : b;
    const hidden = !compare ? which === "b" : which !== mobileSide;
    if (!compare && which === "b") return null;
    return (
      <FireMap
        key={which}
        bounds={index.bounds} center={CENTER} spanKm={SPAN}
        className={hidden ? "hidden md:block" : ""}
        tip={which === "a" ? tipA : tipB}
        caption={<span className="flex items-center gap-1.5"><span style={{ color: SIDE_COLOR[which] }}>▲</span>{which.toUpperCase()} · {plannerLabel(key)}</span>}
        overlay={<>
          {flash && <div className="pointer-events-none absolute top-10 left-1/2 z-10 -translate-x-1/2 bg-gray-1000 text-background-100 px-3 py-1.5 font-mono text-xs uppercase whitespace-nowrap">{flash}</div>}
          {ph === "on" && noPheromone(key) && (
            <div className="pointer-events-none absolute bottom-10 left-2 max-w-[80%] bg-[var(--ds-background-100)]/85 px-1.5 py-1 font-mono text-[11px] uppercase text-gray-900">{noPheromone(key)}</div>
          )}
        </>}
      >
        {(proj) => run && <MissionLayer index={index} run={run} fire={runs!.fire} color={SIDE_COLOR[which]} clock={clock} proj={proj} ph={ph} onTip={which === "a" ? setTipA : setTipB} />}
      </FireMap>
    );
  };

  return (
    <div className="space-y-1.5">
      <section aria-label="Mission settings" className="flex flex-wrap items-end gap-x-6 gap-y-4 mb-8">
        <Field label="Drones"><Seg label="Drones" value={drones} options={[6, 10, 12].map((d) => ({ value: d, label: String(d) }))} onChange={setDrones} /></Field>
        <Field label={<><span style={{ color: SIDE_COLOR.a }}>▲</span>Planner A</>} className="w-full sm:w-auto"><Select label="Planner A" value={a} onChange={setA} groups={groups} /></Field>
        <Field label={<><span style={{ color: SIDE_COLOR.b }}>▲</span>Planner B <Toggle checked={compare} onChange={setCompare} label="Compare with planner B" /></>} className="w-full sm:w-auto">
          <Select label="Planner B" value={b} onChange={setB} groups={groups} disabled={!compare} />
        </Field>
        <Field label="Pheromone"><Seg label="Pheromone shown" value={ph} options={PH_MODES} onChange={setPh} /></Field>
      </section>

      <div className="grid gap-6 min-[961px]:grid-cols-[minmax(320px,380px)_minmax(0,1fr)] min-[961px]:items-end">
        <section aria-label="Live numbers" className="flex flex-col gap-y-6 md:gap-y-8 pb-2">
          <div className="space-y-2">
            <h2 className="my-0 font-mono font-medium text-sm tracking-tight uppercase text-gray-900">Edge points overdue · {hhmm(t)}</h2>
            <div className={`grid gap-x-6 ${compare ? "grid-cols-2" : "grid-cols-1"}`}>
              {(compare ? (["a", "b"] as const) : (["a"] as const)).map((w) => {
                const run = w === "a" ? runs?.a : runs?.b, st = w === "a" ? stA : stB, key = w === "a" ? a : b;
                return (
                  <div key={w} className="min-w-0" title={plannerLabel(key)}>
                    <div className="font-mono text-xs uppercase tracking-tight text-gray-900 truncate"><span style={{ color: SIDE_COLOR[w] }}>▲</span> {w.toUpperCase()} {compact(key)}</div>
                    <div className="text-4xl md:text-5xl font-mono tabular-nums">{st?.true_over ?? "–"}</div>
                    <div className="text-sm text-gray-900 font-mono tabular-nums">{run ? `${meanOf(run).toFixed(1)} avg` : "…"}</div>
                  </div>
                );
              })}
            </div>
            <div className="text-sm text-gray-900 font-mono tabular-nums">{stA ? <>of {stA.front} points on the true fire edge</> : "…"}</div>
          </div>
          <div>
            <CountLine value={drones} label="drones · replan every 10 min" />
            <CountLine value={runs && stA && stageAt(runs.a, t) > 0 ? `${Math.round(stA.changed * 100)}%` : "–"} label="of the map changed at this replan" />
          </div>
        </section>

        <div className="min-w-0 order-first min-[961px]:order-none">
          {compare && (
            <Seg className="md:hidden mb-2" label="Map shown" value={mobileSide} onChange={setMobileSide} options={[
              { value: "a" as const, label: `A ${compact(a)}`, color: SIDE_COLOR.a },
              { value: "b" as const, label: `B ${compact(b)}`, color: SIDE_COLOR.b },
            ]} />
          )}
          <div className={compare ? "grid gap-1.5 md:grid-cols-2" : "max-w-[760px] ml-auto"}>
            {map("a")}
            {map("b")}
          </div>
        </div>
      </div>

      <section aria-label="Playback" className="sticky bottom-0 z-20 -mx-4 sm:mx-0 px-4 sm:px-0 py-2 bg-[var(--ds-background-100)] border-t border-gray-alpha-200 md:static md:border-0 md:py-4">
        <div className="flex items-center gap-3 md:gap-4">
          <button type="button" onClick={toggle} aria-label={playing ? "Pause" : "Play"}
            className="h-9 px-3 font-mono text-xs uppercase bg-gray-1000 text-background-100 cursor-pointer shrink-0 outline-none focus-visible:ring">
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          <div className="font-mono text-lg md:text-xl tabular-nums shrink-0">{hhmm(t)}</div>
          <input type="range" className="range min-w-0 flex-1" min={0} max={DAY} step={60} value={t} onChange={(e) => seek(+e.target.value)} aria-label="Mission time" />
          <div className="hidden md:block"><Seg label="Playback speed" value={speed} options={SPEEDS} onChange={setSpeed} /></div>
          <button type="button" className="md:hidden h-8 px-2 font-mono text-xs uppercase border border-gray-alpha-400 tabular-nums shrink-0 cursor-pointer" aria-label="Change playback speed"
            onClick={() => setSpeed(SPEEDS[(SPEEDS.findIndex((o) => o.value === speed) + 1) % SPEEDS.length].value)}>
            {SPEEDS.find((o) => o.value === speed)?.label}
          </button>
        </div>
      </section>

      <ul aria-label="Map legend" className="list-none pl-0 flex flex-wrap gap-x-5 gap-y-2 pb-6">
        {STALE_BINS.map((s) => <LegendItem key={s.label} mark={<Sq color={s.color} />}>{s.label}</LegendItem>)}
        <LegendItem mark={<DroneIcon ring={SIDE_COLOR.a} />}>drone</LegendItem>
        {ph !== "off" && <LegendItem mark={<span className="inline-flex flex-col gap-0.5"><Ln color={SIDE_COLOR.a} w={3} />{compare && <Ln color={SIDE_COLOR.b} w={3} />}</span>}>pheromone kept from last plan (wider = stronger)</LegendItem>}
        <LegendItem mark={<Ln color="var(--ds-gray-1000)" w={3} />}>satellite perimeter (FEDS)</LegendItem>
        <LegendItem mark={<Ln color="var(--fire)" w={3} />}>modelled fire front</LegendItem>
      </ul>

      <section aria-label="Mission statistics" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1.5">
        <StatCard
          className="md:col-span-2"
          title="Edge points overdue · 24 h"
          infoContent={<>
            <p>Each replan (every 10 minutes) counts the monitoring points on the <b>real</b> fire edge whose staleness is above 1, meaning the time since a drone last visited is longer than that point&apos;s revisit period. Lower is better.</p>
            <p>Dashed lines are the mission means. The dotted line at 12 h marks the second satellite perimeter. Click the chart to jump the replay to that time.</p>
          </>}
        >
          {runs && <OverdueChart runs={runs} a={a} b={compare ? b : null} t={t} onSeek={seek} />}
        </StatCard>
        <StatCard
          title="Map change per replan"
          value={runs ? `${Math.round(medianChange(runs.a) * 100)}%` : "–"}
          infoContent={<>
            <p>Share of monitoring points that were added, removed or given a new risk level since the previous replan (planner A&apos;s map).</p>
            <p>The large dark bars are replans where a new satellite perimeter (FEDS) arrived; the fire edge jumps, so most of the map changes at once.</p>
          </>}
        >
          <p className="text-gray-900 text-sm font-mono mt-1 uppercase">median change at each replan</p>
          {runs && <ChangeChart run={runs.a} />}
        </StatCard>
        <StatCard
          className="lg:col-span-2"
          title={`Mean over 10 seeds · ${drones} drones`}
          infoContent={<>
            <p>Mean overdue edge points over the whole 24 h mission, averaged over 10 mission seeds. Lower is better.</p>
            <p>The highlighted rows are the planners on the maps.</p>
          </>}
        >
          <MeansList index={index} drones={drones} selected={compare ? [a, b] : [a]} />
        </StatCard>
        <StatCard
          title={`At ${hhmm(t)}`}
          infoContent={<>
            <p><b>True edge</b>: overdue points on the real fire edge (what matters). <b>Map</b>: overdue points on the map the planner sees, which can lag the real fire between satellite perimeters.</p>
          </>}
        >
          <ul className="space-y-1 list-none pl-0 mt-2">
            <MetricRow label="Overdue, edge" value={stA?.true_over ?? "–"} sub={compare ? (stB?.true_over ?? "–") : undefined} />
            <MetricRow label="Overdue, map" value={stA?.map_over ?? "–"} sub={compare ? (stB?.map_over ?? "–") : undefined} />
            <MetricRow label="Edge points" value={stA?.front ?? "–"} sub={compare ? (stB?.front ?? "–") : undefined} />
          </ul>
          {compare && <p className="text-gray-900 text-xs font-mono mt-2 uppercase">columns: A · B</p>}
        </StatCard>
        <StatCard
          title="Pheromone kept"
          value={runs ? phCount(runs.a, stageAt(runs.a, t), ph) : "–"}
          infoContent={<>
            <p>Each ant colony keeps a pheromone value for every pair of points. Good routes deposit pheromone, every iteration evaporates 36% of it, and values are kept between a floor and a ceiling (MAX-MIN ant system).</p>
            <p>Links no ant uses sink to the floor. The map draws the two strongest outgoing links of each point, only where pheromone is at least 3× the floor. Wider, darker lines are stronger (the strongest are about 40× the floor).</p>
            <p>The map shows the pheromone each strategy <b>keeps</b> when the map changes, before the new ants run: what the planner remembers from its last plan. Retention keeps everything, selective reset clears links touching changed points, full reset clears it all (so its map shows none), and fast evaporation keeps it but burns it off faster over the next iterations.</p>
            <p>Every planner then runs 60 iterations on the new map, so they all end up with a similar strong set of links. The difference is where they start. Greedy and VNS have no pheromone.</p>
          </>}
        >
          <p className="text-gray-900 text-sm mt-2">{runs && !isACO(runs.a) ? "Planner A is a baseline, so there is no pheromone to show." : ph === "off" ? "hidden" : "strong links planner A carried into this replan"}</p>
        </StatCard>
      </section>
    </div>
  );
}

/** Why a map shows no pheromone, or null when it should show some. */
function noPheromone(key: string) {
  if (PLANNER[key]) return `No pheromone: ${PLANNER[key].short} does not use it`;
  if (strategyOf(key) === "full_reset") return "No pheromone kept: full reset clears it at every change";
  return null;
}

function phCount(run: Run, k: number, mode: PhMode) {
  if (mode === "off" || !isACO(run)) return "–";
  const e = stagePheromone(run, k);
  return e ? e.length : "–";
}

function medianChange(run: Run) {
  const v = run.stages.slice(1).map((s) => s.changed).sort((x, y) => x - y);
  return v[Math.floor(v.length / 2)];
}

function TipBox({ children }: { children: React.ReactNode }) {
  return <div className="bg-[var(--ds-background-200)] border border-[var(--ds-gray-200)] rounded px-2.5 py-1.5 text-xs font-mono shadow-lg uppercase">{children}</div>;
}

const OverdueChart = memo(function OverdueChart({ runs, a, b, t, onSeek }: { runs: { a: Run; b: Run | null }; a: string; b: string | null; t: number; onSeek: (t: number) => void }) {
  const data = useMemo(() => runs.a.stages.map((s, i) => ({ t: s.t, a: s.true_over, b: runs.b?.stages[i]?.true_over })), [runs]);
  const meanA = useMemo(() => runs.a.stages.reduce((s, x) => s + x.true_over, 0) / runs.a.stages.length, [runs]);
  const meanB = useMemo(() => (runs.b ? runs.b.stages.reduce((s, x) => s + x.true_over, 0) / runs.b.stages.length : null), [runs]);
  const config = { a: { label: plannerShort(a), color: SIDE_COLOR.a }, b: { label: b ? plannerShort(b) : "", color: SIDE_COLOR.b } } satisfies ChartConfig;
  return (
    <>
      <ul className="list-none pl-0 my-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs uppercase">
        <li className="flex items-center gap-1.5"><span style={{ color: SIDE_COLOR.a }}>■</span>A {plannerShort(a)} · avg {meanA.toFixed(1)}</li>
        {b && meanB !== null && <li className="flex items-center gap-1.5"><span style={{ color: SIDE_COLOR.b }}>■</span>B {plannerShort(b)} · avg {meanB.toFixed(1)}</li>}
      </ul>
      <ChartContainer config={config} className="aspect-auto h-56 w-full">
        <LineChart data={data} margin={{ left: -16, right: 8, top: 6 }}
          onClick={(e) => { const i = e?.activeTooltipIndex; if (i != null) onSeek(data[+i].t); }} style={{ cursor: "pointer" }}>
          <CartesianGrid vertical={false} stroke="var(--ds-gray-alpha-200)" />
          <XAxis dataKey="t" type="number" domain={[0, DAY]} ticks={[0, 4, 8, 12, 16, 20, 24].map((h) => h * 3600)} tickFormatter={(v) => `${v / 3600}H`} tickLine={false} axisLine={{ stroke: "var(--ds-gray-alpha-400)" }} tick={AXIS} />
          <YAxis tickLine={false} axisLine={false} width={44} tick={AXIS} />
          <ReferenceLine x={12 * 3600} stroke="var(--ds-gray-1000)" strokeDasharray="2 3" strokeOpacity={0.5} />
          <ReferenceLine y={meanA} stroke={SIDE_COLOR.a} strokeDasharray="4 4" strokeOpacity={0.7} />
          {meanB !== null && <ReferenceLine y={meanB} stroke={SIDE_COLOR.b} strokeDasharray="4 4" strokeOpacity={0.7} />}
          <ChartTooltip cursor={{ stroke: "var(--ds-gray-alpha-600)" }} content={({ active, payload, label }) => active && payload?.length ? (
            <TipBox>
              <div className="text-gray-1000">{hhmm(+label!)}</div>
              {payload.map((p) => <div key={String(p.dataKey)} className="flex items-center gap-1.5 text-gray-900"><span style={{ color: String(p.color) }}>■</span>{String(p.dataKey).toUpperCase()} {config[p.dataKey as "a" | "b"].label}: <span className="text-gray-1000">{p.value}</span></div>)}
            </TipBox>) : null} />
          <Line dataKey="a" stroke="var(--color-a)" strokeWidth={2} dot={false} isAnimationActive={false} type="monotone" />
          {b && <Line dataKey="b" stroke="var(--color-b)" strokeWidth={2} dot={false} isAnimationActive={false} type="monotone" />}
          <ReferenceLine x={t} stroke="var(--ds-gray-1000)" strokeWidth={1.5} />
        </LineChart>
      </ChartContainer>
    </>
  );
});

const ChangeChart = memo(function ChangeChart({ run }: { run: Run }) {
  const data = useMemo(() => run.stages.slice(1).map((s) => ({ t: s.t, changed: s.changed, snap: s.snap })), [run]);
  return (
    <ChartContainer config={{ changed: { label: "Changed", color: "var(--ds-gray-400)" } }} className="aspect-auto h-40 w-full mt-2">
      <BarChart data={data} margin={{ left: -8, right: 4, top: 4 }} barCategoryGap={0}>
        <XAxis dataKey="t" type="number" domain={[0, DAY]} ticks={[0, 12, 24].map((h) => h * 3600)} tickFormatter={(v) => `${v / 3600}H`} tickLine={false} axisLine={{ stroke: "var(--ds-gray-alpha-400)" }} tick={AXIS} />
        <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} width={44} tick={AXIS} />
        <ChartTooltip cursor={false} content={({ active, payload }) => active && payload?.length ? (
          <TipBox>{hhmm(payload[0].payload.t)} · {Math.round(payload[0].payload.changed * 100)}% changed{payload[0].payload.snap ? " · new FEDS perimeter" : ""}</TipBox>) : null} />
        <Bar dataKey="changed" isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={d.snap ? "var(--ds-gray-1000)" : "var(--ds-gray-400)"} />)}
        </Bar>
      </BarChart>
    </ChartContainer>
  );
});

function MeansList({ index, drones, selected }: { index: Index; drones: number; selected: string[] }) {
  const rows = useMemo(() => {
    const by = new Map<string, number[]>();
    for (const r of index.summary.filter((r) => r.drones === drones && SHOWN_SETTINGS.includes(r.setting))) {
      const k = r.setting === "baseline" ? r.strategy : `${r.setting}_${r.strategy}`;
      by.set(k, [...(by.get(k) ?? []), r.overdue]);
    }
    return [...by].map(([k, v]) => [k, v.reduce((s, x) => s + x, 0) / v.length] as const).sort((x, y) => x[1] - y[1]);
  }, [index, drones]);
  const max = Math.max(...rows.map((r) => r[1]));
  return (
    <ul className="list-none pl-0 mt-3 space-y-1">
      {rows.map(([k, v]) => {
        const sel = selected.includes(k);
        return (
          <li key={k} className={`grid grid-cols-[minmax(0,1fr)_4ch_minmax(0,1fr)] sm:grid-cols-[minmax(0,14rem)_4ch_minmax(0,1fr)] items-center gap-3 font-mono text-sm ${sel ? "text-gray-1000" : "text-gray-900"}`}>
            <span className="uppercase truncate flex items-center gap-1.5"><span aria-hidden="true" style={{ color: plannerColor(k) }}>■</span>{plannerShort(k)}</span>
            <span className="tabular-nums text-right">{v.toFixed(1)}</span>
            <span aria-hidden="true" className="h-2" style={{ width: `${(100 * v) / max}%`, background: plannerColor(k), opacity: sel ? 1 : 0.4 }} />
          </li>
        );
      })}
    </ul>
  );
}
