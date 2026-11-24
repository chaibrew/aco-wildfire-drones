"use client";

import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ErrorBar, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { readQuery, useJSON, writeQuery, type Grid, type Index, type Trial, type XY } from "@/lib/data";
import { PLANNER, STALE_BINS, pct, staleColor } from "@/lib/planners";
import { BigStat, CountLine, KeyList, KeyRow, MetricRow, StatCard } from "./StatsDisplay";
import { DroneIcon, DroneSprite, FireMap, LegendItem, Ln, Pixel, Sq, heading, polyline, type Proj, type Tip } from "./FireMap";
import { Field, Seg } from "./Controls";

const ORDER = ["aco_retention", "aco_full_reset", "aco_fast_evaporation", "aco_selective_reset", "greedy", "vns"];
const key = (p: string) => p.replace(/^aco_/, "");
type Metric = "total_overdue" | "margin" | "overdue_count";
const METRICS: Record<Metric, { label: string; long: string; digits: number }> = {
  total_overdue: { label: "Total overdue", long: "Sum over points of (staleness − 1) where it is above 1, at the end of the 10 min plan.", digits: 1 },
  margin: { label: "Worst point", long: "The highest staleness of any point at the end of the plan.", digits: 2 },
  overdue_count: { label: "Points overdue", long: "Number of points overdue at the end of the plan.", digits: 1 },
};
const PHASES = [{ value: "before", label: "Before" }, { value: "change", label: "After change" }, { value: "after", label: "After replan" }] as const;
type Phase = (typeof PHASES)[number]["value"];
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
const sd = (v: number[]) => { const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); };
const AXIS = { fill: "var(--ds-gray-900)", fontSize: 11, fontFamily: "var(--font-geist-mono)" };
const CENTER: XY = [-0.2, -1.5];

export default function ExploreView() {
  const params = useMemo(() => readQuery(), []);
  const grid = useJSON<Grid>("/data/change_grid.json");
  const index = useJSON<Index>("/data/index.json");
  const [frac, setFrac] = useState(+(params.get("f") ?? 0.2));
  const [loc, setLoc] = useState(params.get("loc") ?? "concentrated");
  const [stage, setStage] = useState(+(params.get("st") ?? 40));
  const setting = "tuned";
  const [planner, setPlanner] = useState(params.get("p") ?? "aco_selective_reset");
  const [phase, setPhase] = useState<Phase>((params.get("ph") as Phase) ?? "change");
  const [metric, setMetric] = useState<Metric>((params.get("m") as Metric) ?? "total_overdue");
  const [trial, setTrial] = useState(0);
  const [tip, setTip] = useState<Tip | null>(null);

  useEffect(() => { writeQuery({ f: frac, loc, st: stage, p: planner, ph: phase, m: metric }); }, [frac, loc, stage, planner, phase, metric]);

  const cellRows = useMemo(() => grid?.trials.filter((t) => t.stage === stage && t.setting === setting && t.locality === loc && t.fraction === frac) ?? [], [grid, stage, setting, loc, frac]);
  const ex = cellRows.find((t) => t.trial === trial && t.after);
  const stages = useMemo(() => grid ? [...new Set(grid.trials.map((t) => t.stage))].sort((x, y) => x - y).map((s) => ({ s, h: grid.trials.find((t) => t.stage === s && t.time_h != null)!.time_h! })) : [], [grid]);
  const stats = useMemo(() => {
    const ref = "aco_full_reset", eps = 1e-9;
    return ORDER.map((p) => {
      const d = cellRows.map((t) => t.planners[p][metric] - t.planners[ref][metric]);
      return { p, mean: cellRows.length ? mean(cellRows.map((t) => t.planners[p][metric])) : NaN, better: d.filter((x) => x < -eps).length, worse: d.filter((x) => x > eps).length, tied: d.filter((x) => Math.abs(x) <= eps).length };
    });
  }, [cellRows, metric]);
  const sel = stats.find((s) => s.p === planner);

  if (!grid || !index) return <div className="w-full h-[560px] bg-gray-alpha-100 animate-pulse rounded-md" />;
  const m = METRICS[metric];
  const P = ex?.planners[planner];
  const isACO = planner.startsWith("aco_");
  const best = Math.min(...stats.map((s) => s.mean));

  return (
    <div className="space-y-1.5">
      <section aria-label="Experiment settings" className="flex flex-wrap items-end gap-x-6 gap-y-4 mb-8">
        <Field label={<>Share of points changed <span className="text-gray-1000">{pct(frac)}</span></>} className="w-full sm:w-80">
          <input type="range" className="range w-full" min={0} max={grid.fractions.length - 1} step={1} value={grid.fractions.indexOf(frac)}
            onChange={(e) => setFrac(grid.fractions[+e.target.value])} aria-label="Share of points changed" />
          <div className="flex justify-between font-mono text-xs text-gray-900">{grid.fractions.map((f) => <span key={f}>{pct(f)}</span>)}</div>
        </Field>
        <Field label="Where"><Seg label="Where" value={loc} options={[{ value: "concentrated", label: "One stretch" }, { value: "distributed", label: "Spread out" }]} onChange={setLoc} /></Field>
        <Field label="Fire state"><Seg label="Fire state" value={stage} options={stages.map(({ s, h }) => ({ value: s, label: `+${h.toFixed(1)} h` }))} onChange={setStage} /></Field>
      </section>

      <div className="grid gap-6 min-[961px]:grid-cols-[minmax(300px,400px)_minmax(0,1fr)] min-[961px]:items-end">
        <section aria-label="Results at this change size" className="flex flex-col gap-y-6 md:gap-y-8 pb-2">
          <p className="text-sm text-gray-900 leading-relaxed m-0 normal-case">
            One ACO router is warmed up on a real Camp Fire state, then a share of its points is changed at once (new risk, or moved one cell outward),
            in <span className="text-gray-1000">one stretch</span> or <span className="text-gray-1000">spread out</span>. Every planner replans the same changed state with the same budget.
          </p>
          <BigStat
            label={`${m.label} · ${PLANNER[key(planner)].short}`}
            value={sel && Number.isFinite(sel.mean) ? sel.mean.toFixed(m.digits) : "–"}
            sub={<>mean of {cellRows.length} trials at {pct(frac)} changed · lower is better</>}
          />
          <KeyList label="Planners · mean / vs full reset (b·t·w)">
            {stats.map((s) => (
              <KeyRow key={s.p} color={PLANNER[key(s.p)].color} code={`${PLANNER[key(s.p)].short}${Math.abs(s.mean - best) < 1e-9 ? " ★" : ""}`}
                value={s.mean.toFixed(m.digits)} sub={s.p === "aco_full_reset" ? "—" : `${s.better}·${s.tied}·${s.worse}`}
                active={s.p === planner} onClick={() => setPlanner(s.p)} title={`${PLANNER[key(s.p)].label}: better, tied, worse than full reset`} />
            ))}
          </KeyList>
          {ex && <CountLine value={pct(ex.realised)} label={`changed in example #${trial + 1} · ${ex.added + ex.changed} new/changed, ${ex.removed} removed`} />}
        </section>

        <div className="min-w-0 order-first min-[961px]:order-none">
          <div className="flex flex-wrap items-end gap-x-5 gap-y-3 mb-2">
            <Field label="Pheromone"><Seg label="Pheromone" value={phase} options={[...PHASES]} onChange={setPhase} /></Field>
            <Field label="Example"><Seg label="Example" value={trial} options={[0, 1].map((t) => ({ value: t, label: `#${t + 1}` }))} onChange={setTrial} /></Field>
          </div>
          <div className="max-w-[760px] ml-auto">
            <FireMap
              bounds={index.bounds} center={CENTER} spanKm={26} tip={tip}
              caption={ex && P && (
                <span className="flex items-center gap-1.5">
                  <span style={{ color: phase === "before" ? "var(--ds-gray-1000)" : PLANNER[key(planner)].color }}>■</span>
                  {phase === "before" ? `Warmed-up router at +${ex.time_h!.toFixed(1)} h` : `${PLANNER[key(planner)].short}${isACO ? (phase === "change" ? " · pheromone right after the change" : " · pheromone after replanning") : " · no pheromone"}`}
                </span>
              )}
            >
              {(proj) => ex && <TrialLayer ex={ex} planner={planner} phase={phase} proj={proj} onTip={setTip} />}
            </FireMap>
          </div>
        </div>
      </div>

      <ul aria-label="Map legend" className="list-none pl-0 flex flex-wrap gap-x-5 gap-y-2 py-4">
        {STALE_BINS.map((s) => <LegendItem key={s.label} mark={<Sq color={s.color} />}>{s.label}</LegendItem>)}
        <LegendItem mark={<Sq color="var(--ds-gray-400)" ring="var(--ds-gray-1000)" />}>changed point</LegendItem>
        <LegendItem mark={<span className="text-[var(--st-3)] leading-none">×</span>}>removed point</LegendItem>
        <LegendItem mark={<Ln color={PLANNER[key(planner)].color} w={4} />}>pheromone (wider = stronger)</LegendItem>
        <LegendItem mark={<Ln color="var(--ds-gray-1000)" w={1.5} dash />}>planned route</LegendItem>
        <LegendItem mark={<DroneIcon />}>drone (faces its first stop)</LegendItem>
      </ul>

      <section aria-label="Change size statistics" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1.5">
        <StatCard
          className="md:col-span-2"
          title={`${m.label} vs change size`}
          infoContent={<>
            <p><b>{m.label}</b>: {m.long}</p>
            <p>Each dot is the mean of {cellRows.length} trials ({loc === "concentrated" ? "one stretch" : "spread out"}). The shaded column is the share selected above. Click a dot to pick that planner and size. Lower is better.</p>
          </>}
        >
          <div className="my-2"><Seg label="Measure" value={metric} options={(Object.keys(METRICS) as Metric[]).map((k) => ({ value: k, label: METRICS[k].label }))} onChange={setMetric} /></div>
          <SizeChart grid={grid} stage={stage} setting={setting} loc={loc} metric={metric} frac={frac} planner={planner} onPick={(p, f) => { setPlanner(p); setFrac(f); }} />
        </StatCard>
        <StatCard
          title="Example trial"
          infoContent={<p>The map shows one of two stored example trials for these settings (routes and pheromone are only stored for these). Numbers are for the selected planner after replanning.</p>}
        >
          {ex && P && (
            <ul className="space-y-1 list-none pl-0 mt-2">
              <MetricRow label="Points on map" value={ex.after!.points.length} />
              <MetricRow label="New / changed" value={ex.added + ex.changed} />
              <MetricRow label="Removed" value={ex.removed} />
              <MetricRow label="Total overdue" value={P.total_overdue.toFixed(1)} />
              <MetricRow label="Worst point" value={P.margin.toFixed(2)} />
              <MetricRow label="Points overdue" value={P.overdue_count} />
              <MetricRow label="Route length" value={`${P.distance_km.toFixed(1)} km`} />
            </ul>
          )}
        </StatCard>
        <StatCard
          className="md:col-span-2 lg:col-span-3"
          title="Does selective reset help more for local changes?"
          infoContent={<>
            <p>Selective reset minus full reset on <b>{m.label.toLowerCase()}</b>, with 95% intervals over trials. Below 0 means selective reset was better.</p>
            <p>If resetting only near the changes helped more for local changes, the &ldquo;one stretch&rdquo; line would sit clearly below &ldquo;spread out&rdquo; for small changes.</p>
          </>}
        >
          <SelectiveChart grid={grid} stage={stage} setting={setting} metric={metric} />
        </StatCard>
      </section>
    </div>
  );
}

function TrialLayer({ ex, planner, phase, proj, onTip }: { ex: Trial; planner: string; phase: Phase; proj: Proj; onTip: (t: Tip | null) => void }) {
  const before = phase === "before";
  const pts = before ? ex.before!.points : ex.after!.points;
  const P = ex.planners[planner];
  const isACO = planner.startsWith("aco_");
  const edges = ((before ? ex.before!.pheromone : isACO ? (phase === "change" ? P.ph_change : P.ph_after) : []) ?? []).filter((e) => pts[e[0]] && pts[e[1]]);
  const routes = (before ? ex.before!.routes : P.routes) ?? [];
  const color = before ? "var(--ds-gray-1000)" : PLANNER[key(planner)].color;
  const maxR = Math.max(2, ...edges.map((e) => e[2]));
  const w = (r: number) => Math.min(1, Math.log(r) / Math.log(maxR));
  const px = 0.42 * proj.k, X = 0.22 * proj.k;
  return (
    <g>
      {edges.map((e, i) => {
        const [x1, y1] = proj.p([pts[e[0]][0], pts[e[0]][1]]), [x2, y2] = proj.p([pts[e[1]][0], pts[e[1]][1]]);
        return <line key={`e${i}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeOpacity={0.2 + 0.55 * w(e[2])} strokeWidth={2 + 7 * w(e[2])} strokeLinecap="round" />;
      })}
      {routes.map((r, d) => {
        const path: XY[] = [ex.drones![d], ...r.map((i) => [pts[i][0], pts[i][1]] as XY)];
        return path.length > 1 ? <polyline key={`r${d}`} points={polyline(proj, path)} fill="none" stroke="var(--ds-gray-1000)" strokeWidth={1.5} strokeDasharray="5 5" strokeOpacity={0.8} strokeLinejoin="round" /> : null;
      })}
      {!before && ex.after!.removed.map((q, i) => {
        const [x, y] = proj.p(q);
        return <path key={`x${i}`} d={`M${x - X} ${y - X}L${x + X} ${y + X}M${x - X} ${y + X}L${x + X} ${y - X}`} stroke="var(--st-3)" strokeWidth={3.5} />;
      })}
      {pts.map((q, i) => {
        const [x, y] = proj.p([q[0], q[1]]);
        const changed = !before && q[4] > 0;
        return (
          <Pixel key={i} x={x} y={y} size={changed ? px * 1.2 : px} fill={staleColor(q[3])} ring={changed ? "var(--ds-gray-1000)" : undefined} pulse={changed}
            onEnter={() => onTip({ x, y, content: <><span style={{ color: staleColor(q[3]) }}>■</span> STALENESS {q[3].toFixed(2)} · RISK {q[2]}{changed ? " · CHANGED" : ""}</> })}
            onLeave={() => onTip(null)} />
        );
      })}
      {ex.drones!.map((d, i) => {
        const [x, y] = proj.p(d);
        const first = routes[i]?.[0];
        const angle = first != null && pts[first] ? heading(pts[first][0] - d[0], pts[first][1] - d[1]) : 0;
        return <DroneSprite key={`d${i}`} x={x} y={y} size={px * 4.2} angle={angle} frame={(i * 3) % 8} />;
      })}
    </g>
  );
}

function TipBox({ children }: { children: React.ReactNode }) {
  return <div className="bg-[var(--ds-background-200)] border border-[var(--ds-gray-200)] rounded px-2.5 py-1.5 text-xs font-mono shadow-lg uppercase">{children}</div>;
}

function SizeChart({ grid, stage, setting, loc, metric, frac, planner, onPick }: {
  grid: Grid; stage: number; setting: string; loc: string; metric: Metric; frac: number; planner: string; onPick: (p: string, f: number) => void;
}) {
  const data = useMemo(() => grid.fractions.map((f) => {
    const rows = grid.trials.filter((t) => t.stage === stage && t.setting === setting && t.locality === loc && t.fraction === f);
    return Object.fromEntries([["f", f], ...ORDER.map((p) => [p, mean(rows.map((t) => t.planners[p][metric]))])]);
  }), [grid, stage, setting, loc, metric]);
  const config = Object.fromEntries(ORDER.map((p) => [p, { label: PLANNER[key(p)].short, color: PLANNER[key(p)].color }]));
  const digits = METRICS[metric].digits;
  return (
    <>
      <ul className="list-none pl-0 mb-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs uppercase">
        {ORDER.map((p) => (
          <li key={p} className={`flex items-center gap-1.5 ${p === planner ? "text-gray-1000" : "text-gray-900"}`}><span style={{ color: PLANNER[key(p)].color }}>■</span>{PLANNER[key(p)].short}</li>
        ))}
      </ul>
      <ChartContainer config={config} className="aspect-auto h-64 w-full">
        <LineChart data={data} margin={{ left: -8, right: 12, top: 8 }}>
          <CartesianGrid vertical={false} stroke="var(--ds-gray-alpha-200)" />
          <XAxis dataKey="f" tickFormatter={pct} tickLine={false} axisLine={{ stroke: "var(--ds-gray-alpha-400)" }} tick={AXIS} />
          <YAxis domain={["auto", "auto"]} tickLine={false} axisLine={false} width={48} tick={AXIS} tickFormatter={(v) => (+v).toFixed(digits)} />
          <ReferenceLine x={frac} stroke="var(--ds-gray-alpha-200)" strokeWidth={28} />
          <ChartTooltip cursor={{ stroke: "var(--ds-gray-alpha-600)" }} content={({ active, payload, label }) => active && payload?.length ? (
            <TipBox>
              <div className="text-gray-1000 mb-0.5">{pct(+label!)} changed</div>
              {[...payload].sort((x, y) => +x.value! - +y.value!).map((p) => (
                <div key={String(p.dataKey)} className="flex items-center gap-1.5 text-gray-900"><span style={{ color: String(p.color) }}>■</span>{config[String(p.dataKey)].label}: <span className="text-gray-1000">{(+p.value!).toFixed(digits)}</span></div>))}
            </TipBox>) : null} />
          {ORDER.map((p) => (
            <Line key={p} dataKey={p} stroke={`var(--color-${p})`} strokeWidth={p === planner ? 3 : 1.5} strokeOpacity={p === planner ? 1 : 0.7}
              strokeDasharray={p === "vns" ? "5 3" : undefined} isAnimationActive={false}
              dot={{ r: p === planner ? 4 : 3, fill: `var(--color-${p})`, stroke: "var(--ds-background-100)", strokeWidth: 1.5 }}
              activeDot={{ r: 6, onClick: (_e: unknown, d: unknown) => onPick(p, (d as { payload: { f: number } }).payload.f), style: { cursor: "pointer" } }} />
          ))}
        </LineChart>
      </ChartContainer>
    </>
  );
}

function SelectiveChart({ grid, stage, setting, metric }: { grid: Grid; stage: number; setting: string; metric: Metric }) {
  const data = useMemo(() => grid.fractions.map((f) => {
    const row: Record<string, number | [number, number]> = { f };
    for (const loc of grid.localities) {
      const d = grid.trials.filter((t) => t.stage === stage && t.setting === setting && t.locality === loc && t.fraction === f)
        .map((t) => t.planners.aco_selective_reset[metric] - t.planners.aco_full_reset[metric]);
      const mu = mean(d), ci = d.length > 1 ? (1.96 * sd(d)) / Math.sqrt(d.length) : 0;
      row[loc] = mu; row[`${loc}_ci`] = [ci, ci];
    }
    return row;
  }), [grid, stage, setting, metric]);
  const config = { concentrated: { label: "One stretch", color: "var(--p-selective_reset)" }, distributed: { label: "Spread out", color: "var(--ds-gray-500)" } };
  return (
    <>
      <ul className="list-none pl-0 my-2 flex gap-4 font-mono text-xs uppercase text-gray-900">
        <li className="flex items-center gap-1.5"><span style={{ color: config.concentrated.color }}>■</span>one stretch</li>
        <li className="flex items-center gap-1.5"><span style={{ color: config.distributed.color }}>■</span>spread out</li>
      </ul>
      <ChartContainer config={config} className="aspect-auto h-56 w-full">
        <LineChart data={data} margin={{ left: -8, right: 12, top: 8 }}>
          <CartesianGrid vertical={false} stroke="var(--ds-gray-alpha-200)" />
          <XAxis dataKey="f" tickFormatter={pct} tickLine={false} axisLine={{ stroke: "var(--ds-gray-alpha-400)" }} tick={AXIS} />
          <YAxis tickLine={false} axisLine={false} width={48} tick={AXIS} tickFormatter={(v) => (+v).toFixed(METRICS[metric].digits)} />
          <ReferenceLine y={0} stroke="var(--ds-gray-1000)" strokeOpacity={0.6} />
          <ChartTooltip cursor={{ stroke: "var(--ds-gray-alpha-600)" }} content={({ active, payload, label }) => active && payload?.length ? (
            <TipBox>
              <div className="text-gray-1000 mb-0.5">{pct(+label!)} changed</div>
              {payload.map((p) => {
                const ci = (p.payload[`${String(p.dataKey)}_ci`] as [number, number])[0];
                return <div key={String(p.dataKey)} className="text-gray-900">{config[p.dataKey as keyof typeof config].label}: <span className="text-gray-1000">{(+p.value!).toFixed(2)}</span> (±{ci.toFixed(2)})</div>;
              })}
            </TipBox>) : null} />
          {grid.localities.map((loc) => (
            <Line key={loc} dataKey={loc} stroke={`var(--color-${loc})`} strokeWidth={2} isAnimationActive={false} dot={{ r: 3.5, fill: `var(--color-${loc})`, strokeWidth: 0 }}>
              <ErrorBar dataKey={`${loc}_ci`} stroke={`var(--color-${loc})`} strokeOpacity={0.6} width={4} direction="y" />
            </Line>
          ))}
        </LineChart>
      </ChartContainer>
    </>
  );
}
