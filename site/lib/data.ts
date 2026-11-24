"use client"

import { useEffect, useState } from "react"

export type XY = [number, number]

export interface Index {
  bounds: [number, number, number, number]
  perimeters: { h: number; ring: XY[] }[]
  runs: { id: string; setting: string; strategy: string; seed: number; drones: number }[]
  summary: { setting: string; strategy: string; seed: number; drones: number; overdue: number; J: number; planning_s: number }[]
}
export interface RawRun {
  points: [number, number, number][]
  speed: number
  service: number
  stages: { t: number; c: number[]; true_over: number; front: number; map_over: number; changed: number; snap: boolean; pc?: number[] }[]
  trips: number[][]
}
export type Iso = { h: number; path: XY[] }[]

export interface PlannerResult {
  margin: number
  total_overdue: number
  overdue_count: number
  objective: number
  distance_km: number
  runtime_s: number
  routes?: number[][]
  ph_change?: [number, number, number][]
  ph_after?: [number, number, number][]
}
export type GridPoint = [number, number, number, number, number]
export interface Trial {
  stage: number
  setting: string
  trial: number
  fraction: number
  locality: string
  realised: number
  dispersion: number
  added: number
  removed: number
  changed: number
  planners: Record<string, PlannerResult>
  before?: { points: GridPoint[]; pheromone: [number, number, number][]; routes: number[][] }
  after?: { points: GridPoint[]; removed: XY[] }
  drones?: XY[]
  time_h?: number
}
export interface Grid {
  drones: number
  fractions: number[]
  localities: string[]
  trials: Trial[]
}

const cache = new Map<string, Promise<unknown>>()
export function getJSON<T>(path: string): Promise<T> {
  const url = path.startsWith("/") ? path : `/${path}`
  if (!cache.has(url))
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`)
      return r.json()
    }))
  return cache.get(url) as Promise<T>
}

export function useJSON<T>(url: string | null): T | null {
  const [data, setData] = useState<T | null>(null)
  useEffect(() => {
    let live = true
    if (!url) { setData(null); return }
    getJSON<T>(url).then((d) => { if (live) setData(d) })
    return () => { live = false }
  }, [url])
  return data
}

let hill: Promise<HTMLImageElement> | null = null
export function getHill() {
  if (!hill)
    hill = new Promise((res) => {
      const im = new Image()
      im.onload = () => res(im)
      im.src = "/data/hillshade.png"
    })
  return hill
}

export interface Run {
  P: [number, number, number][]
  stages: RawRun["stages"]
  trips: { path: XY[]; ts: number[] }[]
  pts: Map<number, { p: XY; s: number }[]>
}

export function decodeRun(raw: RawRun): Run {
  const P = raw.points
  const trips = raw.trips.map((flat) => {
    const path: XY[] = [], ts: number[] = []
    for (let k = 0; k < flat.length; k += 2) {
      const p = P[flat[k]], t = flat[k + 1]
      const last = ts.length ? ts[ts.length - 1] : -Infinity
      path.push([p[0], p[1]]); ts.push(Math.max(t, last))
      const next = k + 3 < flat.length ? flat[k + 3] : Infinity
      const dwell = Math.min(t + raw.service, next)
      if (dwell > t) { path.push([p[0], p[1]]); ts.push(dwell) }
    }
    return { path, ts }
  })
  return { P, stages: raw.stages, trips, pts: new Map() }
}

export function stageAt(run: Run, t: number) {
  const st = run.stages
  let lo = 0, hi = st.length - 1
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (st[m].t <= t) lo = m; else hi = m - 1 }
  return lo
}

export function stagePoints(run: Run, k: number) {
  let out = run.pts.get(k)
  if (!out) {
    const c = run.stages[k].c
    out = []
    for (let i = 0; i < c.length; i += 2) { const p = run.P[c[i]]; out.push({ p: [p[0], p[1]], s: c[i + 1] / 20 }) }
    run.pts.set(k, out)
  }
  return out
}

export type PhEdge = [XY, XY, number]
const phCache = new WeakMap<Run, Map<string, PhEdge[]>>()
export function stagePheromone(run: Run, k: number): PhEdge[] | null {
  const flat = run.stages[k].pc
  if (!flat) return null
  let m = phCache.get(run)
  if (!m) { m = new Map(); phCache.set(run, m) }
  const key = String(k)
  let out = m.get(key)
  if (!out) {
    out = []
    for (let i = 0; i < flat.length; i += 3) {
      const a = run.P[flat[i]], b = run.P[flat[i + 1]]
      out.push([[a[0], a[1]], [b[0], b[1]], flat[i + 2] / 10])
    }
    m.set(key, out)
  }
  return out
}

export function dronePos(trip: Run["trips"][number], t: number): XY | null {
  const ts = trip.ts
  if (!ts.length || t < ts[0]) return null
  let lo = 0, hi = ts.length - 1
  if (t >= ts[hi]) return trip.path[hi]
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m }
  const f = (t - ts[lo]) / Math.max(ts[hi] - ts[lo], 1e-6), a = trip.path[lo], b = trip.path[hi]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
}

export function readQuery() {
  return new URLSearchParams(typeof window === "undefined" ? "" : window.location.search)
}
export function writeQuery(params: Record<string, string | number>) {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))
  window.history.replaceState(window.history.state, "", `${window.location.pathname}?${q}`)
}
