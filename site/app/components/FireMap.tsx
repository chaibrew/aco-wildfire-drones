"use client";

import { memo, useEffect, useMemo, useState, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { getHill, type XY } from "@/lib/data";

export const SIZE = 1000;

export interface Proj {
  x: (km: number) => number;
  y: (km: number) => number;
  p: (xy: XY) => [number, number];
  k: number;
}

export function makeProj(center: XY, spanKm: number): Proj {
  const k = SIZE / spanKm;
  const x0 = center[0] - spanKm / 2, y1 = center[1] + spanKm / 2;
  const x = (v: number) => (v - x0) * k;
  const y = (v: number) => (y1 - v) * k;
  return { x, y, p: (q) => [x(q[0]), y(q[1])], k };
}

function useTerrain(bounds: [number, number, number, number], proj: Proj, stepKm: number) {
  const [dots, setDots] = useState<{ x: number; y: number; o: number }[]>([]);
  useEffect(() => {
    let live = true;
    getHill().then((img) => {
      if (!live) return;
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, c.width, c.height).data;
      const [bx0, by0, bx1, by1] = bounds;
      const out: { x: number; y: number; o: number }[] = [];
      for (let ky = by0; ky <= by1; ky += stepKm) {
        for (let kx = bx0; kx <= bx1; kx += stepKm) {
          const px = proj.x(kx), py = proj.y(ky);
          if (px < 0 || px > SIZE || py < 0 || py > SIZE) continue;
          const ix = Math.min(c.width - 1, Math.floor(((kx - bx0) / (bx1 - bx0)) * c.width));
          const iy = Math.min(c.height - 1, Math.floor(((by1 - ky) / (by1 - by0)) * c.height));
          const i = (iy * c.width + ix) * 4;
          if (data[i + 3] < 10) continue;
          const shade = data[i] / 255;
          out.push({ x: px, y: py, o: 0.18 + 0.7 * (1 - shade) ** 1.2 });
        }
      }
      setDots(out);
    });
    return () => { live = false; };
  }, [bounds, proj, stepKm]);
  return dots;
}

const Terrain = memo(function Terrain({ bounds, proj }: { bounds: [number, number, number, number]; proj: Proj }) {
  const dots = useTerrain(bounds, proj, 0.4);
  const s = 0.4 * proj.k * 0.38;
  return (
    <g aria-hidden="true">
      {dots.map((d, i) => (
        <rect key={i} x={d.x - s / 2} y={d.y - s / 2} width={s} height={s} fill="var(--terrain)" fillOpacity={d.o} />
      ))}
    </g>
  );
});

export interface Tip { x: number; y: number; content: ReactNode }

export function FireMap({ bounds, center, spanKm, caption, overlay, children, tip, className }: {
  bounds: [number, number, number, number];
  center: XY;
  spanKm: number;
  caption?: ReactNode;
  overlay?: ReactNode;
  children: (proj: Proj) => ReactNode;
  tip?: Tip | null;
  className?: string;
}) {
  const proj = useMemo(() => makeProj(center, spanKm), [center, spanKm]);
  const scaleKm = [1, 2, 5, 10].find((k) => (k * proj.k) / SIZE >= 0.12) ?? 10;
  return (
    <div className={`relative w-full aspect-square ${className || ""}`}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full h-full bg-[var(--ds-background-100)] select-none">
        <Terrain bounds={bounds} proj={proj} />
        {children(proj)}
        <g transform={`translate(${SIZE - 24 - scaleKm * proj.k} ${SIZE - 28})`}>
          <rect width={scaleKm * proj.k} height={4} fill="var(--ds-gray-1000)" />
          <text x={(scaleKm * proj.k) / 2} y={-8} textAnchor="middle" className="font-mono" fontSize={22} fill="var(--ds-gray-900)">{scaleKm} KM</text>
        </g>
      </svg>
      {caption && (
        <div className="absolute top-0 left-0 max-w-full font-mono text-xs md:text-sm uppercase tracking-tight bg-[var(--ds-background-100)]/80 pr-2 py-1 pointer-events-none">
          {caption}
        </div>
      )}
      {overlay}
      <AnimatePresence>
        {tip && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            transition={{ duration: 0.15 }}
            className="absolute pointer-events-none z-10 bg-[var(--ds-background-200)] border border-[var(--ds-gray-200)] rounded px-2.5 py-1.5 text-xs font-mono shadow-lg whitespace-nowrap"
            style={{ left: `${(tip.x / SIZE) * 100}%`, top: `${(tip.y / SIZE) * 100}%`, transform: "translate(-50%, -140%)" }}
          >
            {tip.content}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Pixel({ x, y, size, fill, pulse, ring, onEnter, onLeave }: {
  x: number; y: number; size: number; fill: string; pulse?: boolean; ring?: string;
  onEnter?: () => void; onLeave?: () => void;
}) {
  return (
    <rect
      x={x - size / 2} y={y - size / 2} width={size} height={size} fill={fill}
      className={pulse ? "pixel-pulse" : undefined}
      stroke={ring} strokeWidth={ring ? size * 0.28 : undefined} style={{ paintOrder: "stroke" }}
      onMouseEnter={onEnter} onMouseLeave={onLeave}
    />
  );
}

export const DRONE_FRAMES = Array.from({ length: 8 }, (_, k) => `/sprites/drone-${k}.png`);
if (typeof window !== "undefined") DRONE_FRAMES.forEach((src) => { const im = new Image(); im.src = src; });

export const rotorFrame = (i: number) => (Math.floor(performance.now() / 45) + i * 3) % 8;

export const heading = (dx: number, dy: number) => (Math.atan2(dx, dy) * 180) / Math.PI;

export function DroneSprite({ x, y, size, angle = 0, frame = 0, ring, onEnter, onLeave }: {
  x: number; y: number; size: number; angle?: number; frame?: number; ring?: string;
  onEnter?: () => void; onLeave?: () => void;
}) {
  return (
    <g transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`} onMouseEnter={onEnter} onMouseLeave={onLeave}>
      {ring && <circle r={size * 0.34} fill={ring} fillOpacity={0.35} stroke={ring} strokeWidth={size * 0.06} />}
      <image
        href={DRONE_FRAMES[frame]}
        x={-size / 2} y={-size / 2} width={size} height={size}
        transform={`rotate(${angle.toFixed(1)})`}
        style={{ imageRendering: "pixelated" }}
      />
    </g>
  );
}

export function Triangle({ x, y, size, fill }: { x: number; y: number; size: number; fill: string }) {
  const s = size / 2.3;
  return (
    <polygon
      transform={`translate(${x} ${y}) scale(${s})`}
      points="0,-2.3 -2,1.2 2,1.2"
      fill={fill}
      stroke="var(--ds-background-100)" strokeWidth={0.6} style={{ paintOrder: "stroke" }}
    />
  );
}

export function polyline(proj: Proj, pts: XY[]) {
  return pts.map((q) => `${proj.x(q[0]).toFixed(1)},${proj.y(q[1]).toFixed(1)}`).join(" ");
}

export function LegendItem({ mark, children }: { mark: ReactNode; children: ReactNode }) {
  return <li className="flex items-center gap-2 text-sm text-gray-900 uppercase font-mono">{mark}{children}</li>;
}
export const Sq = ({ color, ring }: { color: string; ring?: string }) => (
  <span aria-hidden="true" className="inline-block size-2.5 shrink-0" style={{ background: color, outline: ring ? `2px solid ${ring}` : undefined, outlineOffset: 1 }} />
);
export const Ln = ({ color, w = 2, dash }: { color: string; w?: number; dash?: boolean }) => (
  <span aria-hidden="true" className="inline-block w-4 shrink-0" style={{ borderTop: `${w}px ${dash ? "dashed" : "solid"} ${color}` }} />
);
export const DroneIcon = ({ ring }: { ring?: string }) => (
  <span aria-hidden="true" className="relative inline-flex size-5 shrink-0 items-center justify-center">
    {ring && <span className="absolute inset-0.5 rounded-full" style={{ background: ring, opacity: 0.35, boxShadow: `0 0 0 1.5px ${ring}` }} />}
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img src={DRONE_FRAMES[0]} alt="" width={20} height={20} className="relative" style={{ imageRendering: "pixelated" }} />
  </span>
);
export const Tri = ({ color }: { color: string }) => <span aria-hidden="true" className="text-[10px] leading-none" style={{ color }}>▲</span>;
