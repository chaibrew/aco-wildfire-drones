"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function InfoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 7V11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="8" cy="5" r="0.75" fill="currentColor" />
    </svg>
  );
}

function PixelGridTransition({
  firstContent,
  secondContent,
  isActive,
  gridSize = 30,
  animationStepDuration = 0.3,
  className,
}: {
  firstContent: React.ReactNode;
  secondContent: React.ReactNode;
  isActive: boolean;
  gridSize?: number;
  animationStepDuration?: number;
  className?: string;
}) {
  const [showPixels, setShowPixels] = useState(false);
  const [animState, setAnimState] = useState<"idle" | "growing" | "shrinking">("idle");
  const hasActivatedRef = useRef(false);

  const pixels = useMemo(() => {
    const total = gridSize * gridSize;
    const result = [];
    for (let n = 0; n < total; n++) {
      const row = Math.floor(n / gridSize);
      const col = n % gridSize;
      const color = Math.random() > 0.85 ? "var(--ds-blue-800, #0070f3)" : "var(--ds-gray-200, #333)";
      result.push({ id: n, row, col, color });
    }
    return result;
  }, [gridSize]);

  const [shuffledOrder, setShuffledOrder] = useState<number[]>([]);

  useEffect(() => {
    if (!hasActivatedRef.current && !isActive) return;
    if (isActive) hasActivatedRef.current = true;

    const indices = pixels.map((_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    setShuffledOrder(indices);

    setShowPixels(true);
    setAnimState("growing");

    const shrinkTimer = setTimeout(() => setAnimState("shrinking"), animationStepDuration * 1000);
    const hideTimer = setTimeout(() => {
      setShowPixels(false);
      setAnimState("idle");
    }, animationStepDuration * 2000);

    return () => {
      clearTimeout(shrinkTimer);
      clearTimeout(hideTimer);
    };
  }, [isActive, animationStepDuration, pixels]);

  const delayPerPixel = useMemo(() => animationStepDuration / pixels.length, [animationStepDuration, pixels.length]);
  const orderMap = useMemo(() => {
    const map = new Map<number, number>();
    shuffledOrder.forEach((idx, order) => map.set(idx, order));
    return map;
  }, [shuffledOrder]);

  return (
    <div className={`w-full overflow-hidden max-w-full relative ${className || ""}`}>
      <motion.div
        className="h-full"
        aria-hidden={isActive}
        initial={{ opacity: 1 }}
        animate={{ opacity: isActive ? 0 : 1 }}
        transition={{ duration: 0, delay: animationStepDuration }}
      >
        {firstContent}
      </motion.div>

      <motion.div
        className="absolute inset-0 w-full h-full z-[2] overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: isActive ? 1 : 0 }}
        transition={{ duration: 0, delay: animationStepDuration }}
        style={{ pointerEvents: isActive ? "auto" : "none" }}
        aria-hidden={!isActive}
      >
        {secondContent}
      </motion.div>

      <div
        className="absolute inset-0 w-full h-full pointer-events-none z-[3]"
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
        }}
      >
        <AnimatePresence>
          {showPixels &&
            pixels.map((pixel) => {
              const order = orderMap.get(pixel.id) ?? 0;
              return (
                <motion.div
                  key={pixel.id}
                  style={{
                    backgroundColor: pixel.color,
                    aspectRatio: "1 / 1",
                    gridArea: `${pixel.row + 1} / ${pixel.col + 1}`,
                  }}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{
                    opacity: animState === "growing" ? 1 : 0,
                    scale: animState === "growing" ? 1 : 0,
                  }}
                  transition={{ duration: 0.01, delay: order * delayPerPixel }}
                />
              );
            })}
        </AnimatePresence>
      </div>
    </div>
  );
}

export function StatCard({
  title,
  value,
  children,
  infoContent,
  links,
  className,
}: {
  title: ReactNode;
  value?: ReactNode;
  children?: ReactNode;
  infoContent?: ReactNode;
  links?: [string, string][];
  className?: string;
}) {
  const [showInfo, setShowInfo] = useState(false);

  const statsContent = (
    <div className="bg-gray-alpha-100 p-4 md:p-6 w-full min-h-[120px] h-full">
      <div className="space-y-2">
        <h2 className="my-0 font-mono font-medium text-sm tracking-tight uppercase text-gray-1000 pr-8">
          {title}
        </h2>
        {value !== undefined && (
          <div className="text-3xl md:text-4xl tracking-normal font-mono tabular-nums">{value}</div>
        )}
        {children}
      </div>
    </div>
  );

  const infoContentView = (
    <div className="bg-gray-alpha-100 p-4 md:p-6 w-full h-full overflow-y-auto flex flex-col gap-y-2">
      <span className="my-0 font-mono font-medium text-sm tracking-tight uppercase text-gray-1000 shrink-0 pr-8">
        {title}
      </span>
      <div className="tracking-tight text-sm text-gray-900 leading-relaxed space-y-2 [&_b]:text-gray-1000 [&_b]:font-medium">
        {infoContent}
      </div>
      {links && links.length > 0 && (
        <div className="flex flex-col gap-1 mt-auto pt-2">
          {links.map(([href, label]) => (
            <a
              key={href}
              href={href}
              tabIndex={showInfo ? 0 : -1}
              className="font-mono text-sm uppercase tracking-tight text-gray-1000 hover:underline underline-offset-2 inline-flex gap-x-0.5 items-center w-fit"
            >
              {label}
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" clipRule="evenodd" d="M6.75011 4H6.00011V5.5H6.75011H9.43945L5.46978 9.46967L4.93945 10L6.00011 11.0607L6.53044 10.5303L10.499 6.56182V9.25V10H11.999V9.25V5C11.999 4.44772 11.5512 4 10.999 4H6.75011Z" />
              </svg>
            </a>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className={`relative group rounded-md overflow-hidden ${className || ""}`}>
      <PixelGridTransition
        firstContent={statsContent}
        secondContent={infoContentView}
        isActive={showInfo}
        gridSize={30}
        animationStepDuration={0.3}
        className="h-full"
      />
      {infoContent && (
        <div className={`absolute top-2 right-2 transition-opacity duration-150 z-[20] isolate ${showInfo ? "opacity-100" : "opacity-100 md:opacity-0 md:group-hover:opacity-100 md:group-focus-within:opacity-100"}`}>
          <button
            aria-label={showInfo ? "Back" : "How to read this"}
            aria-expanded={showInfo}
            type="button"
            onClick={() => setShowInfo(!showInfo)}
            className="p-1 m-0 bg-transparent text-gray-alpha-600 md:text-gray-900 border-none md:border md:border-solid border-gray-alpha-400 hover:text-gray-1000 hover:bg-gray-alpha-200 transition-colors duration-150 flex items-center justify-center outline-none focus-visible:ring cursor-pointer"
          >
            <InfoIcon />
          </button>
        </div>
      )}
    </div>
  );
}

export function MetricRow({ label, value, sub, color, active, onClick }: {
  label: ReactNode; value: ReactNode; sub?: ReactNode; color?: string; active?: boolean; onClick?: () => void
}) {
  const body = (
    <>
      <h3 className={`m-0 font-mono font-normal text-sm uppercase flex items-center gap-1.5 ${active ? "text-gray-1000" : "text-gray-900"}`}>
        {color && <span aria-hidden="true" style={{ color }}>■</span>}
        {label}
      </h3>
      <div className="flex items-center gap-3 md:gap-4 text-right">
        <div className="text-gray-1000 text-sm font-mono tabular-nums">{value}</div>
        {sub !== undefined && <div className="min-w-16 text-gray-900 text-right text-sm font-mono tabular-nums">{sub}</div>}
      </div>
    </>
  );
  return (
    <li className="flex flex-wrap items-center justify-between gap-x-3">
      {onClick ? (
        <button type="button" onClick={onClick} className={`w-full flex flex-wrap items-center justify-between gap-x-3 text-left cursor-pointer hover:bg-gray-alpha-100 -mx-1 px-1 ${active ? "bg-gray-alpha-100" : ""}`}>
          {body}
        </button>
      ) : body}
    </li>
  );
}

export function BigStat({ label, value, sub }: { label: ReactNode; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="space-y-2">
      <h2 className="my-0 font-mono font-medium text-sm tracking-tight uppercase text-gray-900">{label}</h2>
      <div className="text-4xl md:text-5xl tracking-normal font-mono tabular-nums">{value}</div>
      {sub && <div className="text-sm text-gray-900 font-mono tabular-nums">{sub}</div>}
    </div>
  );
}

export function KeyList({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <h2 className="my-0 font-mono font-medium text-sm tracking-tight uppercase text-gray-900">{label}</h2>
      <ul className="list-none pl-0 space-y-1">{children}</ul>
    </div>
  );
}
export function KeyRow({ color, code, value, sub, active, onClick, title }: {
  color: string; code: ReactNode; value: ReactNode; sub?: ReactNode; active?: boolean; onClick?: () => void; title?: string
}) {
  const cls = `grid grid-cols-[auto_minmax(0,1fr)_5ch_9ch] sm:grid-cols-[auto_minmax(0,1fr)_7ch_11ch] items-center gap-x-2 w-full text-left ${active ? "bg-gray-alpha-100" : ""}`;
  const inner = (
    <>
      <span aria-hidden="true" style={{ color }}>■</span>
      <h3 className="my-0 font-medium text-[16px] truncate" style={{ color }}>{code}</h3>
      <span className="text-right tabular-nums">{value}</span>
      <span className="text-right text-gray-900 tabular-nums">{sub}</span>
    </>
  );
  return (
    <li title={title}>
      {onClick ? <button type="button" onClick={onClick} className={`${cls} cursor-pointer hover:bg-gray-alpha-100`}>{inner}</button> : <div className={cls}>{inner}</div>}
    </li>
  );
}

export function CountLine({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="flex items-baseline w-full md:w-fit justify-start mt-2">
      <span aria-hidden="true" className="inline-block translate-y-[-2px] translate-x-[2px] shrink-0">
        <span className="text-[10px]">▲</span>
      </span>
      <div className="text-left">
        <span className="inline-block my-0 font-medium text-[16px] tabular-nums">&nbsp;{value}</span>
        <span className="font-medium text-[16px] text-gray-900 tracking-tight">&nbsp;{label}</span>
      </div>
    </div>
  );
}
