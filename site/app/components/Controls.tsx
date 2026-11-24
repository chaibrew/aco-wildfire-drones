"use client";

import type { ReactNode } from "react";

export function Field({ label, children, className }: { label: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`flex min-w-0 flex-col gap-1.5 ${className || ""}`}>
      <span className="font-mono text-xs uppercase tracking-tight text-gray-900 flex items-center gap-1.5">{label}</span>
      {children}
    </div>
  );
}

export function Seg<T extends string | number>({ value, options, onChange, className, label }: {
  value: T;
  options: { value: T; label: ReactNode; color?: string }[];
  onChange: (v: T) => void;
  className?: string;
  label?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className={`flex flex-wrap gap-px bg-gray-alpha-200 w-fit max-w-full ${className || ""}`}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={`font-mono text-xs uppercase tracking-tight px-2.5 h-8 inline-flex items-center gap-1.5 cursor-pointer transition-colors duration-150 outline-none focus-visible:ring ${on ? "bg-gray-1000 text-background-100" : "bg-background-100 text-gray-900 hover:text-gray-1000 hover:bg-gray-alpha-100"}`}
          >
            {o.color && <span aria-hidden="true" style={{ color: o.color }}>■</span>}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Select({ value, onChange, groups, disabled, label }: {
  value: string;
  onChange: (v: string) => void;
  groups: { label: string; options: { value: string; label: string }[] }[];
  disabled?: boolean;
  label: string;
}) {
  return (
    <select
      aria-label={label}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="font-mono text-xs uppercase tracking-tight h-8 px-2 pr-7 w-full sm:w-80 bg-background-100 text-gray-1000 border border-gray-alpha-400 rounded-none outline-none focus-visible:ring disabled:opacity-40 cursor-pointer"
    >
      {groups.map((g) => (
        <optgroup key={g.label} label={g.label}>
          {g.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </optgroup>
      ))}
    </select>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className="font-mono text-xs uppercase tracking-tight h-5 px-1.5 border border-gray-alpha-400 text-gray-1000 cursor-pointer hover:bg-gray-alpha-100"
    >
      {checked ? "[on]" : "[off]"}
    </button>
  );
}
