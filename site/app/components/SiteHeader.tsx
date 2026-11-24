"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Mission replay" },
  { href: "/explore", label: "Change size" },
];

export function SiteHeader({ sub }: { sub: string }) {
  const path = usePathname();
  return (
    <header className="flex flex-col md:flex-row md:items-start md:justify-between font-mono text-sm xl:text-base uppercase gap-4 mb-8">
      <p className="text-gray-1000 font-mono my-0">
        Wildfire drone routing{" "}
        <span className="block font-mono text-gray-900">[{sub}]</span>
      </p>
      <nav aria-label="Pages">
        <ul className="flex flex-wrap gap-x-4 gap-y-1 list-none pl-0 m-0">
          {NAV.map((n) => {
            const on = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <li key={n.href}>
                <Link
                  href={n.href}
                  aria-current={on ? "page" : undefined}
                  className={`underline-offset-4 hover:underline ${on ? "text-gray-1000 underline" : "text-gray-900"}`}
                >
                  {on ? "▲ " : ""}{n.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-12 font-mono text-xs uppercase tracking-tight text-gray-900">
      UNSW COMP3821 project · Camp Fire perimeters from FEDS (Chen et al., 2022) · Terrain from Copernicus DEM 30 m
    </footer>
  );
}

export function Shell({ sub, children }: { sub: string; children: React.ReactNode }) {
  return (
    <main className="font-mono min-h-screen max-w-[min(100vw,1600px)] mx-auto relative overflow-x-clip md:rounded-md flex flex-col md:block px-4 sm:px-6 pt-8 md:pt-16 pb-10">
      <div className="w-full max-w-[1600px] mx-auto mt-1">
        <SiteHeader sub={sub} />
        {children}
        <SiteFooter />
      </div>
    </main>
  );
}
