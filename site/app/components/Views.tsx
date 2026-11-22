"use client";

import dynamic from "next/dynamic";

/* The pages read data files and the URL in the browser, so they render on the client only
   (the same pattern as the template's MapContainer). */
const loading = () => <div className="w-full h-[560px] bg-gray-alpha-100 animate-pulse rounded-md" />;

export const ReplayContainer = dynamic(() => import("./ReplayView"), { ssr: false, loading });
export const ExploreContainer = dynamic(() => import("./ExploreView"), { ssr: false, loading });
