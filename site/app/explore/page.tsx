import type { Metadata } from "next";
import { Shell } from "../components/SiteHeader";
import { ExploreContainer } from "../components/Views";

export const metadata: Metadata = { title: "Change size · Wildfire Drone Routing" };

export default function Explore() {
  return (
    <Shell sub="Controlled perimeter changes · 288 trials">
      <ExploreContainer />
    </Shell>
  );
}
