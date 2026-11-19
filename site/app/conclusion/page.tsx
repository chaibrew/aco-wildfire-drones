import type { Metadata } from "next";
import { Shell } from "../components/SiteHeader";
import { ConclusionContainer } from "../components/Views";

export const metadata: Metadata = { title: "Conclusion · Wildfire Drone Routing" };

export default function Conclusion() {
  return (
    <Shell sub="What the data supports">
      <ConclusionContainer />
    </Shell>
  );
}
