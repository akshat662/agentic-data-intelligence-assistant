import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a minimal, self-contained `.next/standalone` server -- only used by
  // web/Dockerfile's self-host build. Vercel builds this project with its own pipeline and
  // is unaffected by this setting either way (documented Next.js/Vercel behavior -- Vercel
  // does not use the standalone output artifact, so it's a no-op there, not a conflict).
  output: "standalone",
};

export default nextConfig;
