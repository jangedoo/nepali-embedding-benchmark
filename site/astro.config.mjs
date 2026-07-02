import { defineConfig } from "astro/config";
import react from "@astrojs/react";

const base = process.env.BASE_PATH || "/";

export default defineConfig({
  output: "static",
  base,
  integrations: [react()],
  vite: { define: { __NEB_BASE__: JSON.stringify(base) } },
});

