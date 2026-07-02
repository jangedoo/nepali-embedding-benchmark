import type { Catalog, Result } from "./types";

export function rank(results: Result[], metric: string): Result[] {
  return [...results].sort((a, b) => b.metrics[metric] - a.metrics[metric] || a.model_id.localeCompare(b.model_id));
}

export function formatCount(value: number | "unknown"): string {
  if (value === "unknown") return value;
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(value >= 10_000_000_000 ? 0 : 1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`;
  return value.toLocaleString("en-US");
}

export function coverage(catalog: Catalog, modelId: string): { complete: number; total: number } {
  const expected = catalog.tasks.reduce((count, task) => count + task.views.length, 0);
  const present = new Set(
    catalog.results.filter((result) => result.model_id === modelId).map((result) => `${result.task_id}/${result.view_id}`),
  );
  return { complete: present.size, total: expected };
}

export function initialQuery(name: string): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get(name);
}
