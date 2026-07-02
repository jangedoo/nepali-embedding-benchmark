import type { Catalog, Result } from "./types";

export function rank(results: Result[]): Result[] {
  return [...results].sort((a, b) => b.score - a.score || a.model_id.localeCompare(b.model_id));
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

