import type { Catalog, Model, Result } from "./types";

export function rank(results: Result[]): Result[] {
  return [...results].sort((left, right) => right.main_score - left.main_score ||
    left.model_name.localeCompare(right.model_name) ||
    left.model_revision.localeCompare(right.model_revision));
}

export function visibleModels(models: Model[], showOlder: boolean): Model[] {
  return showOlder ? models : models.filter((model) => model.is_latest);
}

export function modelKey(model: Pick<Model, "name" | "revision">): string {
  return `${model.name}@${model.revision}`;
}

export function resultKey(result: Pick<Result, "model_name" | "model_revision">): string {
  return `${result.model_name}@${result.model_revision}`;
}

export function coverage(catalog: Catalog, model: Model): { complete: number; total: number } {
  const complete = new Set(catalog.results
    .filter((result) => resultKey(result) === modelKey(model))
    .map((result) => `${result.task_name}/${result.split}/${result.subset}`)).size;
  const total = catalog.tasks.reduce(
    (count, task) => count + task.splits.length * task.subsets.length,
    0,
  );
  return { complete, total };
}

export function formatCount(value: number | "unknown"): string {
  if (value === "unknown") return "unknown";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function initialQuery(name: string): string {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get(name) || "";
}
