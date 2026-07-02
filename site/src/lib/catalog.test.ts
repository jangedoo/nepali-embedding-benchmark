import { describe, expect, it } from "vitest";
import { coverage, formatCount, rank } from "./catalog";
import type { Catalog } from "./types";

describe("catalog helpers", () => {
  it("orders only the supplied task scores", () => {
    expect(rank([
      { model_id: "a", task_id: "x", view_id: "v", metrics: { f1: .2 }, status: "verified", model_revision: "r", dataset_revision: "d", parameter_count: 10, vocab_size: 5 },
      { model_id: "b", task_id: "x", view_id: "v", metrics: { f1: .8 }, status: "community", model_revision: "r", dataset_revision: "d", parameter_count: 20, vocab_size: 5 },
    ], "f1")[0].model_id).toBe("b");
  });

  it("reports partial coverage", () => {
    const catalog = { tasks: [{ id: "x", version: 2, display_name: "X", description: "", dataset: { id: "o/d", revision: "r" }, adapter: "sts", views: [{ id: "v", split: "test", languages: ["ne"], metrics: ["f1"], primary_metric: "f1" }] }], models: [], results: [] } as Catalog;
    expect(coverage(catalog, "missing")).toEqual({ complete: 0, total: 1 });
  });

  it("formats missing model metadata", () => {
    expect(formatCount("unknown")).toBe("unknown");
  });
});
