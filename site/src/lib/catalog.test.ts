import { describe, expect, it } from "vitest";
import { coverage, rank } from "./catalog";
import type { Catalog } from "./types";

describe("catalog helpers", () => {
  it("orders only the supplied task scores", () => {
    expect(rank([
      { model_id: "a", task_id: "x", view_id: "v", metric: "f1", score: .2, status: "verified", model_revision: "r", dataset_revision: "d" },
      { model_id: "b", task_id: "x", view_id: "v", metric: "f1", score: .8, status: "community", model_revision: "r", dataset_revision: "d" },
    ])[0].model_id).toBe("b");
  });

  it("reports partial coverage", () => {
    const catalog = { tasks: [{ id: "x", version: 1, display_name: "X", description: "", dataset: { id: "o/d", revision: "r" }, adapter: "sts", views: [{ id: "v", split: "test", languages: ["ne"], primary_metric: "f1" }] }], models: [], results: [] } as Catalog;
    expect(coverage(catalog, "missing")).toEqual({ complete: 0, total: 1 });
  });
});

