import { describe, expect, it } from "vitest";
import { modelKey, rank, visibleModels } from "./catalog";
import type { Model, Result } from "./types";

const models: Model[] = [
  { name: "owner/model", repository: "owner/model", revision: "a".repeat(40), evaluated_at: "2025-01-01", status: "verified", effective_prompts: {}, n_parameters: 10, embed_dim: 4, is_latest: false },
  { name: "owner/model", repository: "owner/model", revision: "b".repeat(40), evaluated_at: "2026-01-01", status: "verified", effective_prompts: {}, n_parameters: 10, embed_dim: 4, is_latest: true },
];

function result(score: number, revision: string): Result {
  return { model_name: "owner/model", model_revision: revision, task_name: "task", task_type: "STS", split: "test", subset: "ne-ne", languages: ["nep-Deva"], metrics: { cosine_spearman: score }, main_score_name: "cosine_spearman", main_score: score, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "verified", result_path: "result.json", result_sha256: "f".repeat(64), effective_prompts: {}, evaluated_at: null };
}

describe("catalog helpers", () => {
  it("defaults to latest canonical revisions", () => {
    expect(visibleModels(models, false).map(modelKey)).toEqual([`owner/model@${"b".repeat(40)}`]);
    expect(visibleModels(models, true)).toHaveLength(2);
  });

  it("ranks only the supplied task view by its main score", () => {
    expect(rank([result(.2, "a".repeat(40)), result(.8, "b".repeat(40))])[0].main_score).toBe(.8);
  });
});
