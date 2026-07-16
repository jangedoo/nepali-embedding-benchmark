// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import React from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CatalogExplorer from "./CatalogExplorer";
import type { Catalog } from "../lib/types";

const catalog: Catalog = {
  schema_version: 3,
  counts: { tasks: 1, models: 2, results: 2 },
  tasks: [{ name: "STSBNepali.v3", display_name: "STS-B Nepali", description: "Similarity", type: "STS", main_score: "cosine_spearman", dataset: { name: "owner/data", revision: "d".repeat(40), url: `https://huggingface.co/datasets/owner/data/tree/${"d".repeat(40)}` }, splits: ["test"], subsets: [{ name: "ne-ne", languages: ["nep-Deva"] }] }],
  models: [
    { name: "owner/model", repository: "owner/model", revision: "a".repeat(40), evaluated_at: "2025-01-01", status: "community", effective_prompts: {}, n_parameters: 10, embed_dim: 4, is_latest: false },
    { name: "owner/model", repository: "owner/model", revision: "b".repeat(40), evaluated_at: "2026-01-01", status: "verified", effective_prompts: { query: "query: " }, n_parameters: 10, embed_dim: 4, is_latest: true },
  ],
  results: [
    { model_name: "owner/model", model_revision: "a".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-ne", languages: ["nep-Deva"], metrics: { cosine_spearman: .5, cosine_pearson: .4 }, main_score_name: "cosine_spearman", main_score: .5, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "community", result_path: "old.json", result_sha256: "e".repeat(64), effective_prompts: {}, evaluated_at: "2025-01-01" },
    { model_name: "owner/model", model_revision: "b".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-ne", languages: ["nep-Deva"], metrics: { cosine_spearman: .8, cosine_pearson: .7 }, main_score_name: "cosine_spearman", main_score: .8, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "verified", result_path: "new.json", result_sha256: "f".repeat(64), effective_prompts: { query: "query: " }, evaluated_at: "2026-01-01" },
  ],
};

beforeEach(() => vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => catalog }))));
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("CatalogExplorer", () => {
  it("shows latest revisions by default and exposes history globally", async () => {
    render(<CatalogExplorer mode="tasks" />);
    await screen.findByText("0.8000");
    expect(screen.queryByText("0.5000")).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Show older revisions"));
    expect(await screen.findByText("0.5000")).toBeInTheDocument();
  });

  it("starts with the main score and expands native metrics", async () => {
    render(<CatalogExplorer mode="tasks" />);
    await screen.findByText("cosine_spearman");
    expect(screen.queryByText("cosine_pearson")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show all native metrics" }));
    expect(screen.getByText("cosine_pearson")).toBeInTheDocument();
  });

  it("links exact revisions and provides accessible evidence detail", async () => {
    render(<CatalogExplorer mode="tasks" />);
    const details = await screen.findByLabelText(/Evidence details/);
    fireEvent.click(details);
    await waitFor(() => expect(screen.getByText(/MTEB: 2.18.3/)).toBeInTheDocument());
    expect(screen.getByRole("link", { name: "owner/model" })).toHaveAttribute("href", expect.stringContaining("b".repeat(40)));
  });
});
