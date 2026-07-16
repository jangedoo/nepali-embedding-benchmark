// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CatalogExplorer from "./CatalogExplorer";
import type { Catalog } from "../lib/types";

const catalog: Catalog = {
  schema_version: 3,
  counts: { tasks: 1, models: 2, results: 4 },
  tasks: [{ name: "STSBNepali.v3", display_name: "STS-B Nepali", description: "Similarity", type: "STS", main_score: "cosine_spearman", dataset: { name: "owner/data", revision: "d".repeat(40), url: `https://huggingface.co/datasets/owner/data/tree/${"d".repeat(40)}` }, splits: ["test"], subsets: [{ name: "ne-ne", languages: ["nep-Deva"] }, { name: "ne-en", languages: ["nep-Deva", "eng-Latn"] }] }],
  models: [
    { name: "owner/model", repository: "owner/model", revision: "a".repeat(40), evaluated_at: "2025-01-01", status: "community", effective_prompts: {}, n_parameters: 10, embed_dim: 4, is_latest: false },
    { name: "owner/model", repository: "owner/model", revision: "b".repeat(40), evaluated_at: "2026-01-01", status: "verified", effective_prompts: { query: "query: " }, n_parameters: 10, embed_dim: 4, is_latest: true },
  ],
  results: [
    { model_name: "owner/model", model_revision: "a".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-ne", languages: ["nep-Deva"], metrics: { cosine_spearman: .5, cosine_pearson: .4 }, main_score_name: "cosine_spearman", main_score: .5, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "community", result_path: "old.json", result_sha256: "e".repeat(64), effective_prompts: {}, evaluated_at: "2025-01-01" },
    { model_name: "owner/model", model_revision: "b".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-ne", languages: ["nep-Deva"], metrics: { cosine_spearman: .8, cosine_pearson: .7 }, main_score_name: "cosine_spearman", main_score: .8, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "verified", result_path: "new.json", result_sha256: "f".repeat(64), effective_prompts: { query: "query: " }, evaluated_at: "2026-01-01" },
    { model_name: "owner/model", model_revision: "a".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-en", languages: ["nep-Deva", "eng-Latn"], metrics: { cosine_spearman: .4, cosine_pearson: .3 }, main_score_name: "cosine_spearman", main_score: .4, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "community", result_path: "old.json", result_sha256: "e".repeat(64), effective_prompts: {}, evaluated_at: "2025-01-01" },
    { model_name: "owner/model", model_revision: "b".repeat(40), task_name: "STSBNepali.v3", task_type: "STS", split: "test", subset: "ne-en", languages: ["nep-Deva", "eng-Latn"], metrics: { cosine_spearman: .7, cosine_pearson: .6 }, main_score_name: "cosine_spearman", main_score: .7, dataset_name: "owner/data", dataset_revision: "d".repeat(40), mteb_version: "2.18.3", status: "verified", result_path: "new.json", result_sha256: "f".repeat(64), effective_prompts: { query: "query: " }, evaluated_at: "2026-01-01" },
  ],
};

expect.extend(toHaveNoViolations);
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
    await screen.findByRole("heading", { name: "ne-ne" });
    expect(screen.queryByText("cosine_pearson")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show all metrics" }));
    expect(screen.getAllByRole("columnheader", { name: "cosine_pearson" })).toHaveLength(2);
    expect(document.querySelectorAll(".ranking-table.expanded-metrics")).toHaveLength(2);
  });

  it("distinguishes the main metric and the best score in every metric", async () => {
    render(<CatalogExplorer mode="tasks" />);
    await screen.findByRole("heading", { name: "ne-ne" });
    fireEvent.click(screen.getByLabelText("Show older revisions"));
    fireEvent.click(screen.getByRole("button", { name: "Show all metrics" }));
    expect(document.querySelectorAll("thead .main-metric")).toHaveLength(2);
    expect(document.querySelectorAll("td.main-metric")).toHaveLength(4);
    expect(document.querySelectorAll("td.best-score")).toHaveLength(4);
    expect(screen.getByText("0.8000").closest("td")).toHaveClass("main-metric", "best-score");
  });

  it("shows all subsets by default and supports filtering them together", async () => {
    render(<CatalogExplorer mode="tasks" />);
    await screen.findByRole("heading", { name: "ne-ne" });
    expect(screen.getByRole("heading", { name: "ne-en" })).toBeInTheDocument();
    expect(screen.getByText("All 2 subsets")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByText("Select at least one subset.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "ne-en" }));
    expect(screen.getByRole("heading", { name: "ne-en" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "ne-ne" })).not.toBeInTheDocument();
  });

  it("shows compact model size and provides formatted inline evidence", async () => {
    const { container } = render(<CatalogExplorer mode="tasks" />);
    const details = (await screen.findAllByLabelText(/Result details/))[0];
    expect(screen.getAllByTitle("Number of model parameters")[0]).toHaveTextContent("10 params");
    fireEvent.click(details);
    const evidence = screen.getByRole("region", { name: "Result details for owner/model" });
    expect(evidence).toHaveTextContent("MTEB version");
    expect(evidence).toHaveTextContent("2.18.3");
    expect(evidence).toHaveTextContent("Effective prompts");
    expect(screen.getAllByRole("link", { name: "owner/model" })[0]).toHaveAttribute("href", expect.stringContaining("b".repeat(40)));
    expect(await axe(container)).toHaveNoViolations();
  });
});
