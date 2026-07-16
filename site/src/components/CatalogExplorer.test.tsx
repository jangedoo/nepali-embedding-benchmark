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
  window.history.replaceState({}, "", "/");
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
    expect([...container.querySelectorAll<HTMLAnchorElement>('a[href^="http"]')].every((link) => link.target === "_blank" && link.rel === "noreferrer")).toBe(true);
    expect(await axe(container)).toHaveNoViolations();
  });

  it("prioritizes model specifications without a prominent latest-revision label", async () => {
    render(<CatalogExplorer mode="models" />);

    await screen.findByRole("heading", { name: "Models" });
    expect(screen.queryByText("Published native MTEB evidence")).not.toBeInTheDocument();
    expect(screen.queryByText("Latest canonical revision")).not.toBeInTheDocument();
    expect(screen.getByText("Parameters")).toBeInTheDocument();
    expect(screen.getByText("Embedding dimension")).toBeInTheDocument();
    expect(screen.getByText("Coverage")).toBeInTheDocument();
    expect(screen.getByText("2 / 2 task views")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Show older revisions"));
    expect(screen.getByText("Historical revision")).toBeInTheDocument();
  });

  it("builds an accessible side-by-side comparison with explicit missing evidence", async () => {
    const partialCatalog: Catalog = {
      ...catalog,
      results: catalog.results.filter((result) => !(result.model_revision === "a".repeat(40) && result.subset === "ne-en")),
    };
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => partialCatalog })));
    const { container } = render(<CatalogExplorer mode="compare" />);

    await screen.findByRole("heading", { name: "Compare models" });
    expect(screen.getByText("Choose a task, then compare up to five model revisions across every subset and split.")).toBeInTheDocument();
    expect(screen.queryByText(/without constructing/i)).not.toBeInTheDocument();
    expect(screen.getByText("Choose at least two model revisions")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Show older revisions"));
    fireEvent.click(screen.getByRole("checkbox", { name: /owner\/model aaaaaaaa/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /owner\/model bbbbbbbb/ }));

    expect(await screen.findByRole("rowheader", { name: /ne-ne/ })).toBeInTheDocument();
    expect(container.querySelector<HTMLDetailsElement>(".comparison-editor")?.open).toBe(true);
    expect(screen.getByRole("rowheader", { name: /ne-en/ })).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(3);
    expect(screen.getByText("Missing result")).toBeInTheDocument();
    expect(container.querySelectorAll(".metric-winner")).toHaveLength(1);
    expect(screen.getAllByRole("img", { name: "Maintainer-verified result" })).toHaveLength(2);
    expect(screen.getByRole("img", { name: "Community-contributed result" })).toBeInTheDocument();

    fireEvent.click(container.querySelector(".metric-select summary")!);
    fireEvent.click(screen.getByRole("checkbox", { name: "cosine_pearson" }));
    expect(container.querySelectorAll("[data-metric='cosine_pearson']")).toHaveLength(3);

    fireEvent.click(screen.getByRole("button", { name: /Result details for owner\/model bbbbbbbb on ne-ne test/ }));
    expect(screen.getByRole("region", { name: "Result details for owner/model" })).toHaveTextContent("Maintainer verified");
    expect(new URLSearchParams(window.location.search).get("models")).toBe(`owner/model@${"a".repeat(40)},owner/model@${"b".repeat(40)}`);
    expect(new URLSearchParams(window.location.search).get("metrics")).toBe("cosine_pearson");
    expect(await axe(container)).toHaveNoViolations();
  });

  it("restores a linked comparison and ignores unavailable metrics", async () => {
    const oldKey = `owner/model@${"a".repeat(40)}`;
    const newKey = `owner/model@${"b".repeat(40)}`;
    const params = new URLSearchParams({ task: "STSBNepali.v3", models: `${oldKey},${newKey}`, metrics: "cosine_pearson,not_a_metric" });
    window.history.replaceState({}, "", `/compare/?${params}`);
    const { container } = render(<CatalogExplorer mode="compare" />);

    expect(await screen.findByRole("rowheader", { name: /ne-ne/ })).toBeInTheDocument();
    expect(container.querySelectorAll(".selected-model")).toHaveLength(2);
    expect(screen.getByText("historical")).toBeInTheDocument();
    expect(container.querySelectorAll("[data-metric='cosine_pearson']")).toHaveLength(4);
    expect(new URLSearchParams(window.location.search).get("metrics")).toBe("cosine_pearson");
  });
});
