// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { axe } from "jest-axe";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Catalog } from "../lib/types";
import CatalogExplorer from "./CatalogExplorer";

const catalog: Catalog = {
  tasks: [{
    id: "task",
    version: 2,
    display_name: "Example task",
    description: "Example description",
    dataset: { id: "owner/dataset", revision: "1234567890abcdef", url: "https://example.com" },
    adapter: "example",
    views: [
      { id: "first", split: "test", languages: ["ne"], metrics: ["f1", "precision"], primary_metric: "f1" },
      { id: "second", split: "test", languages: ["ne"], metrics: ["f1", "precision"], primary_metric: "f1" },
    ],
  }, {
    id: "task-two",
    version: 2,
    display_name: "Second dataset",
    description: "Dataset without selected-model results",
    dataset: { id: "owner/second-dataset", revision: "1234567890abcdef", url: "https://example.com/second" },
    adapter: "example",
    views: [{ id: "only", split: "test", languages: ["ne"], metrics: ["accuracy"], primary_metric: "accuracy" }],
  }],
  models: [
    { id: "model", display_name: "Example model", hf_id: "owner/model", revision: "abcdef", languages: ["ne"], parameter_count: 10_000_000, vocab_size: 30_000 },
    { id: "model-two", display_name: "Second model", hf_id: "owner/model-two", revision: "abcdef", languages: ["ne"], parameter_count: 20_000_000, vocab_size: 40_000 },
  ],
  results: [
    { model_id: "model", task_id: "task", view_id: "first", metrics: { f1: 0.8, precision: 0.95 }, status: "verified", model_revision: "abcdef", dataset_revision: "123456", parameter_count: 10_000_000, vocab_size: 30_000 },
    { model_id: "model", task_id: "task", view_id: "second", metrics: { f1: 0.7, precision: 0.5 }, status: "community", model_revision: "abcdef", dataset_revision: "123456", parameter_count: 10_000_000, vocab_size: 30_000 },
    { model_id: "model-two", task_id: "task", view_id: "first", metrics: { f1: 0.9, precision: 0.7 }, status: "verified", model_revision: "abcdef", dataset_revision: "123456", parameter_count: 20_000_000, vocab_size: 40_000 },
    { model_id: "model-two", task_id: "task", view_id: "second", metrics: { f1: 0.6, precision: 0.8 }, status: "verified", model_revision: "abcdef", dataset_revision: "123456", parameter_count: 20_000_000, vocab_size: 40_000 },
  ],
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("task views", () => {
  it("shows all views and metrics by default and supports multi-selection", async () => {
    vi.stubGlobal("__NEB_BASE__", "/");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => catalog }));
    const { container } = render(<CatalogExplorer mode="tasks" />);

    await screen.findByRole("heading", { name: "first" });
    expect(screen.getByRole("heading", { name: "second" })).not.toBeNull();
    expect(screen.getByText("All views")).not.toBeNull();
    expect((screen.getByRole("checkbox", { name: "first" }) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole("checkbox", { name: "second" }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getAllByRole("columnheader", { name: /f1/ })).toHaveLength(2);
    expect(screen.getAllByRole("columnheader", { name: "precision" })).toHaveLength(2);
    expect(screen.queryByRole("columnheader", { name: "Parameters" })).toBeNull();
    expect(screen.queryByRole("columnheader", { name: "Vocabulary" })).toBeNull();
    expect(screen.getAllByRole("checkbox", { name: "precision" }).every((checkbox) => (checkbox as HTMLInputElement).checked)).toBe(true);
    expect(container.querySelector(".pareto-chart")).toBeNull();
    expect(screen.queryByLabelText("Verified result")).toBeNull();
    expect(screen.getAllByLabelText("Community-contributed result")).toHaveLength(1);

    const detailsTrigger = screen.getAllByLabelText("Model details for Example model")[0];
    const popoverId = detailsTrigger.getAttribute("popovertarget")!;
    const popover = container.querySelector<HTMLElement>(`#${popoverId}`)!;
    expect(popover.getAttribute("popover")).toBe("auto");
    const modelLink = popover.querySelector<HTMLAnchorElement>("a")!;
    expect(modelLink.getAttribute("href")).toBe("https://huggingface.co/owner/model");
    expect(popover.textContent).toContain("Parameters: 10M");
    expect(popover.textContent).toContain("Vocabulary: 30K");

    fireEvent.click(screen.getAllByRole("checkbox", { name: "precision" })[0]);
    expect(screen.getAllByRole("columnheader", { name: "precision" })).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() => expect(screen.queryByRole("heading", { name: "first" })).toBeNull());
    expect(screen.getByText("Select at least one view.")).not.toBeNull();

    fireEvent.click(screen.getByRole("checkbox", { name: "first" }));
    expect(await screen.findByRole("heading", { name: "first" })).not.toBeNull();
    expect(screen.queryByRole("heading", { name: "second" })).toBeNull();
    expect(screen.getByText("1 of 2 views")).not.toBeNull();
  });
});

describe("model comparison", () => {
  it("groups views, defaults filters, and summarizes the collapsed editor", async () => {
    window.history.replaceState({}, "", "/compare/?models=model,model-two");
    vi.stubGlobal("__NEB_BASE__", "/");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => catalog }));
    const { container } = render(<CatalogExplorer mode="compare" />);

    await screen.findByRole("rowheader", { name: "first" });
    expect(screen.getByRole("rowheader", { name: "second" })).not.toBeNull();
    expect(screen.getByRole("rowheader", { name: "only" })).not.toBeNull();
    expect(container.querySelectorAll(".dataset-heading")).toHaveLength(2);
    expect(container.querySelector(".dataset-heading")?.textContent).toBe("Example task");
    expect(screen.queryByText("task/first")).toBeNull();
    expect(screen.getByText("2 models · 2 datasets · 3 metrics")).not.toBeNull();
    expect((container.querySelector(".comparison-editor") as HTMLDetailsElement).open).toBe(false);

    const modelHeaders = screen.getAllByRole("columnheader").slice(1);
    expect(modelHeaders.map((header) => header.textContent)).toEqual(["Example model?Parameters: 10MVocabulary: 30Kowner/model", "Second model?Parameters: 20MVocabulary: 40Kowner/model-two"]);
    expect(screen.getAllByLabelText(/Model details for/)).toHaveLength(2);
    expect(container.querySelectorAll("[data-metric='f1']")).toHaveLength(4);
    expect(container.querySelectorAll("[data-metric='precision']")).toHaveLength(4);
    expect(container.querySelectorAll(".primary-winner")).toHaveLength(2);
    expect(container.querySelectorAll(".secondary-winner")).toHaveLength(2);
    expect(container.querySelectorAll(".secondary-metric").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".missing")).toHaveLength(2);
    expect(screen.getAllByText("Missing result")).toHaveLength(2);
    expect(screen.queryByLabelText("Verified result")).toBeNull();
    expect(screen.getAllByLabelText("Community-contributed result")).toHaveLength(1);

    const firstRow = screen.getByRole("rowheader", { name: "first" }).closest("tr")!;
    expect(within(firstRow).queryByRole("combobox")).toBeNull();
    expect(container.querySelector(".comparison-table-wrap")).not.toBeNull();

    fireEvent.click(screen.getByText("Edit comparison"));
    expect((container.querySelector(".comparison-editor") as HTMLDetailsElement).open).toBe(true);
    expect(screen.getAllByRole("checkbox").every((checkbox) => (checkbox as HTMLInputElement).checked)).toBe(true);
    const primary = screen.getByRole("checkbox", { name: "f1 for Example task" });
    const accuracy = screen.getByRole("checkbox", { name: "accuracy for Second dataset" });
    expect((primary as HTMLInputElement).disabled).toBe(true);
    expect((accuracy as HTMLInputElement).disabled).toBe(true);

    const precision = screen.getByRole("checkbox", { name: "precision for Example task" });
    expect((precision as HTMLInputElement).checked).toBe(true);
    fireEvent.click(precision);
    expect(container.querySelectorAll("[data-metric='precision']")).toHaveLength(0);
    expect(container.querySelectorAll("[data-metric='f1']")).toHaveLength(4);
    expect(screen.getByText("2 models · 2 datasets · 2 metrics")).not.toBeNull();

    fireEvent.click(screen.getByRole("checkbox", { name: "Second dataset" }));
    expect(screen.queryByRole("rowheader", { name: "only" })).toBeNull();
    expect(container.querySelectorAll(".dataset-heading")).toHaveLength(1);
    expect(screen.getByText("2 models · 1 dataset · 1 metric")).not.toBeNull();

    const report = await axe(container);
    expect(report.violations).toHaveLength(0);
  });
});
