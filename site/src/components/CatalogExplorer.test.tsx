// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Catalog } from "../lib/types";
import CatalogExplorer from "./CatalogExplorer";

const catalog: Catalog = {
  tasks: [{
    id: "task",
    version: 1,
    display_name: "Example task",
    description: "Example description",
    dataset: { id: "owner/dataset", revision: "1234567890abcdef", url: "https://example.com" },
    adapter: "example",
    views: [
      { id: "first", split: "test", languages: ["ne"], primary_metric: "f1" },
      { id: "second", split: "test", languages: ["ne"], primary_metric: "f1" },
    ],
  }],
  models: [
    { id: "model", display_name: "Example model", hf_id: "owner/model", revision: "abcdef", languages: ["ne"] },
    { id: "model-two", display_name: "Second model", hf_id: "owner/model-two", revision: "abcdef", languages: ["ne"] },
  ],
  results: [
    { model_id: "model", task_id: "task", view_id: "first", metric: "f1", score: 0.8, status: "verified", model_revision: "abcdef", dataset_revision: "123456" },
    { model_id: "model", task_id: "task", view_id: "second", metric: "f1", score: 0.7, status: "community", model_revision: "abcdef", dataset_revision: "123456" },
    { model_id: "model-two", task_id: "task", view_id: "first", metric: "f1", score: 0.9, status: "verified", model_revision: "abcdef", dataset_revision: "123456" },
    { model_id: "model-two", task_id: "task", view_id: "second", metric: "f1", score: 0.6, status: "verified", model_revision: "abcdef", dataset_revision: "123456" },
  ],
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("task views", () => {
  it("shows all views by default and supports multi-selection", async () => {
    vi.stubGlobal("__NEB_BASE__", "/");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => catalog }));
    render(<CatalogExplorer mode="tasks" />);

    await screen.findByRole("heading", { name: "first" });
    expect(screen.getByRole("heading", { name: "second" })).not.toBeNull();
    expect(screen.getByText("All views")).not.toBeNull();
    expect(screen.getAllByRole("checkbox").every((checkbox) => (checkbox as HTMLInputElement).checked)).toBe(true);

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
  it("shows metrics, highlights row winners, and marks only community results", async () => {
    window.history.replaceState({}, "", "/compare/?models=model,model-two");
    vi.stubGlobal("__NEB_BASE__", "/");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => catalog }));
    const { container } = render(<CatalogExplorer mode="compare" />);

    await screen.findByText("task/first");
    expect(screen.getAllByText("Metric: f1")).toHaveLength(2);
    expect(container.querySelectorAll(".best-result")).toHaveLength(2);
    expect(screen.queryByLabelText("Verified result")).toBeNull();
    expect(screen.getAllByLabelText("Community-contributed result")).toHaveLength(1);
  });
});
