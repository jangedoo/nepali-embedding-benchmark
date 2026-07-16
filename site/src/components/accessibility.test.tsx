// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import CatalogExplorer from "./CatalogExplorer";

beforeEach(() => vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ schema_version: 3, counts: { tasks: 0, models: 0, results: 0 }, tasks: [], models: [], results: [] }) }))));
expect.extend(toHaveNoViolations);
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("has no obvious accessibility violations in an empty catalog", async () => {
  const { container } = render(<CatalogExplorer mode="models" />);
  await screen.findByText("No published model revisions yet.");
  expect(await axe(container)).toHaveNoViolations();
});
