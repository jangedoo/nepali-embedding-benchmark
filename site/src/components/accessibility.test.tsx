// @vitest-environment jsdom

import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import React from "react";
import { describe, expect, it } from "vitest";

import { Badge } from "./Badge";

describe("benchmark evidence", () => {
  it("exposes compact evidence icons without accessibility violations", async () => {
    const { container } = render(
      <main>
        <h1>Task ranking</h1>
        <table>
          <caption>Scores for one task view</caption>
          <thead><tr><th>Model</th><th>Score</th></tr></thead>
          <tbody><tr><th scope="row">Example</th><td><Badge status="verified" /></td></tr></tbody>
        </table>
      </main>,
    );
    const report = await axe(container);
    expect(report.violations).toHaveLength(0);
    expect(container.querySelector("[aria-label='Verified result']")).not.toBeNull();
    expect(container.textContent).not.toContain("verified");
  });
});
