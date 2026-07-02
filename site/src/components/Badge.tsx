import React from "react";

import type { Status } from "../lib/types";

export function Badge({ status }: { status: Status }) {
  const label = status === "verified" ? "Verified result" : "Community-contributed result";
  return <span className={`evidence-icon ${status}`} role="img" aria-label={label} title={label}>
    {status === "verified"
      ? <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m7.5 12 3 3 6-7" /></svg>
      : <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M8.5 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7-1a2.5 2.5 0 1 0 0-5M3 19c0-3 2.5-5 5.5-5s5.5 2 5.5 5m1-5c3 0 5 2 5 4.5" /></svg>}
  </span>;
}
