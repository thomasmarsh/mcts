// RunStatusBadge — one small pill describing where a tuner run stands.
// Fed from the operational journal (`status` + `terminal_outcome`) and,
// when the projection has caught up, its `report_status` / validation
// claim. Pure presentation: no data access.

import { type Component } from "solid-js";
import type { TerminalOutcome, TunerRunLiveness } from "../tuner-types.js";

export interface RunStatusBadgeProps {
  status: TunerRunLiveness | null;
  terminalOutcome?: TerminalOutcome | null;
  reportStatus?: string | null;
}

interface Badge {
  label: string;
  cls: string;
}

function badge(props: RunStatusBadgeProps): Badge {
  if (props.status === "live") return { label: "live", cls: "badge-running" };
  if (props.reportStatus) {
    const s = props.reportStatus.toLowerCase();
    if (s.includes("complete") || s === "ok") return { label: props.reportStatus, cls: "badge-completed" };
    return { label: props.reportStatus, cls: "badge-crashed" };
  }
  if (props.terminalOutcome === "exited") return { label: "exited", cls: "badge-completed" };
  if (props.terminalOutcome === "signalled") return { label: "stopped", cls: "badge-stopped" };
  if (props.terminalOutcome === "spawn_failed") return { label: "spawn failed", cls: "badge-crashed" };
  if (props.status === "exited") return { label: "exited", cls: "badge-completed" };
  return { label: "unknown", cls: "badge-unknown" };
}

export const RunStatusBadge: Component<RunStatusBadgeProps> = (props) => {
  const b = (): Badge => badge(props);
  return (
    <span class={`status-badge ${b().cls}`} data-testid="run-status-badge">
      {b().label}
    </span>
  );
};
