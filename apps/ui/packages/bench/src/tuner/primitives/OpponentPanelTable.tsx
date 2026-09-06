// OpponentPanelTable — the resolved opponent panel as a table:
// `id | role | weight | resolved config`. A schema-default opponent shows
// its expanded config here, so a hidden `rave` opponent is never a surprise.
// Shared by the launch form's Run plan panel and (for a running run) the
// Run Overview.

import { type JSX } from "solid-js";
import type { RunPlanOpponent } from "../tuner-types.js";
import { DataTable } from "./DataTable.js";

export function OpponentPanelTable(props: {
  opponents: RunPlanOpponent[];
  testid?: string;
}): JSX.Element {
  return (
    <DataTable
      testid={props.testid ?? "opponent-panel-table"}
      empty="No opponents resolved."
      rows={props.opponents}
      rowKey={(row) => row.id}
      columns={[
        { key: "id", header: "id", render: (row) => row.label ?? row.id },
        {
          key: "role",
          header: "role",
          render: (row) => (row.role === "default" ? `${row.role} (${row.source})` : row.role),
        },
        { key: "weight", header: "weight", align: "right", render: (row) => String(row.weight) },
        {
          key: "config",
          header: "resolved config",
          render: (row) => <code class="tuner-mono">{row.config}</code>,
        },
      ]}
    />
  );
}
