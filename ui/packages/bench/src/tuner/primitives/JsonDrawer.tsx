// JsonDrawer — a collapsible pretty-printed JSON escape hatch. Kept
// de-emphasised: it's the "show me the raw artifact" fallback behind the
// shaped evidence views, not a primary path.

import { createMemo, type Component } from "solid-js";

export const JsonDrawer: Component<{
  title: string;
  value: unknown;
  open?: boolean;
  testid?: string;
}> = (props) => {
  const text = createMemo(() => {
    try {
      return JSON.stringify(props.value ?? null, null, 2);
    } catch {
      return String(props.value);
    }
  });
  return (
    <details class="tuner-json-drawer" open={props.open} data-testid={props.testid ?? "json-drawer"}>
      <summary>{props.title}</summary>
      <pre class="tuner-json-drawer-body">{text()}</pre>
    </details>
  );
};
