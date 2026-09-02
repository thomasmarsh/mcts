// CopyPresetButton — copies a candidate's config to the clipboard as a
// `presets.json` blob. The serialisation lives in `preset-copy.ts`; this
// owns only the clipboard call and the transient "copied" / "failed"
// announcement.

import { createSignal, Show, type Component } from "solid-js";
import type { JsonValue } from "../../types.js";
import { buildPreset } from "../models/preset-copy.js";

export interface CopyPresetButtonProps {
  candidateId: string;
  gameKind: string | null;
  config: JsonValue | null;
  /** Injectable for tests; defaults to `navigator.clipboard`. */
  clipboard?: { writeText(text: string): Promise<void> };
}

export const CopyPresetButton: Component<CopyPresetButtonProps> = (props) => {
  const [state, setState] = createSignal<"idle" | "copied" | "failed">("idle");

  const result = () =>
    props.gameKind
      ? buildPreset({
          candidateId: props.candidateId,
          gameKind: props.gameKind,
          config: props.config,
        })
      : ({ ok: false, reason: "unknown game kind" } as const);

  const copy = async (): Promise<void> => {
    const r = result();
    if (!r.ok) {
      setState("failed");
      return;
    }
    const writer =
      props.clipboard ?? (globalThis.navigator?.clipboard as CopyPresetButtonProps["clipboard"]);
    try {
      if (!writer) throw new Error("no clipboard");
      await writer.writeText(r.text);
      setState("copied");
      setTimeout(() => setState("idle"), 2000);
    } catch {
      setState("failed");
    }
  };

  return (
    <span class="tuner-copy-preset">
      <button
        type="button"
        disabled={!result().ok}
        onClick={() => void copy()}
        data-testid="copy-preset"
      >
        Copy preset
      </button>
      <Show when={state() === "copied"}>
        <span class="tuner-copy-ok" role="status">
          Copied
        </span>
      </Show>
      <Show when={state() === "failed"}>
        <span class="tuner-copy-fail" role="alert">
          {result().ok ? "Copy failed" : (result() as { reason: string }).reason}
        </span>
      </Show>
    </span>
  );
};
