import { Show, createSignal, type Component } from "solid-js";
import { copyPreset, type PresetBuildResult, type PresetCopyState } from "./preset-copy.js";

/** Shared immutable-snapshot copy control with an announced result. */
export const PresetCopyAction: Component<{ label: string; build: PresetBuildResult }> = (props) => {
  const [status, setStatus] = createSignal<PresetCopyState | null>(null);
  const unavailable = () => !props.build.enabled;
  const announcement = () => status()?.announcement ?? (props.build.enabled ? "" : props.build.reason.message);
  async function copy(): Promise<void> {
    if (!props.build.enabled || !navigator.clipboard) return;
    setStatus(await copyPreset(props.build, navigator.clipboard));
  }
  return (
    <span class="tuning-copy-action">
      <button type="button" disabled={unavailable()} title={unavailable() ? announcement() : `Copy ${props.label} as a preset`} onClick={() => void copy()}>
        {unavailable() ? `${props.label} unavailable` : `Copy ${props.label}`}
      </button>
      <Show when={announcement()}><span class="tuning-copy-status" role="status">{announcement()}</span></Show>
    </span>
  );
};
