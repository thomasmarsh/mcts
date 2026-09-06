// SaveLoadPanel.tsx — Save/load: fully client-side,
// no server round-trip -- the client already holds the full
// game tree, so Export/Import serialize/parse `{gameKind, config, tree}`
// straight from/into local state via `@mcts/game`'s `save-load.ts`.
//
// Per the hard rule, this component never touches the network --
// its only output is `onLoad(gameKind, config, tree)`, which GameShell wires
// to a `load` dispatch. File I/O here is local Blob/`<input type=file>`
// mechanics, not `fetch`, so it doesn't run afoul of that rule (same
// reasoning GameTree's own network-free reducer relies on).

import { type Component, createSignal, Show } from "solid-js";
import type { GameTree } from "@mcts/game";
import { parseSave, serializeSave } from "@mcts/game";
import { GAME_MODULES } from "./games.js";

type S = unknown;
type M = unknown;

export const SaveLoadPanel: Component<{
  gameKind: string;
  config: unknown;
  tree: GameTree<S, M>;
  onLoad: (gameKind: string, config: unknown, tree: GameTree<S, M>) => void;
}> = (props) => {
  const [error, setError] = createSignal<string | null>(null);
  let fileInput: HTMLInputElement | undefined;

  function exportGame(): void {
    const json = serializeSave(props.gameKind, props.config, props.tree);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${props.gameKind}-${new Date().toISOString().replace(/[:.]/g, "-")}.game.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function onFileChange(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ""; // allow re-selecting the same file after an error
    if (!file) return;
    setError(null);
    try {
      const text = await file.text();
      const save = parseSave<S, M>(text);
      if (!GAME_MODULES[save.gameKind]) throw new Error(`Unknown game kind "${save.gameKind}".`);
      props.onLoad(save.gameKind, save.config, save.tree);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load save file.");
    }
  }

  return (
    <div id="save-load">
      <div class="save-load-buttons">
        <button onClick={exportGame}>Save</button>
        <button onClick={() => fileInput?.click()}>Load</button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={(e) => void onFileChange(e)}
      />
      <Show when={error()}>{(e) => <div class="save-load-error">{e()}</div>}</Show>
    </div>
  );
};
