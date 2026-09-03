// EventTicker — a fixed-height, auto-scrolling feed of one-line evidence
// events, newest at the bottom. Layout only: the lines are formatted by
// `tickerLines` in evidence-fold.ts. Auto-scroll pauses while the pointer is
// over the list so a reader can look back without being yanked down.

import { For, Show, createEffect, createSignal, type Component } from "solid-js";
import type { TickerLine } from "../tuner-types.js";

const VISIBLE = 60;

export const EventTicker: Component<{ lines: TickerLine[]; emptyLabel?: string }> = (props) => {
  const [expanded, setExpanded] = createSignal(false);
  const [hovered, setHovered] = createSignal(false);
  let list: HTMLDivElement | undefined;

  const shown = (): TickerLine[] =>
    expanded() || props.lines.length <= VISIBLE ? props.lines : props.lines.slice(-VISIBLE);
  const hiddenCount = (): number => Math.max(0, props.lines.length - shown().length);

  createEffect(() => {
    // Re-run when the line count changes; stay put while hovered.
    const count = props.lines.length;
    if (list && !hovered() && count >= 0) list.scrollTop = list.scrollHeight;
  });

  return (
    <section class="tuner-event-ticker" data-testid="event-ticker">
      <Show when={hiddenCount() > 0}>
        <button class="tuner-ticker-more" onClick={() => setExpanded(true)}>
          {hiddenCount()} older events
        </button>
      </Show>
      <div
        class="tuner-ticker-lines"
        ref={list}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <Show
          when={props.lines.length > 0}
          fallback={
            <p class="tuner-ticker-empty">{props.emptyLabel ?? "Waiting for the first event…"}</p>
          }
        >
          <For each={shown()}>
            {(line) => (
              <div class="tuner-ticker-line">
                <span class="tuner-ticker-seq">{line.seq}</span>
                <span class="tuner-ticker-text">{line.text}</span>
              </div>
            )}
          </For>
        </Show>
      </div>
    </section>
  );
};
