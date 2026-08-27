// pieces.tsx — Per-color piece icon glyphs drawn on the board's black-
// background hex cells: each of Ingenious's six colors gets a distinct
// filled shape (star, hexagon, circle, wavy star, ring), not just a flat
// color swatch, so a cell's color reads even under color-vision deficiency
// and stays legible against the busy multi-color board at a glance.

import { type Component, Match, Switch } from "solid-js";
import { COLOR_HEX } from "./summary.js";
import type { Color } from "./types.js";

/** Vertices of a regular flat-top hexagon centered at (cx, cy). Shared by
 * the board's own cell outlines and the Orange piece glyph, which is drawn
 * as a smaller hexagon of the same shape. */
export function hexPoints(cx: number, cy: number, size: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    pts.push(`${cx + size * Math.cos(angle)},${cy + size * Math.sin(angle)}`);
  }
  return pts.join(" ");
}

/** Vertices of a `points`-pointed star polygon (alternating outer/inner
 * radius), pointing straight up. */
function starPoints(
  cx: number,
  cy: number,
  points: number,
  outerR: number,
  innerR: number,
): string {
  const pts: string[] = [];
  const step = Math.PI / points;
  for (let i = 0; i < points * 2; i++) {
    const angle = -Math.PI / 2 + i * step;
    const r = i % 2 === 0 ? outerR : innerR;
    pts.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
  }
  return pts.join(" ");
}

/** A near-circular polygon whose radius wobbles sinusoidally -- an "obtuse",
 * many-pointed star with rounded-looking points rather than sharp spikes,
 * approximated as a smooth polygon of many samples. 11 sine periods over the
 * full turn give 11 peaks + 11 troughs, i.e. a ~22-sided outline. */
function wavyStarPoints(cx: number, cy: number, r: number): string {
  const bumps = 11;
  const amplitude = 0.16;
  const samples = 132;
  const pts: string[] = [];
  for (let i = 0; i < samples; i++) {
    const theta = (i / samples) * Math.PI * 2;
    const radius = r * (1 + amplitude * Math.sin(bumps * theta));
    pts.push(`${cx + radius * Math.cos(theta)},${cy + radius * Math.sin(theta)}`);
  }
  return pts.join(" ");
}

/** One color's board glyph, centered at (cx, cy) and sized to fit within a
 * hex of circumradius `r`:
 * Red = 12-pointed star, Blue = 6-pointed star, Orange = filled hexagon,
 * Green = filled circle, Yellow = ~22-sided wavy star, Purple = a thick
 * ring ("donut"). */
export const PieceIcon: Component<{ color: Color; cx: number; cy: number; r: number }> = (
  props,
) => {
  const fill = () => COLOR_HEX[props.color];
  return (
    <Switch>
      <Match when={props.color === "Red"}>
        <polygon
          class="ingenious-icon"
          points={starPoints(props.cx, props.cy, 12, props.r, props.r * 0.55)}
          style={{ fill: fill() }}
        />
      </Match>
      <Match when={props.color === "Blue"}>
        <polygon
          class="ingenious-icon"
          points={starPoints(props.cx, props.cy, 6, props.r, props.r * 0.45)}
          style={{ fill: fill() }}
        />
      </Match>
      <Match when={props.color === "Orange"}>
        <polygon
          class="ingenious-icon"
          points={hexPoints(props.cx, props.cy, props.r * 0.85)}
          style={{ fill: fill() }}
        />
      </Match>
      <Match when={props.color === "Green"}>
        <circle
          class="ingenious-icon"
          cx={props.cx}
          cy={props.cy}
          r={props.r * 0.72}
          style={{ fill: fill() }}
        />
      </Match>
      <Match when={props.color === "Yellow"}>
        <polygon
          class="ingenious-icon"
          points={wavyStarPoints(props.cx, props.cy, props.r * 0.85)}
          style={{ fill: fill() }}
        />
      </Match>
      <Match when={props.color === "Purple"}>
        <circle
          class="ingenious-icon"
          cx={props.cx}
          cy={props.cy}
          r={props.r * 0.5}
          style={{ fill: "none", stroke: fill(), "stroke-width": `${props.r * 0.5}px` }}
        />
      </Match>
    </Switch>
  );
};
