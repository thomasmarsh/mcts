// star-points.ts — Standard Go hoshi (star point) layouts, purely a visual
// reference grid for the eye; no goban game attaches rules to them. Follows
// the traditional 5-point layout (four corner points plus tengen) used on
// both 9×9 and 13×13 boards, just with a wider corner inset on 13×13.

/** Cell indices (row-major) of the star points for an `n`×`n` board, or `[]`
 * below 7×7 where a hoshi layout isn't conventional. */
export function standardStarPoints(n: number): number[] {
  if (n < 7) return [];
  const inset = n >= 13 ? 3 : 2;
  const near = inset;
  const far = n - 1 - inset;
  const mid = Math.floor(n / 2);
  const points = new Set<number>();
  for (const r of [near, far]) {
    for (const c of [near, far]) {
      points.add(r * n + c);
    }
  }
  points.add(mid * n + mid);
  return Array.from(points);
}
