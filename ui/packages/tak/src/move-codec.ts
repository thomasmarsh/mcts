// move-codec.ts — PTN (Portable Tak Notation) parse/format helpers.
// Moves on the wire are PTN strings, not a custom JSON shape -- see
// games/tak/src/main.rs's `WireMove` removal and `Move::from_ptn`.
//
// Internal representation (`ParsedMove`) mirrors the old `WireMove` tagged
// union -- the renderer and mode filters parse from PTN into this shape
// on demand; the PTN string is what goes back over the wire unchanged.

export type Direction = "North" | "East" | "South" | "West";

/** Matches `games/tak/src/lib.rs`'s `DIRS` (indexed the same way `Move::dir()`
 * is: 0 = North, 1 = East, 2 = South, 3 = West), in board coordinates where
 * row 0 is the south edge -- so North is `+row`. */
const DIR_DELTA: Record<Direction, [number, number]> = {
  North: [0, 1],
  East: [1, 0],
  South: [0, -1],
  West: [-1, 0],
};

const DIR_GLYPH: Record<Direction, string> = {
  North: "+",
  East: ">",
  South: "-",
  West: "<",
};

/** Parsed move shape -- the internal representation the renderer and mode
 * filters use. The PTN string itself is the canonical wire format. */
export type ParsedMove =
  | { tag: "Place"; square: number; kind: "Flat" | "Wall" | "Cap" }
  | { tag: "Spread"; square: number; direction: Direction; drop_sizes: number[] };

/** PTN direction glyph to compass direction. */
function parseDirectionGlyph(glyph: string): Direction {
  switch (glyph) {
    case "+": return "North";
    case ">": return "East";
    case "-": return "South";
    case "<": return "West";
    default: throw new Error(`Invalid PTN direction '${glyph}'`);
  }
}

/** Parse a PTN move string into a `ParsedMove`. `n` is the board width. */
export function parsePtn(ptn: string, n: number): ParsedMove {
  const trimmed = ptn.trim();
  if (trimmed.length === 0) throw new Error("empty PTN string");

  const dirMatch = /[+\-<>]/.exec(trimmed);
  if (!dirMatch) return parsePlacement(trimmed, n);

  const dirPos = dirMatch.index!;
  const dirGlyph = dirMatch[0]!;
  const direction = parseDirectionGlyph(dirGlyph);
  const beforeDir = trimmed.slice(0, dirPos);
  const afterDir = trimmed.slice(dirPos + 1);

  // Before the direction: optional take count, then the coordinate.
  const coordStart = beforeDir.search(/[a-h]/);
  if (coordStart < 0) throw new Error(`No coordinate in spread '${trimmed}'`);
  const takeStr = beforeDir.slice(0, coordStart);
  const coordStr = beforeDir.slice(coordStart);

  const take = takeStr.length === 0 ? 1 : parseInt(takeStr, 10);
  if (isNaN(take) || take <= 0 || take > n) {
    throw new Error(`Take count ${takeStr} out of range (1..${n})`);
  }

  const square = parseCoord(coordStr, n);

  let dropSizes: number[];
  if (afterDir.length === 0) {
    dropSizes = take === 1 ? [1] : [take];
  } else {
    dropSizes = [];
    for (const ch of afterDir) {
      const d = parseInt(ch, 10);
      if (isNaN(d) || d <= 0) throw new Error(`Invalid drop digit '${ch}'`);
      dropSizes.push(d);
    }
    const sum = dropSizes.reduce((a, b) => a + b, 0);
    if (sum !== take) throw new Error(`Drop counts sum to ${sum} but take is ${take}`);
  }

  return { tag: "Spread", square, direction, drop_sizes: dropSizes };
}

function parsePlacement(ptn: string, n: number): ParsedMove {
  let kind: "Flat" | "Wall" | "Cap" = "Flat";
  let coordStr = ptn;
  if (ptn.startsWith("S")) {
    kind = "Wall";
    coordStr = ptn.slice(1);
  } else if (ptn.startsWith("C")) {
    kind = "Cap";
    coordStr = ptn.slice(1);
  } else if (ptn.startsWith("F")) {
    coordStr = ptn.slice(1);
  }
  const square = parseCoord(coordStr, n);
  return { tag: "Place", square, kind };
}

/** Parse a PTN coordinate like `a1`, `c3`, `h6` into a row-major square
 * index (row * n + col, row 0 = south edge). */
function parseCoord(coord: string, n: number): number {
  if (coord.length < 2) throw new Error(`Invalid coordinate '${coord}'`);
  const colChar = coord[0]!;
  const col = colChar.charCodeAt(0) - 97; // 'a' -> 0
  const row = parseInt(coord.slice(1), 10);
  if (isNaN(row) || row < 1 || row > n || col < 0 || col >= n) {
    throw new Error(`Coordinate '${coord}' out of bounds for a ${n}x${n} board`);
  }
  return (row - 1) * n + col;
}

/** The ordered path of board indices a move touches: `[square]` for a
 * placement, `[src, ...one per drop]` for a spread (in walk order). */
export function footprintFor(move: ParsedMove, n: number): number[] {
  if (move.tag === "Place") return [move.square];
  const [dc, dr] = DIR_DELTA[move.direction];
  let col = move.square % n;
  let row = Math.floor(move.square / n);
  const path = [move.square];
  for (let i = 0; i < move.drop_sizes.length; i++) {
    col += dc;
    row += dr;
    path.push(row * n + col);
  }
  return path;
}

/** Column + row string for a square index (e.g. `a1`, `c3`). */
export function coordFor(square: number, n: number): string {
  const col = square % n;
  const row = Math.floor(square / n);
  return `${String.fromCharCode(97 + col)}${row + 1}`;
}

/** Format a `ParsedMove` back into a PTN string. Not needed on the wire
 * (moves are already PTN strings), but useful for test round-trips and as
 * the inverse of `parsePtn`. */
export function notation(move: ParsedMove, n: number): string {
  const at = coordFor(move.square, n);
  if (move.tag === "Place") {
    const prefix = move.kind === "Wall" ? "S" : move.kind === "Cap" ? "C" : "";
    return `${prefix}${at}`;
  }
  const take = move.drop_sizes.reduce((a, b) => a + b, 0);
  const prefix = take > 1 ? String(take) : "";
  const suffix = take > 1 ? move.drop_sizes.join("") : "";
  return `${prefix}${at}${DIR_GLYPH[move.direction]}${suffix}`;
}

/** Quick PTN move classifier for mode filters (no full parse): placement
 * flat/wall/cap vs. spread. Direction glyphs `+`/`-`/`<`/`>` only appear
 * in spreads. */
export function isPlacement(ptn: string): boolean {
  return !/[+\-<>]/.test(ptn);
}
export function isFlatPlacement(ptn: string): boolean {
  return isPlacement(ptn) && !ptn.startsWith("S") && !ptn.startsWith("C");
}
export function isWallPlacement(ptn: string): boolean {
  return isPlacement(ptn) && ptn.startsWith("S");
}
export function isCapPlacement(ptn: string): boolean {
  return isPlacement(ptn) && ptn.startsWith("C");
}