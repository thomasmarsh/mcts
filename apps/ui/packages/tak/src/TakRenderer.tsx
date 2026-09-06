// TakRenderer.tsx — Tak's three.js board. Scene setup (renderer/camera/
// OrbitControls/animate loop/resize/raycasting) is ported directly from
// `ui/packages/druid/src/DruidRenderer.tsx`; what's specific to Tak:
//
// - Pieces are built from the TPS string in `props.state.tps` via
//   `parseTps`, not from a custom JSON cell array -- TPS is the standard
//   Tak board format shared by the ecosystem.
// - Piece geometry, built once as shared geometries from the physical piece
//   dimensions the user supplied: Black gets round "cane" stones with a flat
//   chord cut, White gets trapezoid stones -- a shape-by-color distinction
//   some physical Tak sets use so pieces read apart by touch/silhouette, not
//   just color (see the dimension constants' own comment). Capstones are the
//   same domed shape for both.
// - Unplaced reserves render as piles beside the board (west = White, east =
//   Black), not just the HUD's numeric counts -- the customary physical-Tak
//   layout for seeing your own resources at a glance.
// - Move picking has two shapes: placement modes (`legalMoves` already
//   mode-filtered by `GameShell` to one kind) work exactly like Druid --
//   click a highlighted empty cell. The `Move stack` mode is two-stage: click
//   one of your stacks to select it (`selectedSrc`, component-local -- see
//   its own comment below for why), then either click an unambiguous
//   highlighted landing cell or pick from the candidate list panel, since a
//   single source can have many legal (direction, take, drop-schedule)
//   combinations that plain raycasting can't disambiguate.

import {
  type Component,
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
} from "solid-js";
import { For, Show } from "solid-js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GameRendererProps } from "@mcts/game";
import {
  coordFor,
  footprintFor,
  isFlatPlacement,
  isPlacement,
  parsePtn,
  type ParsedMove,
} from "./move-codec.js";
import { parseTps, type ParsedStack } from "./tps-parser.js";
import type { GameState, GameView, Move, Player } from "./types.js";
import "./tak.css";

// --- Piece dimensions, normalized to board-square-side = 1 (see plan/user-
// supplied 5x5 physical dimensions: e.g. cane stone diameter 2.57 / square
// side 4.40 = 0.584). Resolution-independent of board size N, since the
// board itself is rendered at 1 world unit per cell pitch (same convention
// as Druid) regardless of N.
//
// Flats/walls are shaped differently per color -- Black gets round "cane"
// stones, White gets trapezoid stones (both physical shapes the user
// supplied dimensions for) -- a tactile/visual distinction some physical Tak
// sets use so pieces read apart by shape, not just color. Both fit the same
// square footprint envelope so neither color's pieces look mismatched in
// scale on the board.
const PIECE_THICKNESS = 0.205;
const CANE_DIAMETER = 0.584;
const CANE_RADIUS = CANE_DIAMETER / 2;
// Depth of the flat cut into the circle -- also what a standing wall
// actually rests on: a round piece can't balance on its curved edge, so the
// chord cut is what makes standing it up on end physically stable.
const CANE_SAGITTA = 0.116;
const TRAPEZOID_LONG = 0.584;
const TRAPEZOID_SHORT = 0.35;
const TRAPEZOID_DEPTH = 0.584;
const CAPSTONE_DIAMETER = 0.502;
const CAPSTONE_HEIGHT = 0.793;

const WHITE_COLOR = 0xf2e9d8;
const BLACK_COLOR = 0x3a3d46;
const MOVE_HILITE = 0x52c2ee;
const SELECT_HILITE = 0xffd166;
const ANALYSIS_HEAT_COLOR = 0xffa94d;
const ANALYSIS_PROVEN_COLOR = 0x4caf7a;
const SUGGESTED_RING_COLOR = "#ffe066";
const WINNER_GLOW_BLACK = "#8f9bff";
const WINNER_GLOW_WHITE = "#ffd98a";

const WOOD_LIGHT = "#d9bb8e";
const WOOD_DARK = "#8a5a34";
const WOOD_GRAIN_LIGHT = "#c7a575";
const WOOD_GRAIN_DARK = "#6f4527";

// --- Shared geometry (built once, module scope -- same pattern as Druid's
// shared `cubeGeo`) -----------------------------------------------------

/** A "D"-shaped profile: a circle with one flat chord cut, matching the
 * cane stones' physical shape. The chord gap is centered at angle 0 (the
 * shape's local +X), so the flat face's outward normal points toward +X --
 * the same local convention `trapezoidProfile` uses for its own flat edge,
 * so both shapes share one flat/wall orientation pipeline below. */
function caneProfile(): THREE.Shape {
  const d = CANE_RADIUS - CANE_SAGITTA;
  const halfAngle = Math.acos(Math.min(1, Math.max(-1, d / CANE_RADIUS)));
  const shape = new THREE.Shape();
  shape.absarc(0, 0, CANE_RADIUS, halfAngle, Math.PI * 2 - halfAngle, false);
  shape.closePath();
  return shape;
}

/** An isosceles trapezoid: long base at local +X (mirroring `caneProfile`'s
 * chord), short base at local -X, both centered on y = 0. */
function trapezoidProfile(): THREE.Shape {
  const shape = new THREE.Shape();
  shape.moveTo(TRAPEZOID_DEPTH / 2, TRAPEZOID_LONG / 2);
  shape.lineTo(TRAPEZOID_DEPTH / 2, -TRAPEZOID_LONG / 2);
  shape.lineTo(-TRAPEZOID_DEPTH / 2, -TRAPEZOID_SHORT / 2);
  shape.lineTo(-TRAPEZOID_DEPTH / 2, TRAPEZOID_SHORT / 2);
  shape.closePath();
  return shape;
}

/** Extrudes a flat profile to `PIECE_THICKNESS`, oriented flat (footprint in
 * XZ, height along Y) with its local origin translated so it rests on y = 0. */
function buildFlatGeometry(profile: THREE.Shape): THREE.BufferGeometry {
  const geo = new THREE.ExtrudeGeometry(profile, {
    depth: PIECE_THICKNESS,
    bevelEnabled: false,
    curveSegments: 24,
  });
  geo.translate(0, 0, -PIECE_THICKNESS / 2);
  geo.rotateX(-Math.PI / 2);
  geo.computeBoundingBox();
  const minY = geo.boundingBox?.min.y ?? 0;
  geo.translate(0, -minY, 0);
  geo.computeBoundingBox();
  return geo;
}

/** The same flat profile, tipped up 90° onto its flat local-+X edge (the
 * cane's chord, or the trapezoid's long base) -- the wall orientation.
 * Rotating the already flat-oriented geometry about world Z pivots it onto
 * that edge (see the file header: the chord is what makes a round piece
 * stable standing up), leaving a thin standing piece rather than a
 * lying-flat one. */
function buildWallGeometry(flatGeo: THREE.BufferGeometry): THREE.BufferGeometry {
  const geo = flatGeo.clone();
  geo.translate(0, -(geo.boundingBox?.max.y ?? 0) / 2, 0); // re-center before rotating about the origin
  geo.rotateZ(-Math.PI / 2);
  geo.computeBoundingBox();
  const minY = geo.boundingBox?.min.y ?? 0;
  geo.translate(0, -minY, 0);
  geo.computeBoundingBox();
  return geo;
}

function buildCapGeometry(): THREE.BufferGeometry {
  const radius = CAPSTONE_DIAMETER / 2;
  const length = Math.max(0, CAPSTONE_HEIGHT - CAPSTONE_DIAMETER);
  const geo = new THREE.CapsuleGeometry(radius, length, 8, 16);
  geo.computeBoundingBox();
  const minY = geo.boundingBox?.min.y ?? 0;
  geo.translate(0, -minY, 0);
  geo.computeBoundingBox();
  return geo;
}

const CANE_FLAT_GEO = buildFlatGeometry(caneProfile());
const CANE_WALL_GEO = buildWallGeometry(CANE_FLAT_GEO);
const TRAPEZOID_FLAT_GEO = buildFlatGeometry(trapezoidProfile());
const TRAPEZOID_WALL_GEO = buildWallGeometry(TRAPEZOID_FLAT_GEO);
const CAP_GEO = buildCapGeometry();

// Rim outlines for each piece shape, derived once from the shared
// geometries above (see `edgesFor`): stacked pieces of the same color are
// otherwise indistinguishable from a single tall piece from most camera
// angles, since a flat's top face exactly matches the next flat's bottom
// face with nothing to break up the silhouette. A thin rim at each level's
// boundary is what lets a stack's height actually be read at a glance.
const CANE_FLAT_EDGES = new THREE.EdgesGeometry(CANE_FLAT_GEO, 15);
const CANE_WALL_EDGES = new THREE.EdgesGeometry(CANE_WALL_GEO, 15);
const TRAPEZOID_FLAT_EDGES = new THREE.EdgesGeometry(TRAPEZOID_FLAT_GEO, 15);
const TRAPEZOID_WALL_EDGES = new THREE.EdgesGeometry(TRAPEZOID_WALL_GEO, 15);
const CAP_EDGES = new THREE.EdgesGeometry(CAP_GEO, 15);

const SHARED_GEOMETRIES: ReadonlySet<THREE.BufferGeometry> = new Set([
  CANE_FLAT_GEO,
  CANE_WALL_GEO,
  TRAPEZOID_FLAT_GEO,
  TRAPEZOID_WALL_GEO,
  CAP_GEO,
  CANE_FLAT_EDGES,
  CANE_WALL_EDGES,
  TRAPEZOID_FLAT_EDGES,
  TRAPEZOID_WALL_EDGES,
  CAP_EDGES,
]);

/** Black gets the round cane shape, White the trapezoid (see the dimension
 * constants' comment above) -- capstones are the same domed shape for both,
 * since only flats/walls get the shape-by-color distinction. */
function geometryFor(color: Player, topKind: ParsedStack["topKind"]): THREE.BufferGeometry {
  if (topKind === "Cap") return CAP_GEO;
  const wall = color === "Black" ? CANE_WALL_GEO : TRAPEZOID_WALL_GEO;
  const flat = color === "Black" ? CANE_FLAT_GEO : TRAPEZOID_FLAT_GEO;
  return topKind === "Wall" ? wall : flat;
}

/** The rim-outline counterpart to `geometryFor`, for the same piece shape. */
function edgesFor(color: Player, topKind: ParsedStack["topKind"]): THREE.BufferGeometry {
  if (topKind === "Cap") return CAP_EDGES;
  const wall = color === "Black" ? CANE_WALL_EDGES : TRAPEZOID_WALL_EDGES;
  const flat = color === "Black" ? CANE_FLAT_EDGES : TRAPEZOID_FLAT_EDGES;
  return topKind === "Wall" ? wall : flat;
}

const PIECE_EDGE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x000000,
  transparent: true,
  opacity: 0.35,
});

/** A gap left between stacked pieces' Y positions, on top of
 * `PIECE_THICKNESS` -- purely cosmetic (there's no gameplay meaning to the
 * spacing), just enough to catch a sliver of ambient shadow between levels
 * so a stack doesn't read as one solid block from the side. */
const STACK_GAP = 0.014;

function pieceMaterial(color: Player): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: color === "White" ? WHITE_COLOR : BLACK_COLOR,
    roughness: 0.45,
    metalness: 0.05,
  });
}

// --- Board (procedural hardwood inlay) ----------------------------------

function woodTexture(base: string, streak: string): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = streak;
  ctx.globalAlpha = 0.3;
  for (let i = 0; i < 12; i++) {
    const y = (i / 12) * size + Math.sin(i * 12.9898) * 5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(size * 0.3, y + 7, size * 0.7, y - 7, size, y);
    ctx.lineWidth = 1.5 + (i % 3);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return new THREE.CanvasTexture(canvas);
}

function disposeMaterial(mat: THREE.Material | THREE.Material[] | undefined): void {
  if (!mat) return;
  if (Array.isArray(mat)) {
    mat.forEach(disposeMaterial);
    return;
  }
  const withMap = mat as THREE.Material & { map?: THREE.Texture | null };
  withMap.map?.dispose();
  mat.dispose();
}

function clearGroup(group: THREE.Group): void {
  while (group.children.length) {
    const child = group.children.pop();
    if (!child) break;
    if ("geometry" in child) {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry && !SHARED_GEOMETRIES.has(mesh.geometry)) {
        mesh.geometry.dispose();
      }
    }
    if ("material" in child) disposeMaterial((child as THREE.Mesh).material);
  }
}

/** A coordinate-letter/number label baked flush into the bezel -- a flat
 * plane textured with the glyph and laid down with the same rotation as the
 * cell tiles (rather than a `Sprite`, which always faces the camera and so
 * would read as a decal floating above the board instead of inlaid into
 * it), so it stays flush with the wood surface as the camera orbits. */
function makeLabelPlane(text: string): THREE.Mesh {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "bold 88px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#e9d9b8";
  ctx.fillText(text, size / 2, size / 2 + 4);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.42), material);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

// --- Client-side winning-road lookup (display only -- the server's `view`
// already carries the authoritative `terminal`/`winner`; this just finds
// *which* cells to glow, mirroring DruidRenderer.tsx's own client-side
// `findWinningPath`, which duplicates the same kind of path-finding for its
// minimap glow). Tak roads can connect either edge pair for either player
// (no fixed per-player axis, unlike Druid), so both axes are tried. ---

function topOwner(cell: ParsedStack | null): Player | null {
  return cell ? (cell.colors[cell.colors.length - 1] ?? null) : null;
}
function isRoadCell(cell: ParsedStack | null, player: Player): boolean {
  return !!cell && cell.topKind !== "Wall" && topOwner(cell) === player;
}

function bfsPath(
  cells: (ParsedStack | null)[],
  n: number,
  player: Player,
  starts: number[],
  goal: Set<number>,
): number[] | null {
  const prev = new Map<number, number>();
  const queue = starts.filter((i) => isRoadCell(cells[i] ?? null, player));
  queue.forEach((i) => prev.set(i, -1));
  let reached = -1;
  for (let head = 0; head < queue.length && reached < 0; head++) {
    const cur = queue[head]!;
    if (goal.has(cur)) {
      reached = cur;
      break;
    }
    const cx = cur % n;
    const cy = Math.floor(cur / n);
    const neighbors: [number, number][] = [
      [cx - 1, cy],
      [cx + 1, cy],
      [cx, cy - 1],
      [cx, cy + 1],
    ];
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
      const ni = ny * n + nx;
      if (!isRoadCell(cells[ni] ?? null, player) || prev.has(ni)) continue;
      prev.set(ni, cur);
      queue.push(ni);
    }
  }
  if (reached < 0) return null;
  const path: number[] = [];
  for (let i = reached; i >= 0; i = prev.get(i) ?? -1) path.push(i);
  return path.reverse();
}

/** Exported for its own pure-function test (tests/winning-road.test.ts) --
 * this is real reimplemented road-connectivity logic (see the file header),
 * not just wiring, so it's worth testing directly against known board
 * layouts rather than only through a full component render. */
export function findWinningRoad(
  cells: (ParsedStack | null)[],
  size: number,
  winner: Player,
): number[] | null {
  const n = size;
  const west: number[] = [],
    east = new Set<number>();
  const south: number[] = [],
    north = new Set<number>();
  for (let row = 0; row < n; row++) {
    west.push(row * n);
    east.add(row * n + (n - 1));
  }
  for (let col = 0; col < n; col++) {
    south.push(col);
    north.add((n - 1) * n + col);
  }
  return bfsPath(cells, n, winner, west, east) ?? bfsPath(cells, n, winner, south, north);
}

// --- Component -----------------------------------------------------------

/** One clickable target: either an unambiguous move (fires immediately) or
 * a source-selection target (enters the two-stage spread picker). */
type Pickable = {
  mesh: THREE.Mesh;
  target: { kind: "move"; move: Move } | { kind: "selectSrc"; square: number };
};

export const TakRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;

  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let controls: OrbitControls;
  let raycaster: THREE.Raycaster;
  let boardGroup: THREE.Group;
  let piecesGroup: THREE.Group;
  let highlightGroup: THREE.Group;
  let ghostGroup: THREE.Group;
  let analysisGroup: THREE.Group;
  let winnerGroup: THREE.Group;
  let reservesGroup: THREE.Group;
  let pickables: Pickable[] = [];
  const mouse = new THREE.Vector2();
  let animationHandle = 0;
  let builtSize: number | null = null;

  // Parsed TPS: cached so every piece/highlight rebuild doesn't re-parse.
  const parsed = createMemo(() => parseTps(props.state.tps));

  // Which stack (if any) the "Move stack" mode has selected as a spread
  // source. Component-local (not lifted to the store) -- purely a renderer
  // interaction detail, the same level as `hoveredMove`, but nothing outside
  // this renderer needs it (there's no analysis-panel equivalent to preview
  // it from).
  const [selectedSrc, setSelectedSrc] = createSignal<number | null>(null);

  function n(): number {
    return parsed().size;
  }

  function buildBoard(size: number): void {
    clearGroup(boardGroup);
    const lightMat = new THREE.MeshStandardMaterial({
      map: woodTexture(WOOD_LIGHT, WOOD_GRAIN_LIGHT),
      roughness: 0.55,
    });
    const darkMat = new THREE.MeshStandardMaterial({
      map: woodTexture(WOOD_DARK, WOOD_GRAIN_DARK),
      roughness: 0.55,
    });
    const cellGeo = new THREE.PlaneGeometry(1, 1);

    // The bezel extends `BEZEL_MARGIN` past the outermost cell edge on every
    // side -- wide enough that the coordinate labels below (inset less than
    // that) land on the wood border itself instead of floating past it.
    const BEZEL_MARGIN = 0.6;
    const bezel = new THREE.Mesh(
      new THREE.PlaneGeometry(size + BEZEL_MARGIN * 2, size + BEZEL_MARGIN * 2),
      new THREE.MeshStandardMaterial({ color: WOOD_DARK, roughness: 0.8 }),
    );
    bezel.rotation.x = -Math.PI / 2;
    bezel.position.set((size - 1) / 2, -0.03, (size - 1) / 2);
    boardGroup.add(bezel);

    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const cell = new THREE.Mesh(cellGeo, (row + col) % 2 === 0 ? lightMat : darkMat);
        cell.rotation.x = -Math.PI / 2;
        cell.position.set(col, -0.01, row);
        boardGroup.add(cell);
      }
    }

    const labelInset = BEZEL_MARGIN * 0.55;
    for (let i = 0; i < size; i++) {
      const letter = String.fromCharCode(97 + i);
      [-0.5 - labelInset, size - 0.5 + labelInset].forEach((z) => {
        const label = makeLabelPlane(letter);
        label.position.set(i, -0.02, z);
        boardGroup.add(label);
      });
    }
    for (let j = 0; j < size; j++) {
      const number = String(j + 1);
      [-0.5 - labelInset, size - 0.5 + labelInset].forEach((x) => {
        const label = makeLabelPlane(number);
        label.position.set(x, -0.02, j);
        boardGroup.add(label);
      });
    }

    const center = new THREE.Vector3((size - 1) / 2, 0, (size - 1) / 2);
    controls.target.copy(center);
    camera.position.set(center.x - size * 0.3, Math.max(size, 4) * 1.3, center.z + size * 0.9);
    camera.lookAt(center);
  }

  function addPiece(
    x: number,
    y: number,
    z: number,
    color: Player,
    kind: ParsedStack["topKind"],
  ): void {
    const mesh = new THREE.Mesh(geometryFor(color, kind), pieceMaterial(color));
    mesh.position.set(x, y, z);
    piecesGroup.add(mesh);
    // A separate top-level sibling (not a child of `mesh`) so `clearGroup`'s
    // flat child scan disposes its geometry every rebuild same as any other
    // piece -- a child of `mesh` would otherwise never get disposed there.
    const outline = new THREE.LineSegments(edgesFor(color, kind), PIECE_EDGE_MATERIAL);
    outline.position.set(x, y, z);
    piecesGroup.add(outline);
  }

  function buildPieces(size: number): void {
    clearGroup(piecesGroup);
    const cells = parsed().cells;
    cells.forEach((stack, idx) => {
      if (!stack) return;
      const x = idx % size;
      const z = Math.floor(idx / size);
      let y = 0;
      // Every piece below the top is always flat (walls/capstones can never
      // be covered), so only the top gets its own kind's geometry/height.
      // `STACK_GAP` (on top of each piece's real thickness) is what keeps a
      // same-color stack from reading as one solid block -- see its own
      // comment -- with a rim outline on every level as the second cue.
      for (let level = 0; level < stack.colors.length - 1; level++) {
        addPiece(x, y, z, stack.colors[level]!, "Flat");
        y += PIECE_THICKNESS + STACK_GAP;
      }
      const top = stack.colors[stack.colors.length - 1]!;
      addPiece(x, y, z, top, stack.topKind);
    });
  }

  const RESERVE_PIECES_PER_COLUMN = 6;
  const RESERVE_COLUMN_SPACING = 0.68;

  /** Unplaced reserves piled beside the board -- the customary "see your own
   * resources at a glance" physical-Tak layout, not just the HUD's numeric
   * counts. White piles to the board's west, Black to the east, each stone
   * count arranged in short stacked columns (so 21 stones doesn't render as
   * one absurdly tall tower) with capstones in their own column past them. */
  function buildReservePile(
    color: Player,
    x: number,
    stoneCount: number,
    capCount: number,
    size: number,
  ): void {
    const centerZ = (size - 1) / 2;
    for (let i = 0; i < stoneCount; i++) {
      const col = Math.floor(i / RESERVE_PIECES_PER_COLUMN);
      const level = i % RESERVE_PIECES_PER_COLUMN;
      const mesh = new THREE.Mesh(geometryFor(color, "Flat"), pieceMaterial(color));
      mesh.position.set(x, level * PIECE_THICKNESS, centerZ + col * RESERVE_COLUMN_SPACING);
      reservesGroup.add(mesh);
    }
    const capCol = Math.ceil(stoneCount / RESERVE_PIECES_PER_COLUMN);
    for (let i = 0; i < capCount; i++) {
      const mesh = new THREE.Mesh(CAP_GEO, pieceMaterial(color));
      mesh.position.set(
        x,
        0,
        centerZ + (capCol + 0.6) * RESERVE_COLUMN_SPACING + i * RESERVE_COLUMN_SPACING,
      );
      reservesGroup.add(mesh);
    }
  }

  function buildReserves(size: number): void {
    clearGroup(reservesGroup);
    const west = -1.6;
    const east = size - 1 + 1.6;
    buildReservePile("White", west, props.state.stones[0], props.state.caps[0], size);
    buildReservePile("Black", east, props.state.stones[1], props.state.caps[1], size);
  }

  function stackTopY(idx: number): number {
    const stack = parsed().cells[idx];
    if (!stack) return 0;
    return (stack.colors.length - 1) * (PIECE_THICKNESS + STACK_GAP);
  }

  function highlightPlane(color: number, opacity: number): THREE.Mesh {
    const geo = new THREE.PlaneGeometry(0.86, 0.86);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    return new THREE.Mesh(geo, mat);
  }

  /** Placement modes highlight one pickable per legal move, exactly like
   * Druid. The `Move stack` mode (`legalMoves` are all spreads, pre-filtered
   * by `GameShell`) is two-stage: with no source selected, one pickable per
   * distinct controllable stack; with a source selected, one highlight per
   * cell any of its candidate spreads touches (visual only, except a cell
   * that's the *unique* final landing cell of exactly one candidate, which
   * fires that move directly as a shortcut). */
  function rebuildHighlights(): void {
    clearGroup(highlightGroup);
    pickables = [];
    if (props.busy) return;
    const size = n();
    const legalMoves = props.legalMoves; // PTN strings
    const isSpreadMode = legalMoves.some((m) => !isPlacement(m));

    if (!isSpreadMode) {
      legalMoves.forEach((mv) => {
        const pm = parsePtn(mv, size);
        if (pm.tag !== "Place") return;
        const x = pm.square % size;
        const z = Math.floor(pm.square / size);
        const plane = highlightPlane(MOVE_HILITE, 0.55);
        plane.rotation.x = -Math.PI / 2;
        plane.position.set(x, stackTopY(pm.square) + 0.03, z);
        highlightGroup.add(plane);
        pickables.push({ mesh: plane, target: { kind: "move", move: mv } });
      });
      return;
    }

    const src = selectedSrc();
    if (src === null) {
      const sources = new Set(
        legalMoves
          .map((m) => parsePtn(m, size))
          .filter((pm): pm is ParsedMove & { tag: "Spread" } => pm.tag === "Spread")
          .map((pm) => pm.square),
      );
      sources.forEach((square) => {
        const x = square % size;
        const z = Math.floor(square / size);
        const plane = highlightPlane(SELECT_HILITE, 0.5);
        plane.rotation.x = -Math.PI / 2;
        plane.position.set(x, stackTopY(square) + 0.03, z);
        highlightGroup.add(plane);
        pickables.push({ mesh: plane, target: { kind: "selectSrc", square } });
      });
      return;
    }

    const candidates = legalMoves
      .map((m) => ({ ptn: m, parsed: parsePtn(m, size) }))
      .filter(
        (e): e is { ptn: string; parsed: ParsedMove & { tag: "Spread" } } =>
          e.parsed.tag === "Spread" && e.parsed.square === src,
      );
    // Which final-landing cells are unambiguous (exactly one candidate ends
    // there) -- those get a direct-fire pickable; the rest are visual only,
    // resolved through the candidate list panel instead.
    const finalCellCounts = new Map<number, number>();
    candidates.forEach(({ parsed: mv }) => {
      const path = footprintFor(mv, size);
      const last = path[path.length - 1]!;
      finalCellCounts.set(last, (finalCellCounts.get(last) ?? 0) + 1);
    });

    const touched = new Set<number>();
    candidates.forEach(({ parsed: mv }) =>
      footprintFor(mv, size).forEach((cell) => touched.add(cell)),
    );
    touched.forEach((cell) => {
      const x = cell % size;
      const z = Math.floor(cell / size);
      const plane = highlightPlane(MOVE_HILITE, 0.4);
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(x, stackTopY(cell) + 0.03, z);
      highlightGroup.add(plane);
      if (finalCellCounts.get(cell) === 1) {
        const entry = candidates.find(({ parsed: mv }) => footprintFor(mv, size).at(-1) === cell)!;
        pickables.push({ mesh: plane, target: { kind: "move", move: entry.ptn } });
      }
    });

    // The selected source cell itself stays clickable, to deselect.
    const x = src % size;
    const z = Math.floor(src / size);
    const selPlane = highlightPlane(SELECT_HILITE, 0.6);
    selPlane.rotation.x = -Math.PI / 2;
    selPlane.position.set(x, stackTopY(src) + 0.035, z);
    highlightGroup.add(selPlane);
    pickables.push({ mesh: selPlane, target: { kind: "selectSrc", square: src } });
  }

  function buildGhost(move: Move | null): void {
    clearGroup(ghostGroup);
    if (!move || props.busy || !isFlatPlacement(move)) return;
    const size = n();
    const pm = parsePtn(move, size);
    if (pm.tag !== "Place") return;
    const color = props.state.turn;
    const mat = pieceMaterial(color);
    mat.transparent = true;
    mat.opacity = 0.5;
    mat.depthWrite = false;
    const geo = geometryFor(
      color,
      pm.kind === "Wall" ? "Wall" : pm.kind === "Cap" ? "Cap" : "Flat",
    );
    const x = pm.square % size;
    const z = Math.floor(pm.square / size);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, stackTopY(pm.square), z);
    ghostGroup.add(mesh);
  }

  function makeSquareOutline(half: number, color: string): THREE.LineLoop {
    const points = [
      new THREE.Vector3(-half, 0, -half),
      new THREE.Vector3(half, 0, -half),
      new THREE.Vector3(half, 0, half),
      new THREE.Vector3(-half, 0, half),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color }));
  }

  function rebuildAnalysisOverlay(): void {
    clearGroup(analysisGroup);
    const overlay = props.analysisOverlay;
    if (!overlay || overlay.length === 0) return;
    const size = n();
    const maxShare = overlay.reduce((m, e) => Math.max(m, e.visitShare), 0);
    const tileGeo = new THREE.PlaneGeometry(0.78, 0.78);

    overlay.forEach((entry) => {
      const intensity = maxShare > 0 ? entry.visitShare / maxShare : 0;
      const color = entry.isProven ? ANALYSIS_PROVEN_COLOR : ANALYSIS_HEAT_COLOR;
      const opacity = 0.12 + intensity * 0.55;
      const path = footprintFor(parsePtn(entry.move, size), size);
      const cellIdx = path[path.length - 1]!;
      const x = cellIdx % size;
      const z = Math.floor(cellIdx / size);
      const y = stackTopY(cellIdx) + 0.04;

      const tile = new THREE.Mesh(
        tileGeo,
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(x, y, z);
      analysisGroup.add(tile);

      if (entry.isSuggested) {
        const ring = makeSquareOutline(0.46, SUGGESTED_RING_COLOR);
        ring.position.set(x, y + 0.002, z);
        analysisGroup.add(ring);
      }
    });
  }

  /** A `LineBasicMaterial` line renders at a hairline 1px regardless of
   * `linewidth` on most platforms (a long-standing WebGL limitation three.js
   * can't work around), which made the winning road nearly invisible. A
   * `TubeGeometry` ribbon through the path's cell centers, plus a sphere at
   * each joint to round the corners, gives it real on-screen thickness. */
  function rebuildWinnerGlow(): void {
    clearGroup(winnerGroup);
    if (!props.view.terminal || !props.view.winner) return;
    const path = findWinningRoad(parsed().cells, n(), props.view.winner);
    if (!path) return;
    const size = n();
    const glowColor = props.view.winner === "Black" ? WINNER_GLOW_BLACK : WINNER_GLOW_WHITE;
    const points = path.map(
      (i) => new THREE.Vector3(i % size, stackTopY(i) + 0.08, Math.floor(i / size)),
    );
    const mat = new THREE.MeshBasicMaterial({ color: glowColor, transparent: true, opacity: 0.92 });

    if (points.length > 1) {
      const curve = new THREE.CatmullRomCurve3(points, false, "catmullrom", 0);
      const tubeGeo = new THREE.TubeGeometry(
        curve,
        Math.max(points.length * 6, 8),
        0.09,
        12,
        false,
      );
      winnerGroup.add(new THREE.Mesh(tubeGeo, mat));
    }
    const jointGeo = new THREE.SphereGeometry(0.12, 12, 12);
    points.forEach((p) => {
      const joint = new THREE.Mesh(jointGeo, mat);
      joint.position.copy(p);
      winnerGroup.add(joint);
    });
  }

  function pickAt(clientX: number, clientY: number): Pickable["target"] | null {
    if (!canvasRef || pickables.length === 0) return null;
    const rect = canvasRef.getBoundingClientRect();
    mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(
      pickables.map((p) => p.mesh),
      false,
    );
    if (hits.length === 0) return null;
    const hit = pickables.find((p) => p.mesh === hits[0]!.object);
    return hit ? hit.target : null;
  }

  function onClick(event: MouseEvent): void {
    if (props.busy) return;
    const hit = pickAt(event.clientX, event.clientY);
    if (!hit) {
      setSelectedSrc(null);
      return;
    }
    if (hit.kind === "selectSrc") {
      setSelectedSrc((prev) => (prev === hit.square ? null : hit.square));
      return;
    }
    setSelectedSrc(null);
    props.onMove(hit.move);
  }

  function onPointerMove(event: MouseEvent): void {
    if (props.busy) {
      props.onHover(null);
      return;
    }
    const hit = pickAt(event.clientX, event.clientY);
    props.onHover(hit?.kind === "move" ? hit.move : null);
  }

  function onPointerLeave(): void {
    props.onHover(null);
  }

  function onResize(): void {
    if (!canvasRef) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function animate(): void {
    animationHandle = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  onMount(() => {
    if (!canvasRef) return;
    renderer = new THREE.WebGLRenderer({ canvas: canvasRef, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xc7c9cf);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.49;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(8, 14, 6);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xaac4ff, 0.25);
    fill.position.set(-8, 6, -10);
    scene.add(fill);

    boardGroup = new THREE.Group();
    piecesGroup = new THREE.Group();
    highlightGroup = new THREE.Group();
    ghostGroup = new THREE.Group();
    analysisGroup = new THREE.Group();
    winnerGroup = new THREE.Group();
    reservesGroup = new THREE.Group();
    scene.add(
      boardGroup,
      piecesGroup,
      highlightGroup,
      ghostGroup,
      analysisGroup,
      winnerGroup,
      reservesGroup,
    );

    raycaster = new THREE.Raycaster();

    canvasRef.addEventListener("click", onClick);
    canvasRef.addEventListener("mousemove", onPointerMove);
    canvasRef.addEventListener("mouseleave", onPointerLeave);
    window.addEventListener("resize", onResize);

    animate();
  });

  onCleanup(() => {
    cancelAnimationFrame(animationHandle);
    window.removeEventListener("resize", onResize);
    canvasRef?.removeEventListener("click", onClick);
    canvasRef?.removeEventListener("mousemove", onPointerMove);
    canvasRef?.removeEventListener("mouseleave", onPointerLeave);
    clearGroup(boardGroup);
    clearGroup(piecesGroup);
    clearGroup(highlightGroup);
    clearGroup(ghostGroup);
    clearGroup(analysisGroup);
    clearGroup(winnerGroup);
    clearGroup(reservesGroup);
    renderer?.dispose();
  });

  createEffect(() => {
    const size = n();
    if (!boardGroup) return;
    if (builtSize === size) return;
    buildBoard(size);
    builtSize = size;
  });

  createEffect(() => {
    void props.state; // rebuild on every state change (a fresh cell array each time)
    if (!piecesGroup) return;
    buildPieces(n());
  });

  createEffect(() => {
    void props.legalMoves;
    void props.busy;
    void selectedSrc();
    if (!highlightGroup) return;
    rebuildHighlights();
  });

  createEffect(() => {
    if (!ghostGroup) return;
    buildGhost(props.hoveredMove);
  });

  createEffect(() => {
    if (!analysisGroup) return;
    rebuildAnalysisOverlay();
  });

  createEffect(() => {
    if (!winnerGroup) return;
    rebuildWinnerGlow();
  });

  createEffect(() => {
    void props.state.stones;
    void props.state.caps;
    const size = n();
    if (!reservesGroup) return;
    buildReserves(size);
  });

  // A new position (move applied, or a genuinely new game) drops any
  // in-progress source selection -- otherwise a stale `selectedSrc` from the
  // previous position could point at a cell with a totally different stack.
  createEffect(() => {
    void props.state;
    setSelectedSrc(null);
  });

  const candidates = () => {
    const src = selectedSrc();
    if (src === null) return [] as { ptn: string; parsed: ParsedMove & { tag: "Spread" } }[];
    const size = n();
    return props.legalMoves
      .map((m) => ({ ptn: m, parsed: parsePtn(m, size) }))
      .filter(
        (e): e is { ptn: string; parsed: ParsedMove & { tag: "Spread" } } =>
          e.parsed.tag === "Spread" && e.parsed.square === src,
      );
  };

  return (
    <>
      <canvas ref={canvasRef} class="tak-board" />
      <Show when={candidates().length > 0}>
        <div class="tak-candidates">
          <div class="tak-candidates-title">Move from {coordFor(selectedSrc()!, n())}</div>
          <For each={candidates()}>
            {({ ptn }) => (
              <button
                class="tak-candidate"
                onClick={() => {
                  setSelectedSrc(null);
                  props.onMove(ptn);
                }}
              >
                {ptn}
              </button>
            )}
          </For>
        </div>
      </Show>
    </>
  );
};
