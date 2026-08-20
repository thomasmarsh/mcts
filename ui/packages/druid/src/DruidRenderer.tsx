// DruidRenderer.tsx — Druid's three.js board, ported from
// server/static/app.js into a typed SolidJS component against the
// `GameRendererProps` contract. Behavior parity with
// app.js is the bar: same scene setup, piece placement, ghost-preview-on-
// hover, minimap, and goal-edge framing -- componentized, and driven by
// props/effects instead of app.js's page-global mutable state and manual
// DOM event wiring.
//
// One notable behavior change from app.js: piece stacking is now built from
// `props.history` via `buildStackModel` (a pure function of the whole move
// sequence, computed fresh on every render) rather than app.js's replay-as-
// you-go bookkeeping -- see layers.ts's header comment for why this is
// strictly more correct, not just a refactor.

import { type Component, createEffect, onCleanup, onMount } from "solid-js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GameRendererProps } from "@mcts/game";
import { buildStackModel, footprintFor, type Beam, type LayerEntry } from "./layers.js";
import type { GameState, GameView, Move, Player, Size } from "./types.js";
import "./druid.css";

const CUBE = 1.0; // full-size so adjacent/stacked blocks touch; the black outline shell is what separates them
const LEVEL_H = 1.0; // vertical spacing per stacked layer

const BLACK_COLOR = 0x3a3d46;
const WHITE_COLOR = 0xf2e9d8;
const MOVE_HILITE = 0x52c2ee;

// The analysis heatmap's color -- a warm hue deliberately distinct from
// MOVE_HILITE's cyan (legal-move highlights and heatmap tiles can render on
// the same cell simultaneously) so the two never read as one signal. Proven
// wins get their own color outright (matches the engine's own MCTS-Solver
// priority for `suggested_move`, not just "happens to be highly visited").
const ANALYSIS_HEAT_COLOR = 0xffa94d;
const ANALYSIS_PROVEN_COLOR = 0x4caf7a;
const SUGGESTED_RING_COLOR = "#ffe066";

// Marks which border each player connects across (Black: top <-> bottom,
// White: left <-> right) -- shown as a mitered frame along the board edges
// and a matching frame on the minimap.
const EDGE_COLOR_BLACK = "#000000";
const EDGE_COLOR_WHITE = "#ffffff";

const WINNER_GLOW_BLACK = "#8f9bff";
const WINNER_GLOW_WHITE = "#ffd98a";

const PLAY_AREA_COLOR = 0x9a9da6;

// A drag that moves the pointer more than this many CSS pixels between
// mousedown and mouseup is an OrbitControls pan/rotate, not a click -- see
// `onPointerDown`/`onClick`.
const DRAG_CLICK_THRESHOLD = 6;

const BORDER_WORLD = 0.012;
const BORDER_COLOR = "#5a5c66";
const TEX_DENSITY = 64;

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
    // THREE.Sprite instances share a single module-level plane geometry
    // (there is no per-instance geometry) -- disposing it here would break
    // every label sprite created afterwards, including on the next rebuild.
    if ("geometry" in child && !(child as THREE.Sprite).isSprite) {
      (child as THREE.Mesh).geometry?.dispose();
    }
    if ("material" in child) disposeMaterial((child as THREE.Mesh).material);
  }
}

function makeLabelSprite(text: string): THREE.Sprite {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.font = "bold 88px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 6;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
  ctx.strokeText(text, size / 2, size / 2 + 4);
  ctx.fillStyle = "#2a2b32";
  ctx.fillText(text, size / 2, size / 2 + 4);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.6, 0.6, 1);
  return sprite;
}

function frameQuad(v0: [number, number], v1: [number, number], v2: [number, number], v3: [number, number], color: string): THREE.Mesh {
  const y = 0.01;
  const positions = new Float32Array([
    v0[0], y, v0[1],
    v1[0], y, v1[1],
    v2[0], y, v2[1],
    v0[0], y, v0[1],
    v2[0], y, v2[1],
    v3[0], y, v3[1],
  ]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
  return new THREE.Mesh(geo, mat);
}

function makeFaceTexture(fillColor: string, faceW: number, faceH: number): THREE.CanvasTexture {
  const w = Math.max(8, Math.round(faceW * TEX_DENSITY));
  const h = Math.max(8, Math.round(faceH * TEX_DENSITY));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = BORDER_COLOR;
  ctx.fillRect(0, 0, w, h);
  const bx = Math.min(w / 2 - 1, BORDER_WORLD * TEX_DENSITY);
  const by = Math.min(h / 2 - 1, BORDER_WORLD * TEX_DENSITY);
  ctx.fillStyle = fillColor;
  ctx.fillRect(bx, by, w - bx * 2, h - by * 2);
  return new THREE.CanvasTexture(canvas);
}

function buildBoxMaterials(colorHex: number, sizeX: number, sizeY: number, sizeZ: number): THREE.MeshStandardMaterial[] {
  const fillColor = `#${colorHex.toString(16).padStart(6, "0")}`;
  const matFor = (faceW: number, faceH: number) =>
    new THREE.MeshStandardMaterial({
      map: makeFaceTexture(fillColor, faceW, faceH),
      roughness: 0.6,
      metalness: 0.05,
      flatShading: true,
    });
  const xMat = matFor(sizeZ, sizeY);
  const yMat = matFor(sizeX, sizeZ);
  const zMat = matFor(sizeX, sizeY);
  return [xMat, xMat, yMat, yMat, zMat, zMat];
}

// --- Minimap (top-down, 2D canvas) ---

function shadeForHeight(piece: Player, height: number): string {
  const t = Math.min(1, height / 12);
  const base = piece === "Black" ? [58, 61, 70] : [242, 233, 216];
  const lit = piece === "Black" ? [112, 120, 142] : [255, 253, 246];
  const c = base.map((v, i) => Math.round(v + ((lit[i] ?? 0) - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function playerAccent(player: Player): string {
  return player === "Black" ? "#9aa2b8" : "#f2e9d8";
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** BFS for a border-to-border path of `color`'s cells (Black: top row to
 * bottom row; White: left column to right column). Returns cell indices
 * along one winning route, or `null`. Computed client-side (not returned by
 * the server) purely for the minimap's cosmetic glow. */
function findWinningPath(board: GameView["board"], w: number, h: number, color: Player): number[] | null {
  const idx = (x: number, y: number) => y * w + x;
  const owned = (i: number) => board[i]?.piece === color;
  const starts: number[] = [];
  const goal = new Set<number>();
  if (color === "Black") {
    for (let x = 0; x < w; x++) starts.push(idx(x, 0));
    for (let x = 0; x < w; x++) goal.add(idx(x, h - 1));
  } else {
    for (let y = 0; y < h; y++) starts.push(idx(0, y));
    for (let y = 0; y < h; y++) goal.add(idx(w - 1, y));
  }

  const prev = new Map<number, number>();
  const queue = starts.filter(owned);
  queue.forEach((i) => prev.set(i, -1));

  let reached = -1;
  for (let head = 0; head < queue.length && reached < 0; head++) {
    const cur = queue[head]!;
    if (goal.has(cur)) {
      reached = cur;
      break;
    }
    const cx = cur % w;
    const cy = Math.floor(cur / w);
    const neighbors: [number, number][] = [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]];
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ni = idx(nx, ny);
      if (!owned(ni) || prev.has(ni)) continue;
      prev.set(ni, cur);
      queue.push(ni);
    }
  }
  if (reached < 0) return null;

  const path: number[] = [];
  for (let i = reached; i >= 0; i = prev.get(i) ?? -1) path.push(i);
  return path.reverse();
}

export const DruidRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let minimapRef: HTMLCanvasElement | undefined;
  let dotRef: HTMLSpanElement | undefined;

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
  let pickables: THREE.Mesh[] = [];
  const mouse = new THREE.Vector2();
  let animationHandle = 0;
  let builtSize: Size | null = null;
  let minimapDpr = 1;

  function buildGoalEdges(size: Size): void {
    const { w, h } = size;
    const t = 0.14;
    const x0 = -0.5, x1 = w - 0.5;
    const z0 = -0.5, z1 = h - 0.5;
    boardGroup.add(frameQuad([x0, z0], [x1, z0], [x1 + t, z0 - t], [x0 - t, z0 - t], EDGE_COLOR_BLACK));
    boardGroup.add(frameQuad([x0, z1], [x1, z1], [x1 + t, z1 + t], [x0 - t, z1 + t], EDGE_COLOR_BLACK));
    boardGroup.add(frameQuad([x0, z0], [x0, z1], [x0 - t, z1 + t], [x0 - t, z0 - t], EDGE_COLOR_WHITE));
    boardGroup.add(frameQuad([x1, z0], [x1, z1], [x1 + t, z1 + t], [x1 + t, z0 - t], EDGE_COLOR_WHITE));
  }

  function buildLabels(size: Size): void {
    const { w, h } = size;
    const margin = 0.85;
    for (let i = 0; i < w; i++) {
      const letter = String.fromCharCode(65 + i);
      [-0.5 - margin, h - 0.5 + margin].forEach((z) => {
        const sprite = makeLabelSprite(letter);
        sprite.position.set(i, 0.02, z);
        boardGroup.add(sprite);
      });
    }
    for (let j = 0; j < h; j++) {
      const number = String(j + 1);
      [-0.5 - margin, w - 0.5 + margin].forEach((x) => {
        const sprite = makeLabelSprite(number);
        sprite.position.set(x, 0.02, j);
        boardGroup.add(sprite);
      });
    }
  }

  function buildBoard(size: Size): void {
    clearGroup(boardGroup);
    const { w, h } = size;

    const base = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshStandardMaterial({ color: PLAY_AREA_COLOR, roughness: 1 }),
    );
    base.rotation.x = -Math.PI / 2;
    base.position.set((w - 1) / 2, -0.02, (h - 1) / 2);
    boardGroup.add(base);

    const points: THREE.Vector3[] = [];
    for (let i = 0; i <= w; i++) {
      points.push(new THREE.Vector3(i - 0.5, 0, -0.5), new THREE.Vector3(i - 0.5, 0, h - 0.5));
    }
    for (let j = 0; j <= h; j++) {
      points.push(new THREE.Vector3(-0.5, 0, j - 0.5), new THREE.Vector3(w - 0.5, 0, j - 0.5));
    }
    const gridGeo = new THREE.BufferGeometry().setFromPoints(points);
    const gridMat = new THREE.LineBasicMaterial({ color: 0x4b4d55, transparent: true, opacity: 0.55 });
    boardGroup.add(new THREE.LineSegments(gridGeo, gridMat));

    buildGoalEdges(size);
    buildLabels(size);

    const center = new THREE.Vector3((w - 1) / 2, 0, (h - 1) / 2);
    controls.target.copy(center);
    camera.position.set(center.x - w * 0.3, Math.max(w, h) * 1.3, center.z + h * 0.9);
    camera.lookAt(center);
  }

  function buildPieces(size: Size, layers: LayerEntry[][], beams: Beam[]): void {
    clearGroup(piecesGroup);
    const { w } = size;
    const cubeGeo = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
    const unitMats: Record<Player, THREE.MeshStandardMaterial[]> = {
      Black: buildBoxMaterials(BLACK_COLOR, CUBE, CUBE, CUBE),
      White: buildBoxMaterials(WHITE_COLOR, CUBE, CUBE, CUBE),
    };

    layers.forEach((col, idx) => {
      const x = idx % w;
      const z = Math.floor(idx / w);
      col.forEach((entry, level) => {
        if (!entry || typeof entry === "object") return; // gap or beam-claimed
        const cube = new THREE.Mesh(cubeGeo, unitMats[entry]);
        cube.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
        piecesGroup.add(cube);
      });
    });

    beams.forEach((beam) => {
      const [, c1] = beam.cells;
      const x = c1 % w;
      const z = Math.floor(c1 / w);
      const colorHex = beam.color === "Black" ? BLACK_COLOR : WHITE_COLOR;
      const sizeX = beam.orientation === "Horizontal" ? 2 + CUBE : CUBE;
      const sizeZ = beam.orientation === "Vertical" ? 2 + CUBE : CUBE;
      const geo = new THREE.BoxGeometry(sizeX, CUBE, sizeZ);
      const box = new THREE.Mesh(geo, buildBoxMaterials(colorHex, sizeX, CUBE, sizeZ));
      box.position.set(x, beam.level * LEVEL_H + LEVEL_H / 2, z);
      piecesGroup.add(box);
    });
  }

  function rebuildHighlights(): void {
    clearGroup(highlightGroup);
    pickables = [];
    if (props.legalMoves.length === 0 || props.busy) return;

    const { w } = props.state.size;
    const geo = new THREE.PlaneGeometry(0.86, 0.86);
    const mat = new THREE.MeshBasicMaterial({
      color: MOVE_HILITE,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    props.legalMoves.forEach((mv) => {
      const footprint = footprintFor(mv, w);
      footprint.forEach((cellIdx) => {
        const square = props.state.board[cellIdx];
        if (!square) return;
        const x = cellIdx % w;
        const z = Math.floor(cellIdx / w);
        const plane = new THREE.Mesh(geo, mat.clone());
        plane.rotation.x = -Math.PI / 2;
        plane.position.set(x, square.height * LEVEL_H + 0.03, z);
        plane.userData.move = mv;
        highlightGroup.add(plane);
        pickables.push(plane);
      });
    });
  }

  function buildGhost(move: Move | null): void {
    clearGroup(ghostGroup);
    if (!move || props.busy) return;
    const { w } = props.state.size;
    const color = props.state.player === "Black" ? BLACK_COLOR : WHITE_COLOR;
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.6,
      metalness: 0.05,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    });
    const [piece, moveIndex] = move;
    const level = props.state.board[moveIndex]?.height ?? 0;

    if (piece === "Sarsen") {
      const x = moveIndex % w;
      const z = Math.floor(moveIndex / w);
      const cube = new THREE.Mesh(new THREE.BoxGeometry(CUBE, CUBE, CUBE), mat);
      cube.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
      ghostGroup.add(cube);
      return;
    }

    const orientation = piece.Lintel;
    const cells = footprintFor(move, w);
    const mid = cells[1]!;
    const x = mid % w;
    const z = Math.floor(mid / w);
    const sizeX = orientation === "Horizontal" ? 2 + CUBE : CUBE;
    const sizeZ = orientation === "Vertical" ? 2 + CUBE : CUBE;
    const box = new THREE.Mesh(new THREE.BoxGeometry(sizeX, CUBE, sizeZ), mat);
    box.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
    ghostGroup.add(box);
  }

  /** A flat square outline (a `LineLoop`, not a wireframe plane -- a
   * wireframe `PlaneGeometry` draws its diagonal too, which reads as a
   * crossed-out cell rather than a ring) marking the suggested move's
   * footprint, layered on top of its heat tile. */
  function makeSquareOutline(half: number, color: string): THREE.LineLoop {
    const points = [
      new THREE.Vector3(-half, 0, -half),
      new THREE.Vector3(half, 0, -half),
      new THREE.Vector3(half, 0, half),
      new THREE.Vector3(-half, 0, half),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({ color });
    return new THREE.LineLoop(geo, mat);
  }

  /** Renders `props.analysisOverlay` as translucent tiles over
   * each candidate's footprint, intensity scaled by `visitShare` relative to
   * the strongest candidate (not to 1.0 -- a position with no single
   * dominant move would otherwise render every tile faint even when the
   * analysis is confident in aggregate), plus a gold outline on the
   * suggested move's cells. Reuses `rebuildHighlights`'s shared-geometry/
   * cloned-material pattern (one base geometry, a fresh material clone per
   * tile so per-candidate opacity can differ) and app.js's ghost-preview
   * visual language (translucent colored planes, not a new chrome style). */
  function rebuildAnalysisOverlay(): void {
    clearGroup(analysisGroup);
    const overlay = props.analysisOverlay;
    if (!overlay || overlay.length === 0) return;

    const { w } = props.state.size;
    const maxShare = overlay.reduce((m, e) => Math.max(m, e.visitShare), 0);
    const tileGeo = new THREE.PlaneGeometry(0.78, 0.78);

    overlay.forEach((entry) => {
      const intensity = maxShare > 0 ? entry.visitShare / maxShare : 0;
      const color = entry.isProven ? ANALYSIS_PROVEN_COLOR : ANALYSIS_HEAT_COLOR;
      const opacity = 0.12 + intensity * 0.55;

      footprintFor(entry.move, w).forEach((cellIdx) => {
        const square = props.state.board[cellIdx];
        if (!square) return;
        const x = cellIdx % w;
        const z = Math.floor(cellIdx / w);
        const y = square.height * LEVEL_H + 0.04;

        const tile = new THREE.Mesh(
          tileGeo,
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
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
    });
  }

  function pickMoveAt(clientX: number, clientY: number): Move | null {
    if (!canvasRef || pickables.length === 0) return null;
    const rect = canvasRef.getBoundingClientRect();
    mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    return hits.length > 0 ? (hits[0]!.object.userData.move as Move) : null;
  }

  let pointerDownAt: { x: number; y: number } | null = null;

  function onPointerDown(event: MouseEvent): void {
    pointerDownAt = { x: event.clientX, y: event.clientY };
  }

  function onClick(event: MouseEvent): void {
    if (props.busy) return;
    // A native `click` fires on mouseup regardless of how far the pointer
    // travelled since mousedown -- so an OrbitControls rotate/pan that
    // happens to start and end over a legal-move cell would otherwise place
    // a piece there. Only treat it as a placement if the pointer barely
    // moved, i.e. this was actually a click and not a drag.
    const dx = pointerDownAt ? event.clientX - pointerDownAt.x : 0;
    const dy = pointerDownAt ? event.clientY - pointerDownAt.y : 0;
    if (Math.hypot(dx, dy) > DRAG_CLICK_THRESHOLD) return;
    const move = pickMoveAt(event.clientX, event.clientY);
    if (move) props.onMove(move);
  }

  function onPointerMove(event: MouseEvent): void {
    if (props.busy) {
      props.onHover(null);
      return;
    }
    props.onHover(pickMoveAt(event.clientX, event.clientY));
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

  function setupMinimap(): void {
    if (!minimapRef) return;
    const css = 236;
    minimapDpr = window.devicePixelRatio || 1;
    minimapRef.style.width = `${css}px`;
    minimapRef.style.height = `${css}px`;
    minimapRef.width = Math.round(css * minimapDpr);
    minimapRef.height = Math.round(css * minimapDpr);
  }

  function updateMinimap(view: GameView): void {
    if (!minimapRef) return;
    const ctx = minimapRef.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(minimapDpr, 0, 0, minimapDpr, 0, 0);
    const cssW = minimapRef.width / minimapDpr;
    const cssH = minimapRef.height / minimapDpr;
    ctx.clearRect(0, 0, cssW, cssH);

    const { w, h } = view.size;
    const pad = 14;
    const gap = 3;
    const cell = Math.max(3, Math.min((cssW - pad * 2 - (w - 1) * gap) / w, (cssH - pad * 2 - (h - 1) * gap) / h));
    const gridW = cell * w + gap * (w - 1);
    const gridH = cell * h + gap * (h - 1);
    const ox = (cssW - gridW) / 2;
    const oy = (cssH - gridH) / 2;

    const frameT = 5;
    const ringOffset = frameT + 4;
    const backingPad = ringOffset + 3;

    roundRect(ctx, ox - backingPad, oy - backingPad, gridW + backingPad * 2, gridH + backingPad * 2, 10);
    ctx.fillStyle = "#9a9da6";
    ctx.fill();

    ctx.lineCap = "round";
    for (let i = 0; i < view.board.length; i++) {
      const sq = view.board[i]!;
      if (!sq.piece) continue;
      const x = i % w;
      const y = Math.floor(i / w);
      const cx = ox + x * (cell + gap) + cell / 2;
      const cy = oy + y * (cell + gap) + cell / 2;
      ctx.strokeStyle = shadeForHeight(sq.piece, sq.height);
      ctx.lineWidth = cell * 0.85;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      if (x + 1 < w && view.board[i + 1]?.piece === sq.piece) {
        ctx.lineTo(ox + (x + 1) * (cell + gap) + cell / 2, cy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
      }
      if (y + 1 < h && view.board[i + w]?.piece === sq.piece) {
        ctx.lineTo(cx, oy + (y + 1) * (cell + gap) + cell / 2);
        ctx.stroke();
      }
    }

    for (let i = 0; i < view.board.length; i++) {
      const sq = view.board[i]!;
      const x = i % w;
      const y = Math.floor(i / w);
      const px = ox + x * (cell + gap);
      const py = oy + y * (cell + gap);
      roundRect(ctx, px, py, cell, cell, Math.max(2, cell * 0.22));
      ctx.fillStyle = sq.piece ? shadeForHeight(sq.piece, sq.height) : "#9a9da6";
      ctx.fill();
      if (!sq.piece) {
        ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    const frameQuad2d = (points: [number, number][], color: string) => {
      ctx.beginPath();
      points.forEach(([x, y], k) => (k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    };
    const fx0 = ox, fx1 = ox + gridW, fy0 = oy, fy1 = oy + gridH;
    frameQuad2d([[fx0, fy0], [fx1, fy0], [fx1 + frameT, fy0 - frameT], [fx0 - frameT, fy0 - frameT]], EDGE_COLOR_BLACK);
    frameQuad2d([[fx0, fy1], [fx1, fy1], [fx1 + frameT, fy1 + frameT], [fx0 - frameT, fy1 + frameT]], EDGE_COLOR_BLACK);
    frameQuad2d([[fx0, fy0], [fx0, fy1], [fx0 - frameT, fy1 + frameT], [fx0 - frameT, fy0 - frameT]], EDGE_COLOR_WHITE);
    frameQuad2d([[fx1, fy0], [fx1, fy1], [fx1 + frameT, fy1 + frameT], [fx1 + frameT, fy0 - frameT]], EDGE_COLOR_WHITE);

    if (view.terminal && view.winner) {
      const path = findWinningPath(view.board, w, h, view.winner);
      if (path) {
        const glowColor = view.winner === "Black" ? WINNER_GLOW_BLACK : WINNER_GLOW_WHITE;
        ctx.save();
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 7;
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = Math.max(2, cell * 0.18);
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        path.forEach((i, k) => {
          const x = ox + (i % w) * (cell + gap) + cell / 2;
          const y = oy + Math.floor(i / w) * (cell + gap) + cell / 2;
          if (k === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.restore();
      }
    }

    const ringColor = view.terminal ? (view.winner ? playerAccent(view.winner) : "#6b6e78") : playerAccent(view.player);
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 2;
    roundRect(ctx, ox - ringOffset, oy - ringOffset, gridW + ringOffset * 2, gridH + ringOffset * 2, 8);
    ctx.stroke();

    if (dotRef) {
      dotRef.style.background = view.terminal
        ? view.winner
          ? view.winner === "Black" ? "#3a3d46" : "#f2e9d8"
          : "#6b6e78"
        : view.player === "Black" ? "#3a3d46" : "#f2e9d8";
    }
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
    controls.minDistance = 3;
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
    scene.add(boardGroup, piecesGroup, highlightGroup, ghostGroup, analysisGroup);

    raycaster = new THREE.Raycaster();

    canvasRef.addEventListener("mousedown", onPointerDown);
    canvasRef.addEventListener("click", onClick);
    canvasRef.addEventListener("mousemove", onPointerMove);
    canvasRef.addEventListener("mouseleave", onPointerLeave);
    window.addEventListener("resize", onResize);

    setupMinimap();
    animate();
  });

  onCleanup(() => {
    cancelAnimationFrame(animationHandle);
    window.removeEventListener("resize", onResize);
    canvasRef?.removeEventListener("mousedown", onPointerDown);
    canvasRef?.removeEventListener("click", onClick);
    canvasRef?.removeEventListener("mousemove", onPointerMove);
    canvasRef?.removeEventListener("mouseleave", onPointerLeave);
    clearGroup(boardGroup);
    clearGroup(piecesGroup);
    clearGroup(highlightGroup);
    clearGroup(ghostGroup);
    clearGroup(analysisGroup);
    renderer?.dispose();
  });

  // Board geometry only needs rebuilding when the size actually changes
  // (i.e. a new game) -- not on every move.
  createEffect(() => {
    const size = props.state.size;
    if (!boardGroup) return;
    if (builtSize && builtSize.w === size.w && builtSize.h === size.h) return;
    buildBoard(size);
    builtSize = size;
  });

  createEffect(() => {
    const size = props.state.size;
    const history = props.history;
    if (!piecesGroup) return;
    const { layers, beams } = buildStackModel(size, history);
    buildPieces(size, layers, beams);
    rebuildHighlights();
  });

  createEffect(() => {
    // Track legalMoves/busy explicitly (Solid only tracks what's actually
    // read during the effect body).
    void props.legalMoves;
    void props.busy;
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
    if (!minimapRef) return;
    updateMinimap(props.view);
  });

  return (
    <>
      <canvas ref={canvasRef} class="druid-board" />
      <div class="druid-minimap">
        <div class="druid-minimap-title">
          <span ref={dotRef} class="druid-minimap-dot" />
          Board
        </div>
        <canvas ref={minimapRef} class="druid-minimap-canvas" />
      </div>
    </>
  );
};
