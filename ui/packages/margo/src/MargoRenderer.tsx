// MargoRenderer.tsx — Margo's three.js board: marbles stacked into a
// Shibumi-family pyramid, following Druid's `DruidRenderer.tsx` structure
// (scene setup, click-to-place, ghost-preview-on-hover, analysis heatmap)
// but over spheres instead of cubes/beams, and with no piece-stacking
// replay needed: unlike Druid (whose `Square` wire shape only remembers
// each cell's *current* top owner/height, forcing `layers.ts` to replay the
// whole move history to reconstruct what's physically underneath), Margo
// has no movement variant in scope and a captured group is simply removed,
// so `props.view.cells` (one entry per flat pyramid index, `null` for
// empty) already is the complete physical picture on every render.
//
// Marbles get a `MeshPhysicalMaterial` clearcoat layer rather than Druid's
// flat-shaded, baked-texture faces: a thin, near-mirror top coat over a
// duller base coat is what actually produces a small, bright, moving
// specular highlight under the scene's directional "sun" light -- cheap
// (no environment map/reflection probe) but a real material response, not
// a hand-placed sprite that would drift out of place as the camera orbits.

import { type Component, createEffect, onCleanup, onMount } from "solid-js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GameRendererProps } from "@mcts/game";
import { LEVEL_RISE, positionFor } from "./geometry.js";
import type { Action, GameState, GameView } from "./types.js";
import "./margo.css";

// Slightly over the exact touching radius (0.5, per `positionFor`'s unit-
// diameter spacing) so resting marbles visibly press against their
// neighbors instead of leaving the sliver of daylight a geometrically
// "correct" 0.5 would render as.
const RADIUS = 0.505;
const SOCKET_RADIUS = 0.32;
const BLACK_COLOR = 0x2c2e35;
const WHITE_COLOR = 0xf4ecdd;
const MOVE_HILITE = 0x52c2ee;
const ANALYSIS_HEAT_COLOR = 0xffa94d;
const ANALYSIS_PROVEN_COLOR = 0x4caf7a;
const SUGGESTED_RING_COLOR = "#ffe066";
const GHOST_OPACITY = 0.45;
const ZOMBIE_RING_COLOR = "#c94b4b";
// A drag that moves the pointer more than this many CSS pixels between
// mousedown and mouseup is an OrbitControls pan/rotate, not a click -- see
// `onPointerDown`/`onClick`.
const DRAG_CLICK_THRESHOLD = 6;

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
    if ("geometry" in child) (child as THREE.Mesh).geometry?.dispose();
    if ("material" in child) disposeMaterial((child as THREE.Mesh).material);
  }
}

/** A flat, textured plane laid on the board surface -- as opposed to a
 * `THREE.Sprite`, which always billboards to face the camera and so would
 * appear to float above the board, re-facing the viewer as `OrbitControls`
 * orbits around it instead of staying "printed" on the board like the rest
 * of its geometry. */
function makeLabelPlane(text: string): THREE.Mesh {
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
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.6), material);
  plane.rotation.x = -Math.PI / 2;
  return plane;
}

/** A marble material with a thin, glossy clearcoat over a duller base coat
 * -- the combination that gives a small, bright specular highlight under a
 * directional light without needing an environment map. `zombie` pieces
 * (permanently excluded from connectivity, per the rules, but still on the
 * board) get a flattened, desaturated look instead -- visually "dead"
 * alongside the equatorial ring [`buildZombieRing`] adds. */
function buildMarbleMaterial(colorHex: number, zombie: boolean): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: colorHex,
    roughness: zombie ? 0.75 : 0.28,
    metalness: 0.05,
    clearcoat: zombie ? 0.15 : 0.9,
    clearcoatRoughness: 0.12,
  });
}

function buildZombieRing(): THREE.Mesh {
  const geo = new THREE.TorusGeometry(RADIUS * 0.98, 0.02, 8, 32);
  const mat = new THREE.MeshBasicMaterial({ color: ZOMBIE_RING_COLOR });
  return new THREE.Mesh(geo, mat);
}

export const MargoRenderer: Component<GameRendererProps<GameState, Action, GameView>> = (props) => {
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
  let pickables: THREE.Mesh[] = [];
  const mouse = new THREE.Vector2();
  let animationHandle = 0;
  let builtN: number | null = null;

  const sphereGeo = new THREE.SphereGeometry(RADIUS, 28, 20);
  const ringGeo = new THREE.RingGeometry(RADIUS * 0.55, RADIUS * 0.88, 32);
  // Cursor hit-testing uses this full disc, not `ringGeo` -- `ringGeo` is an
  // annulus with a hole in the middle, and raycasting straight at pickables
  // means most of a legal cell's visual footprint (everything inside the
  // ring's wall) would never register a hit. `pickGeo` is invisible; the
  // ring stays purely cosmetic.
  const pickGeo = new THREE.CircleGeometry(0.48, 24);

  function buildBoard(n: number): void {
    clearGroup(boardGroup);
    const center = new THREE.Vector3((n - 1) / 2, 0, (n - 1) / 2);

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(n * 0.78, n * 0.85, 0.3, 48),
      new THREE.MeshStandardMaterial({ color: 0x6b6350, roughness: 0.9 }),
    );
    base.position.set(center.x, -0.3 - RADIUS, center.z);
    boardGroup.add(base);

    // A Shibumi board is a slab drilled with sockets the ground-level
    // marbles sit in -- not a flat plane with a marble balanced on top of
    // it, which is what gridlines over a bare surface would suggest. A
    // real cut hole needs CSG boolean subtraction three.js doesn't have
    // built in; a recessed, darker, narrower cylinder standing in for the
    // hole opening reads the same way (marble sitting *in* something) from
    // any camera angle this board is viewed at.
    // Marbles rest at y = -RADIUS (that's where a level-0 sphere's bottom
    // touches the board), so the socket's opening has to sit at or just
    // below that same plane -- centering the cylinder there, as an earlier
    // version did, put its *top* above -RADIUS and made it read as a boss
    // poking up out of the board rather than a hole sunk into it.
    const socketHeight = 0; // 0.14;
    const socketTopY = -RADIUS - 0.002;
    const socketGeo = new THREE.CylinderGeometry(SOCKET_RADIUS, SOCKET_RADIUS * 0.9, socketHeight, 24);
    const socketMat = new THREE.MeshStandardMaterial({ color: 0x3a362c, roughness: 1 });
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const socket = new THREE.Mesh(socketGeo, socketMat);
        socket.position.set(col, socketTopY - socketHeight / 2, row);
        boardGroup.add(socket);
      }
    }

    const margin = 0.85;
    for (let i = 0; i < n; i++) {
      const letter = makeLabelPlane(String.fromCharCode(65 + i));
      letter.position.set(i, -RADIUS + 0.02, -0.5 - margin);
      boardGroup.add(letter);
      const number = makeLabelPlane(String(i + 1));
      number.position.set(-0.5 - margin, -RADIUS + 0.02, i);
      boardGroup.add(number);
    }

    controls.target.copy(center);
    const rise = n * LEVEL_RISE;
    camera.position.set(center.x - n * 0.5, rise * 1.1 + n * 0.5, center.z + n * 1.3);
    camera.lookAt(center);
  }

  function buildPieces(view: GameView): void {
    clearGroup(piecesGroup);
    view.cells.forEach((cell, index) => {
      if (!cell) return;
      const [x, y, z] = positionFor(view.n, index);
      const mat = buildMarbleMaterial(cell.piece === "Black" ? BLACK_COLOR : WHITE_COLOR, cell.zombie);
      const sphere = new THREE.Mesh(sphereGeo, mat);
      sphere.position.set(x, y, z);
      piecesGroup.add(sphere);
      if (cell.zombie) {
        const ring = buildZombieRing();
        ring.position.set(x, y, z);
        ring.rotation.x = Math.PI / 2;
        piecesGroup.add(ring);
      }
    });
  }

  function placementCells(move: Action): number | null {
    return move === "Swap" ? null : move.Place[0];
  }

  function rebuildHighlights(): void {
    clearGroup(highlightGroup);
    pickables = [];
    if (props.busy) return;

    const n = props.state.n;
    const mat = new THREE.MeshBasicMaterial({
      color: MOVE_HILITE,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    const pickMat = new THREE.MeshBasicMaterial({ visible: false });

    props.legalMoves.forEach((move) => {
      const index = placementCells(move);
      if (index === null) return;
      const [x, y, z] = positionFor(n, index);
      const ring = new THREE.Mesh(ringGeo, mat.clone());
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, y - RADIUS + 0.02, z);
      highlightGroup.add(ring);

      // The visible ring is a thin annulus (a hole in the middle), so it
      // alone would only register hover/click over its wall. Pick against
      // this separate, invisible full disc instead, covering the whole
      // playable footprint the ring merely outlines.
      const pick = new THREE.Mesh(pickGeo, pickMat);
      pick.rotation.x = -Math.PI / 2;
      pick.position.set(x - 0.14 , y - RADIUS + 0.02, z);
      pick.userData.move = move;
      highlightGroup.add(pick);
      pickables.push(pick);
    });
  }

  function buildGhost(move: Action | null): void {
    clearGroup(ghostGroup);
    if (!move || props.busy) return;
    const index = placementCells(move);
    if (index === null) return;
    const [x, y, z] = positionFor(props.state.n, index);
    const color = props.state.turn === "Black" ? BLACK_COLOR : WHITE_COLOR;
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.4,
      transparent: true,
      opacity: GHOST_OPACITY,
      depthWrite: false,
    });
    const sphere = new THREE.Mesh(sphereGeo, mat);
    sphere.position.set(x, y, z);
    ghostGroup.add(sphere);
  }

  function makeCircleOutline(radius: number, color: string): THREE.LineLoop {
    const points: THREE.Vector3[] = [];
    const segments = 32;
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color }));
  }

  function rebuildAnalysisOverlay(): void {
    clearGroup(analysisGroup);
    const overlay = props.analysisOverlay;
    if (!overlay || overlay.length === 0) return;

    const n = props.state.n;
    const maxShare = overlay.reduce((m, e) => Math.max(m, e.visitShare), 0);

    overlay.forEach((entry) => {
      const index = placementCells(entry.move);
      if (index === null) return;
      const intensity = maxShare > 0 ? entry.visitShare / maxShare : 0;
      const color = entry.isProven ? ANALYSIS_PROVEN_COLOR : ANALYSIS_HEAT_COLOR;
      const opacity = 0.15 + intensity * 0.55;

      const [x, y, z] = positionFor(n, index);
      const tile = new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
      );
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(x, y - RADIUS + 0.03, z);
      analysisGroup.add(tile);

      if (entry.isSuggested) {
        const ring = makeCircleOutline(RADIUS * 0.9, SUGGESTED_RING_COLOR);
        ring.position.set(x, y - RADIUS + 0.032, z);
        analysisGroup.add(ring);
      }
    });
  }

  function pickMoveAt(clientX: number, clientY: number): Action | null {
    if (!canvasRef || pickables.length === 0) return null;
    const rect = canvasRef.getBoundingClientRect();
    mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    return hits.length > 0 ? (hits[0]!.object.userData.move as Action) : null;
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

  onMount(() => {
    if (!canvasRef) return;
    renderer = new THREE.WebGLRenderer({ canvas: canvasRef, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1b1c22);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.49;

    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    // A small, bright "sun" is what actually produces the clearcoat
    // materials' visible specular highlight -- a broad/soft light source
    // would spread it into an unreadable dim smear across each marble.
    const sun = new THREE.DirectionalLight(0xffffff, 1.15);
    sun.position.set(6, 12, 8);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xaac4ff, 0.2);
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
    sphereGeo.dispose();
    ringGeo.dispose();
    pickGeo.dispose();
    renderer?.dispose();
  });

  // Board geometry only needs rebuilding when the board size actually
  // changes (a new game) -- not on every move.
  createEffect(() => {
    const n = props.state.n;
    if (!boardGroup) return;
    if (builtN === n) return;
    buildBoard(n);
    builtN = n;
  });

  createEffect(() => {
    const view = props.view;
    if (!piecesGroup) return;
    buildPieces(view);
    rebuildHighlights();
  });

  createEffect(() => {
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

  const canSwap = () => props.legalMoves.some((m) => m === "Swap");

  return (
    <>
      <canvas ref={canvasRef} class="margo-board" />
      {canSwap() && (
        <div class="margo-swap-panel">
          <span class="margo-swap-title">Pie rule</span>
          <button
            type="button"
            class="margo-swap-button"
            disabled={props.busy}
            onClick={() => props.onMove("Swap")}
          >
            Swap colours
          </button>
        </div>
      )}
    </>
  );
};

