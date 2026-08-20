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

const RADIUS = 0.47;
const BLACK_COLOR = 0x2c2e35;
const WHITE_COLOR = 0xf4ecdd;
const MOVE_HILITE = 0x52c2ee;
const ANALYSIS_HEAT_COLOR = 0xffa94d;
const ANALYSIS_PROVEN_COLOR = 0x4caf7a;
const SUGGESTED_RING_COLOR = "#ffe066";
const GHOST_OPACITY = 0.45;
const ZOMBIE_RING_COLOR = "#c94b4b";

function disposeMaterial(mat: THREE.Material | THREE.Material[] | undefined): void {
  if (!mat) return;
  if (Array.isArray(mat)) {
    mat.forEach(disposeMaterial);
    return;
  }
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

  function buildBoard(n: number): void {
    clearGroup(boardGroup);
    const center = new THREE.Vector3((n - 1) / 2, 0, (n - 1) / 2);

    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(n * 0.78, n * 0.85, 0.3, 48),
      new THREE.MeshStandardMaterial({ color: 0x6b6350, roughness: 0.9 }),
    );
    base.position.set(center.x, -0.3 - RADIUS, center.z);
    boardGroup.add(base);

    const points: THREE.Vector3[] = [];
    for (let i = 0; i < n; i++) {
      points.push(new THREE.Vector3(i, -RADIUS + 0.01, -0.5), new THREE.Vector3(i, -RADIUS + 0.01, n - 0.5));
      points.push(new THREE.Vector3(-0.5, -RADIUS + 0.01, i), new THREE.Vector3(n - 0.5, -RADIUS + 0.01, i));
    }
    const gridGeo = new THREE.BufferGeometry().setFromPoints(points);
    const gridMat = new THREE.LineBasicMaterial({ color: 0x8a8370, transparent: true, opacity: 0.4 });
    boardGroup.add(new THREE.LineSegments(gridGeo, gridMat));

    const margin = 0.85;
    for (let i = 0; i < n; i++) {
      const letter = String.fromCharCode(65 + i);
      const sprite = makeLabelSprite(letter);
      sprite.position.set(i, -RADIUS + 0.02, -0.5 - margin);
      boardGroup.add(sprite);
      const number = makeLabelSprite(String(i + 1));
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

    props.legalMoves.forEach((move) => {
      const index = placementCells(move);
      if (index === null) return;
      const [x, y, z] = positionFor(n, index);
      const ring = new THREE.Mesh(ringGeo, mat.clone());
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, y - RADIUS + 0.02, z);
      ring.userData.move = move;
      highlightGroup.add(ring);
      pickables.push(ring);
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

  function onClick(event: MouseEvent): void {
    if (props.busy) return;
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
    sphereGeo.dispose();
    ringGeo.dispose();
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

