// AkronRenderer.tsx — Akron's three.js board, built on `@mcts/pyramid`'s
// shared pyramid board/marble rendering (the same physical board and marble
// look `@mcts/margo`'s `MargoRenderer.tsx` uses) plus what Akron needs that
// Margo doesn't: a two-step click-to-move interaction (select one of the
// mover's own pieces, then a destination) alongside the existing
// click-to-add interaction, and an animated relocate for `Action::Move`
// (including its cascade drop, when the vacated cell had a single
// dependent).
//
// The wire format (`GameView`) carries no connectivity/cut information, so
// unlike the plan's "if feasible" over/under visualization, this renderer
// draws no cut/uncut distinction -- there is nothing in `view` to draw it
// from without a server-side change, which is out of this phase's scope.
//
// Cascade animation pairing: `view.cells` only carries final occupancy, not
// piece identity, so which *physical* piece ends up where isn't directly
// observable. But a legal `Move` only ever cascades a single linear chain
// (`pyramid::Pyramid::relocate`'s "pinned" rejection rules out branching),
// so comparing `history`'s last step's `before` state against the new
// `view` finds at most one cell that was occupied and is now empty besides
// the source (`findCascadeTop`) -- the top of that chain. Animating a
// second piece falling from there straight to the source cell looks
// identical to animating every intermediate hop individually, since the
// chain's interior cells never change occupant-color across the move.

import { type Component, createEffect, onCleanup, onMount } from "solid-js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GameRendererProps } from "@mcts/game";
import {
  addStandardLighting,
  buildMarbleMaterial,
  buildPyramidBoard,
  clearGroup,
  DRAG_CLICK_THRESHOLD,
  frameBoard,
  PieRuleSwap,
  positionFor,
  RADIUS,
} from "@mcts/pyramid";
import { isAdd, isMove, type Action, type GameState, type GameView } from "./types.js";

const BLACK_COLOR = 0x2c2e35;
const WHITE_COLOR = 0xf4ecdd;
// Board-edge rim colors marking each player's two target sides (see
// `buildSideMarkers`) -- matching `summary.ts`'s `BLACK_SWATCH`/
// `WHITE_SWATCH` HUD dots, not `BLACK_COLOR`/`WHITE_COLOR` (the marble
// materials): a flat `MeshBasicMaterial` rim in the exact marble color
// would be easy to misread as a row of pieces sitting off the board, and a
// lighter, HUD-matching slate reads clearly against the dark scene
// background even unlit.
const BLACK_SIDE_COLOR = 0x3a3d46;
const WHITE_SIDE_COLOR = 0xf2e9d8;
const ADD_HILITE = 0x52c2ee;
const MOVABLE_HILITE = 0x3ddc97;
const SELECTED_HILITE = 0xffe066;
const ANALYSIS_HEAT_COLOR = 0xffa94d;
const ANALYSIS_PROVEN_COLOR = 0x4caf7a;
const SUGGESTED_RING_COLOR = "#ffe066";
const GHOST_OPACITY = 0.45;
const MOVE_ANIM_MS = 280;

type Pick =
  { kind: "add"; move: Action } | { kind: "dest"; move: Action } | { kind: "select"; src: number };

function targetIndex(move: Action): number | null {
  if (move === "Swap") return null;
  if (isAdd(move)) return move.Add[0];
  return move.Move[1];
}

/** The one cell (besides `src`) that was occupied in `before` and is empty
 * in `view` -- the top of a cascade chain, if `Action::Move` triggered one.
 * `null` when the move was a plain relocation with no cascade. */
function findCascadeTop(before: GameState, view: GameView, src: number): number | null {
  const after = new Set<number>();
  view.cells.forEach((cell, i) => {
    if (cell) after.add(i);
  });
  for (const idx of before.occupied) {
    if (idx === src) continue;
    if (!after.has(idx)) return idx;
  }
  return null;
}

interface FallAnim {
  mesh: THREE.Mesh;
  from: THREE.Vector3;
  to: THREE.Vector3;
  start: number;
}

export const AkronRenderer: Component<GameRendererProps<GameState, Action, GameView>> = (props) => {
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
  let animGroup: THREE.Group;
  let sideMarkersGroup: THREE.Group;
  let pickables: THREE.Mesh[] = [];
  const mouse = new THREE.Vector2();
  let animationHandle = 0;
  let builtN: number | null = null;
  let lastHistoryLen = 0;
  let selectedSource: number | null = null;
  let activeAnims: FallAnim[] = [];

  const sphereGeo = new THREE.SphereGeometry(RADIUS, 28, 20);
  const ringGeo = new THREE.RingGeometry(RADIUS * 0.55, RADIUS * 0.88, 32);
  const outlineGeo = new THREE.RingGeometry(RADIUS * 0.92, RADIUS * 1.02, 32);
  // Cursor hit-testing uses this full disc, not `ringGeo`/`outlineGeo` --
  // those are annuli with a hole in the middle, and raycasting straight at
  // pickables means most of a legal cell's visual footprint (everything
  // inside the ring's wall) would never register a hit. `pickGeo` is
  // invisible; the rings stay purely cosmetic.
  const pickGeo = new THREE.CircleGeometry(0.48, 24);
  const pickMat = new THREE.MeshBasicMaterial({ visible: false });

  function buildBoard(n: number): void {
    const center = buildPyramidBoard(boardGroup, n);
    frameBoard(camera, controls, center, n);
    buildSideMarkers(n);
  }

  /** A flat quad in the XZ plane at height `y`, wound for a from-above
   * camera but rendered `DoubleSide` regardless (see the ring highlights
   * elsewhere in this file, which do the same) -- corners in order around
   * the quad's perimeter, not diagonal pairs. */
  function flatQuad(
    p1: [number, number],
    p2: [number, number],
    p3: [number, number],
    p4: [number, number],
    y: number,
  ): THREE.BufferGeometry {
    const v = (p: [number, number]): [number, number, number] => [p[0], y, p[1]];
    const positions = new Float32Array([...v(p1), ...v(p2), ...v(p3), ...v(p1), ...v(p3), ...v(p4)]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }

  /** Colored trapezoid markers along the board's four edges, marking which
   * two (opposite) sides each player is trying to connect -- per
   * `game_akron::State::has_span`'s fixed assignment, Black spans row 0 to
   * row `n-1` (the near/far edges here) and White spans column 0 to column
   * `n-1` (the left/right edges), a board-level fact the `winner`/legal-move
   * data alone gives no visual cue for.
   *
   * The board's base is a circular slab (`buildPyramidBoard`), not a
   * square, so a plain rectangular bar along a socket row/column overhangs
   * the round edge at the corners. Each marker is instead a trapezoid whose
   * inner edge runs along its own side's pair of corner sockets and whose
   * outer edge is pulled radially out from the board's center to near the
   * slab's rim -- since two adjacent sides share a corner socket, they also
   * share that corner's radial projection, so adjacent trapezoids meet
   * exactly along a seam with no gap or overlap, and nothing crosses the
   * circular rim. */
  function buildSideMarkers(n: number): void {
    clearGroup(sideMarkersGroup);
    const y = -RADIUS + 0.02;
    const center: [number, number] = [(n - 1) / 2, (n - 1) / 2];
    const outerRadius = n * 0.74;
    const innerPad = 0.55;

    const project = (corner: [number, number], radius: number): [number, number] => {
      const dx = corner[0] - center[0];
      const dz = corner[1] - center[1];
      const dist = Math.hypot(dx, dz) || 1;
      return [center[0] + (dx / dist) * radius, center[1] + (dz / dist) * radius];
    };
    const innerOf = (corner: [number, number]): [number, number] =>
      project(corner, Math.hypot(corner[0] - center[0], corner[1] - center[1]) + innerPad);
    const outerOf = (corner: [number, number]): [number, number] => project(corner, outerRadius);

    const nearA: [number, number] = [0, 0];
    const nearB: [number, number] = [n - 1, 0];
    const farA: [number, number] = [0, n - 1];
    const farB: [number, number] = [n - 1, n - 1];

    function addTrapezoid(a: [number, number], b: [number, number], color: number): void {
      const geo = flatQuad(innerOf(a), innerOf(b), outerOf(b), outerOf(a), y);
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
      sideMarkersGroup.add(mesh);
    }

    addTrapezoid(nearA, nearB, BLACK_SIDE_COLOR);
    addTrapezoid(farA, farB, BLACK_SIDE_COLOR);
    addTrapezoid(nearA, farA, WHITE_SIDE_COLOR);
    addTrapezoid(nearB, farB, WHITE_SIDE_COLOR);
  }

  function buildPieces(view: GameView, skip?: ReadonlySet<number>): void {
    clearGroup(piecesGroup);
    view.cells.forEach((cell, index) => {
      if (!cell || skip?.has(index)) return;
      const [x, y, z] = positionFor(view.n, index);
      const mat = buildMarbleMaterial(cell.piece === "Black" ? BLACK_COLOR : WHITE_COLOR);
      const sphere = new THREE.Mesh(sphereGeo, mat);
      sphere.position.set(x, y, z);
      piecesGroup.add(sphere);
    });
  }

  function cancelAnimations(): void {
    clearGroup(animGroup);
    activeAnims = [];
  }

  function animateMove(
    before: GameState,
    move: { Move: [number, number, number] },
    view: GameView,
  ): void {
    cancelAnimations();
    const [src, dst, n] = move.Move;
    const cascadeTop = findCascadeTop(before, view, src);

    const skip = new Set<number>([dst]);
    if (cascadeTop !== null) skip.add(src);
    buildPieces(view, skip);

    const now = performance.now();
    const moverColor = before.turn === "Black" ? BLACK_COLOR : WHITE_COLOR;
    const mover = new THREE.Mesh(sphereGeo, buildMarbleMaterial(moverColor));
    animGroup.add(mover);
    activeAnims.push({
      mesh: mover,
      from: new THREE.Vector3(...positionFor(n, src)),
      to: new THREE.Vector3(...positionFor(n, dst)),
      start: now,
    });

    if (cascadeTop !== null) {
      const fallerColor = before.black.includes(cascadeTop) ? BLACK_COLOR : WHITE_COLOR;
      const faller = new THREE.Mesh(sphereGeo, buildMarbleMaterial(fallerColor));
      animGroup.add(faller);
      activeAnims.push({
        mesh: faller,
        from: new THREE.Vector3(...positionFor(n, cascadeTop)),
        to: new THREE.Vector3(...positionFor(n, src)),
        start: now,
      });
    }
  }

  function stepAnimations(now: number): void {
    if (activeAnims.length === 0) return;
    let allDone = true;
    for (const anim of activeAnims) {
      const t = Math.min(1, (now - anim.start) / MOVE_ANIM_MS);
      if (t < 1) allDone = false;
      const eased = 1 - (1 - t) * (1 - t);
      anim.mesh.position.lerpVectors(anim.from, anim.to, eased);
    }
    if (allDone) {
      cancelAnimations();
      buildPieces(props.view);
    }
  }

  function addPickRing(
    geo: THREE.BufferGeometry,
    index: number,
    n: number,
    color: number,
    pick: Pick,
  ): void {
    const [x, y, z] = positionFor(n, index);
    const ring = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(x, y - RADIUS + 0.02, z);
    highlightGroup.add(ring);

    const pickMesh = new THREE.Mesh(pickGeo, pickMat);
    pickMesh.rotation.x = -Math.PI / 2;
    pickMesh.position.set(x, y - RADIUS + 0.02, z);
    pickMesh.userData.pick = pick satisfies Pick;
    highlightGroup.add(pickMesh);
    pickables.push(pickMesh);
  }

  function rebuildHighlights(): void {
    clearGroup(highlightGroup);
    pickables = [];
    if (props.busy) return;

    const n = props.state.n;
    const addMoves = props.legalMoves.filter(isAdd);
    const moveMoves = props.legalMoves.filter(isMove);
    const movesBySrc = new Map<number, { Move: [number, number, number] }[]>();
    for (const m of moveMoves) {
      const src = m.Move[0];
      const list = movesBySrc.get(src);
      if (list) list.push(m);
      else movesBySrc.set(src, [m]);
    }

    if (selectedSource !== null && movesBySrc.has(selectedSource)) {
      addPickRing(outlineGeo, selectedSource, n, SELECTED_HILITE, {
        kind: "select",
        src: selectedSource,
      });
      for (const m of movesBySrc.get(selectedSource)!) {
        addPickRing(ringGeo, m.Move[1], n, ADD_HILITE, { kind: "dest", move: m });
      }
      return;
    }

    selectedSource = null;
    for (const m of addMoves) {
      addPickRing(ringGeo, m.Add[0], n, ADD_HILITE, { kind: "add", move: m });
    }
    for (const src of movesBySrc.keys()) {
      addPickRing(outlineGeo, src, n, MOVABLE_HILITE, { kind: "select", src });
    }
  }

  function buildGhost(move: Action | null): void {
    clearGroup(ghostGroup);
    if (!move || props.busy) return;
    const index = targetIndex(move);
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
      const index = targetIndex(entry.move);
      if (index === null) return;
      const intensity = maxShare > 0 ? entry.visitShare / maxShare : 0;
      const color = entry.isProven ? ANALYSIS_PROVEN_COLOR : ANALYSIS_HEAT_COLOR;
      const opacity = 0.15 + intensity * 0.55;

      const [x, y, z] = positionFor(n, index);
      const tile = new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
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

  function pickAt(clientX: number, clientY: number): Pick | null {
    if (!canvasRef || pickables.length === 0) return null;
    const rect = canvasRef.getBoundingClientRect();
    mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    return hits.length > 0 ? (hits[0]!.object.userData.pick as Pick) : null;
  }

  let pointerDownAt: { x: number; y: number } | null = null;

  function onPointerDown(event: MouseEvent): void {
    pointerDownAt = { x: event.clientX, y: event.clientY };
  }

  function onClick(event: MouseEvent): void {
    if (props.busy) return;
    const dx = pointerDownAt ? event.clientX - pointerDownAt.x : 0;
    const dy = pointerDownAt ? event.clientY - pointerDownAt.y : 0;
    if (Math.hypot(dx, dy) > DRAG_CLICK_THRESHOLD) return;

    const hit = pickAt(event.clientX, event.clientY);
    if (!hit) {
      if (selectedSource !== null) {
        selectedSource = null;
        rebuildHighlights();
      }
      return;
    }
    if (hit.kind === "select") {
      selectedSource = selectedSource === hit.src ? null : hit.src;
      rebuildHighlights();
      return;
    }
    props.onMove(hit.move);
  }

  function onPointerMove(event: MouseEvent): void {
    if (props.busy) {
      props.onHover(null);
      return;
    }
    const hit = pickAt(event.clientX, event.clientY);
    props.onHover(hit && hit.kind !== "select" ? hit.move : null);
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
    stepAnimations(performance.now());
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

    addStandardLighting(scene);

    boardGroup = new THREE.Group();
    piecesGroup = new THREE.Group();
    highlightGroup = new THREE.Group();
    ghostGroup = new THREE.Group();
    analysisGroup = new THREE.Group();
    animGroup = new THREE.Group();
    sideMarkersGroup = new THREE.Group();
    scene.add(
      boardGroup,
      piecesGroup,
      highlightGroup,
      ghostGroup,
      analysisGroup,
      animGroup,
      sideMarkersGroup,
    );

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
    clearGroup(animGroup);
    clearGroup(sideMarkersGroup);
    sphereGeo.dispose();
    ringGeo.dispose();
    outlineGeo.dispose();
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
    const history = props.history;
    if (!piecesGroup) return;
    selectedSource = null;

    const grew = history.length === lastHistoryLen + 1;
    const lastStep = grew ? history[history.length - 1] : undefined;
    lastHistoryLen = history.length;

    if (grew && lastStep && typeof lastStep.move === "object" && isMove(lastStep.move)) {
      animateMove(lastStep.before, lastStep.move, view);
    } else {
      cancelAnimations();
      buildPieces(view);
    }
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
      <canvas ref={canvasRef} class="pyramid-board" />
      <PieRuleSwap canSwap={canSwap()} busy={props.busy} onSwap={() => props.onMove("Swap")} />
    </>
  );
};
