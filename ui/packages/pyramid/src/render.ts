// render.ts — Shared three.js building blocks for every pyramid-family
// board renderer (Margo, Akron, ...): the physical board (base slab +
// drilled sockets + coordinate labels), marble materials, and the small
// scene-graph utilities (`clearGroup`/`disposeMaterial`) every one of them
// needs to rebuild its `THREE.Group`s on each render without leaking
// geometry/materials. Pulled out of `@mcts/margo`'s `MargoRenderer.tsx` once
// Akron needed the identical board -- see that package's doc comments for
// the reasoning a single renderer's file comments used to carry.
//
// Deliberately *not* shared: per-game pick/highlight/ghost logic, move
// interaction, and anything that depends on a specific `Action` shape --
// those differ enough between an add-only game (Margo) and an add-or-move
// game (Akron) that forcing them into this module would cost more in
// indirection than it saves in duplication.

import * as THREE from "three";
import { LEVEL_RISE } from "./geometry.js";

/** Slightly over the exact touching radius (0.5, per `positionFor`'s unit-
 * diameter spacing) so resting marbles visibly press against their
 * neighbors instead of leaving the sliver of daylight a geometrically
 * "correct" 0.5 would render as. */
export const RADIUS = 0.505;
export const SOCKET_RADIUS = 0.32;

/** A drag that moves the pointer more than this many CSS pixels between
 * mousedown and mouseup is an OrbitControls pan/rotate, not a click. */
export const DRAG_CLICK_THRESHOLD = 6;

export function disposeMaterial(mat: THREE.Material | THREE.Material[] | undefined): void {
  if (!mat) return;
  if (Array.isArray(mat)) {
    mat.forEach(disposeMaterial);
    return;
  }
  const withMap = mat as THREE.Material & { map?: THREE.Texture | null };
  withMap.map?.dispose();
  mat.dispose();
}

export function clearGroup(group: THREE.Group): void {
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
export function makeLabelPlane(text: string): THREE.Mesh {
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
 * directional light without needing an environment map. `matte` flattens
 * and desaturates the finish for a piece a game wants to read as visually
 * "dead" or otherwise de-emphasized (Margo's zombie pieces). */
export function buildMarbleMaterial(colorHex: number, matte = false): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: colorHex,
    roughness: matte ? 0.75 : 0.28,
    metalness: 0.05,
    clearcoat: matte ? 0.15 : 0.9,
    clearcoatRoughness: 0.12,
  });
}

/** Builds the physical pyramid board -- base slab plus one drilled-looking
 * socket per level-0 cell plus A.../1... coordinate labels -- into `group`
 * (cleared first). A real cut socket needs CSG boolean subtraction three.js
 * doesn't have built in; a recessed, darker, narrower cylinder standing in
 * for the hole opening reads the same way (marble sitting *in* something)
 * from any camera angle this board is viewed at. Returns the board's
 * center, for camera framing. */
export function buildPyramidBoard(group: THREE.Group, n: number): THREE.Vector3 {
  clearGroup(group);
  const center = new THREE.Vector3((n - 1) / 2, 0, (n - 1) / 2);

  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(n * 0.78, n * 0.85, 0.3, 48),
    new THREE.MeshStandardMaterial({ color: 0x6b6350, roughness: 0.9 }),
  );
  base.position.set(center.x, -0.3 - RADIUS, center.z);
  group.add(base);

  // Marbles rest at y = -RADIUS (that's where a level-0 sphere's bottom
  // touches the board), so the socket's opening has to sit at or just below
  // that same plane -- centering the cylinder there would put its *top*
  // above -RADIUS and make it read as a boss poking up out of the board
  // rather than a hole sunk into it.
  const socketHeight = 0;
  const socketTopY = -RADIUS - 0.002;
  const socketGeo = new THREE.CylinderGeometry(
    SOCKET_RADIUS,
    SOCKET_RADIUS * 0.9,
    socketHeight,
    24,
  );
  const socketMat = new THREE.MeshStandardMaterial({ color: 0x3a362c, roughness: 1 });
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const socket = new THREE.Mesh(socketGeo, socketMat);
      socket.position.set(col, socketTopY - socketHeight / 2, row);
      group.add(socket);
    }
  }

  const margin = 0.85;
  for (let i = 0; i < n; i++) {
    const letter = makeLabelPlane(String.fromCharCode(65 + i));
    letter.position.set(i, -RADIUS + 0.02, -0.5 - margin);
    group.add(letter);
    const number = makeLabelPlane(String(i + 1));
    number.position.set(-0.5 - margin, -RADIUS + 0.02, i);
    group.add(number);
  }

  return center;
}

/** Points `camera`/`controls` at a freshly-built board the same way every
 * pyramid-family renderer wants to on a new game/board-size change. */
export function frameBoard(
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3 },
  center: THREE.Vector3,
  n: number,
): void {
  controls.target.copy(center);
  const rise = n * LEVEL_RISE;
  camera.position.set(center.x - n * 0.5, rise * 1.1 + n * 0.5, center.z + n * 1.3);
  camera.lookAt(center);
}

/** Standard three-point lighting rig every pyramid renderer uses: ambient
 * fill plus a small, bright "sun" (what actually produces the clearcoat
 * materials' visible specular highlight -- a broad/soft source would spread
 * it into an unreadable dim smear) plus a dim cool backfill. */
export function addStandardLighting(scene: THREE.Scene): void {
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const sun = new THREE.DirectionalLight(0xffffff, 1.15);
  sun.position.set(6, 12, 8);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xaac4ff, 0.2);
  fill.position.set(-8, 6, -10);
  scene.add(fill);
}
