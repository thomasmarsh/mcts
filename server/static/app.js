(function () {
  const CUBE = 0.92; // visual cube size, < 1.0 so stacked blocks read as distinct layers
  const LEVEL_H = 1.0; // vertical spacing per stacked layer

  const BLACK_COLOR = 0x3a3d46;
  const WHITE_COLOR = 0xf2e9d8;
  const SARSEN_HILITE = 0xffcf5c;
  const LINTEL_HILITE = 0x63d3ff;

  let scene, camera, renderer, controls, raycaster, mouse;
  let boardGroup, piecesGroup, highlightGroup, ghostGroup;
  let pickables = [];
  let mode = "sarsen"; // "sarsen" | "lintelH" | "lintelV"
  let currentState = null;
  let currentLegalMoves = [];
  let busy = false; // true while a move/AI request is in flight
  let hoveredMove = null; // the move currently under the cursor, for the ghost preview
  let autoplayPaused = false; // user-toggled; blocks maybeTriggerAiTurn from chaining
  // Bumped by startNewGame. Now that "New Game" stays clickable mid-AI-turn,
  // an in-flight /api/move or /api/ai_move request from the *old* game can
  // resolve after a new one has started; callers stamp the epoch they
  // started with and drop their response if it's gone stale.
  let gameEpoch = 0;

  // Who controls each color this session: "human" or an AI preset id (e.g.
  // "strong"). Purely client-side -- the server has no notion of seats, it
  // just executes whatever move/preset a request asks for. Reset on "New Game".
  let seats = { Black: "human", White: "human" };
  let aiPresets = []; // [{id, label, description}], loaded from /api/ai_presets

  // Client-side reconstruction of the *physical* stack, since the server's
  // `Square` model only stores each cell's current top owner/height and
  // overwrites the middle cell of a bridging lintel to match the endpoints'
  // height -- it has no memory of what was physically built underneath.
  // `layers[cellIndex]` is an array of per-level entries, bottom to top:
  //   - an owner string ("Black"/"White") for a real placed unit cube
  //   - null for a gap (empty air under a bridging lintel)
  //   - { beam: id } for a level that's part of a merged lintel beam mesh,
  //     rendered once (see `beams`) rather than as a separate unit cube
  // This is built by replaying moves as they're applied during this page's
  // session, seeded from whatever state was current on load/new-game (which
  // is the best available guess for pre-existing stacks -- see non-goals
  // in the phase-3 charter about not modeling full piece history).
  let layers = [];
  let beams = [];
  let nextBeamId = 0;

  function initScene() {
    const canvas = document.getElementById("board");
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1b1d22);

    camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      200
    );

    controls = new THREE.OrbitControls(camera, renderer.domElement);
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
    scene.add(boardGroup, piecesGroup, highlightGroup, ghostGroup);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer.domElement.addEventListener("click", onClick);
    renderer.domElement.addEventListener("mousemove", onPointerMove);
    renderer.domElement.addEventListener("mouseleave", clearGhost);
    window.addEventListener("resize", onResize);

    animate();
  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  function clearGroup(group) {
    while (group.children.length) {
      const child = group.children.pop();
      // THREE.Sprite instances share a single module-level plane geometry
      // (there is no per-instance geometry) -- disposing it here would
      // break every label sprite created afterwards, including on the very
      // next "New Game". Only dispose geometry we know is per-instance.
      if (child.geometry && !child.isSprite) child.geometry.dispose();
      if (child.material) {
        child.material.map && child.material.map.dispose();
        child.material.dispose();
      }
    }
  }

  // Renders text to a small canvas and wraps it in a billboarded sprite --
  // no font/texture assets needed, and it stays readable from any angle
  // under OrbitControls (a flat decal would foreshorten at grazing angles).
  function makeLabelSprite(text) {
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#e8e8ec";
    ctx.font = "bold 88px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, size / 2, size / 2 + 4);
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.6, 0.6, 1);
    return sprite;
  }

  // Row/column labels sit just outside the grid on all four sides, so at
  // least one copy is legible regardless of camera azimuth.
  function buildLabels(size) {
    const { w, h } = size;
    const margin = 0.85;
    for (let i = 0; i < w; i++) {
      const letter = String.fromCharCode(65 + i);
      [-0.5 - margin, h - 0.5 + margin].forEach((z) => {
        const sprite = makeLabelSprite(letter);
        sprite.position.set(i, 0.35, z);
        boardGroup.add(sprite);
      });
    }
    for (let j = 0; j < h; j++) {
      const number = String(j + 1);
      [-0.5 - margin, w - 0.5 + margin].forEach((x) => {
        const sprite = makeLabelSprite(number);
        sprite.position.set(x, 0.35, j);
        boardGroup.add(sprite);
      });
    }
  }

  // Builds the static grid + base plane for the current board size. Only
  // needs to run once per board size (i.e. on load and after "New Game",
  // since size never changes mid-game).
  function buildBoard(size) {
    clearGroup(boardGroup);
    const { w, h } = size;

    const base = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshStandardMaterial({ color: 0x2a2c34, roughness: 1 })
    );
    base.rotation.x = -Math.PI / 2;
    base.position.set((w - 1) / 2, -0.02, (h - 1) / 2);
    boardGroup.add(base);

    const points = [];
    for (let i = 0; i <= w; i++) {
      points.push(
        new THREE.Vector3(i - 0.5, 0, -0.5),
        new THREE.Vector3(i - 0.5, 0, h - 0.5)
      );
    }
    for (let j = 0; j <= h; j++) {
      points.push(
        new THREE.Vector3(-0.5, 0, j - 0.5),
        new THREE.Vector3(w - 0.5, 0, j - 0.5)
      );
    }
    const gridGeo = new THREE.BufferGeometry().setFromPoints(points);
    const gridMat = new THREE.LineBasicMaterial({
      color: 0x6b6f7d,
      transparent: true,
      opacity: 0.8,
    });
    boardGroup.add(new THREE.LineSegments(gridGeo, gridMat));

    buildLabels(size);

    const center = new THREE.Vector3((w - 1) / 2, 0, (h - 1) / 2);
    controls.target.copy(center);
    camera.position.set(center.x - w * 0.3, Math.max(w, h) * 1.3, center.z + h * 0.9);
    camera.lookAt(center);
  }

  // Seeds `layers` from whatever state is current (page load or new game).
  // Pre-existing stacks are assumed solid in their current owner's color --
  // the server state has no record of what was really underneath, so this
  // is the best available guess. Moves applied *during this session* from
  // here on are replayed precisely via `applyMoveToLayers`, so gaps under
  // bridging lintels render correctly going forward.
  function initLayers(state) {
    layers = state.board.map((square) => {
      const col = [];
      for (let i = 0; i < square.height; i++) col.push(square.piece);
      return col;
    });
    beams = [];
    nextBeamId = 0;
  }

  // Replays one applied move into the physical layer/beam model. `owner` is
  // the player who made the move (the mover, not the post-move `player`
  // field which has already advanced to the other side).
  function applyMoveToLayers(move, owner) {
    if (!move || !currentState) return;
    const { w } = currentState.size;
    const [pieceTag, index] = move;

    if (pieceTag === "Sarsen") {
      layers[index].push(owner);
      return;
    }

    const orientation = pieceTag.Lintel;
    const step = orientation === "Horizontal" ? 1 : w;
    const cells = [index, index + step, index + 2 * step];
    const level = layers[cells[0]].length;
    const beamId = nextBeamId++;

    layers[cells[0]].push({ beam: beamId });
    layers[cells[2]].push({ beam: beamId });
    while (layers[cells[1]].length < level) layers[cells[1]].push(null);
    layers[cells[1]].push({ beam: beamId });

    beams.push({ level, orientation, cells, color: owner });
  }

  // Renders the physical stack from `layers` (unit cubes, skipping gaps and
  // beam-claimed levels) plus one merged 3x1x1 box per lintel in `beams` --
  // a placed lintel should read as a single beam, not three cubes, and the
  // cell it bridges should stay visually empty below the beam if nothing
  // was actually built there.
  function buildPieces(state) {
    clearGroup(piecesGroup);
    const { w } = state.size;
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 });
    const cubeGeo = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
    const cubeEdges = new THREE.EdgesGeometry(cubeGeo);

    layers.forEach((col, idx) => {
      const x = idx % w;
      const z = Math.floor(idx / w);
      col.forEach((entry, level) => {
        if (!entry || typeof entry === "object") return; // gap or beam-claimed
        const color = entry === "Black" ? BLACK_COLOR : WHITE_COLOR;
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 });
        const cube = new THREE.Mesh(cubeGeo, mat);
        cube.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
        piecesGroup.add(cube);
        const line = new THREE.LineSegments(cubeEdges, edgeMat);
        line.position.copy(cube.position);
        piecesGroup.add(line);
      });
    });

    beams.forEach((beam) => {
      const [c0, c1, c2] = beam.cells;
      const x = c1 % w;
      const z = Math.floor(c1 / w);
      const color = beam.color === "Black" ? BLACK_COLOR : WHITE_COLOR;
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 });
      const sizeX = beam.orientation === "Horizontal" ? 2 + CUBE : CUBE;
      const sizeZ = beam.orientation === "Vertical" ? 2 + CUBE : CUBE;
      const geo = new THREE.BoxGeometry(sizeX, CUBE, sizeZ);
      const box = new THREE.Mesh(geo, mat);
      box.position.set(x, beam.level * LEVEL_H + LEVEL_H / 2, z);
      piecesGroup.add(box);
      const line = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
      line.position.copy(box.position);
      piecesGroup.add(line);
    });
  }

  function footprintFor(pieceTag, index, w) {
    if (pieceTag === "Sarsen") return [index];
    const orientation = pieceTag.Lintel;
    if (orientation === "Horizontal") return [index, index + 1, index + 2];
    return [index, index + w, index + 2 * w];
  }

  function movesForMode() {
    return currentLegalMoves.filter((mv) => {
      const piece = mv[0];
      if (mode === "sarsen") return piece === "Sarsen";
      if (mode === "lintelH")
        return typeof piece === "object" && piece.Lintel === "Horizontal";
      if (mode === "lintelV")
        return typeof piece === "object" && piece.Lintel === "Vertical";
      return false;
    });
  }

  function rebuildHighlights() {
    clearGroup(highlightGroup);
    pickables = [];
    clearGhost();
    if (!currentState || currentState.terminal || busy) return;

    const { w } = currentState.size;
    const color = mode === "sarsen" ? SARSEN_HILITE : LINTEL_HILITE;
    const geo = new THREE.PlaneGeometry(0.86, 0.86);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    movesForMode().forEach((mv) => {
      const index = mv[1];
      const footprint = footprintFor(mv[0], index, w);
      footprint.forEach((cellIdx) => {
        const square = currentState.board[cellIdx];
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

  function onClick(event) {
    if (busy || !pickables.length) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    if (hits.length === 0) return;
    postMove(hits[0].object.userData.move);
  }

  function clearGhost() {
    clearGroup(ghostGroup);
    hoveredMove = null;
  }

  // Renders a translucent preview of the piece `mode`'s current move would
  // place, at the cell(s)/height it would actually land on -- reuses
  // `footprintFor` and the same beam-vs-cube shaping as `buildPieces` so the
  // preview matches what placing it would actually look like.
  function buildGhost(move) {
    clearGroup(ghostGroup);
    if (!move || !currentState) return;
    const { w } = currentState.size;
    const color = currentState.player === "Black" ? BLACK_COLOR : WHITE_COLOR;
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.6,
      metalness: 0.05,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    });
    const [pieceTag, index] = move;
    const level = currentState.board[index].height;

    if (pieceTag === "Sarsen") {
      const x = index % w;
      const z = Math.floor(index / w);
      const cube = new THREE.Mesh(new THREE.BoxGeometry(CUBE, CUBE, CUBE), mat);
      cube.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
      ghostGroup.add(cube);
      return;
    }

    const orientation = pieceTag.Lintel;
    const cells = footprintFor(pieceTag, index, w);
    const mid = cells[1];
    const x = mid % w;
    const z = Math.floor(mid / w);
    const sizeX = orientation === "Horizontal" ? 2 + CUBE : CUBE;
    const sizeZ = orientation === "Vertical" ? 2 + CUBE : CUBE;
    const box = new THREE.Mesh(new THREE.BoxGeometry(sizeX, CUBE, sizeZ), mat);
    box.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
    ghostGroup.add(box);
  }

  function onPointerMove(event) {
    if (busy || !pickables.length) {
      clearGhost();
      return;
    }
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    const move = hits.length ? hits[0].object.userData.move : null;
    if (JSON.stringify(move) === JSON.stringify(hoveredMove)) return;
    hoveredMove = move;
    buildGhost(move);
  }

  function setMode(next) {
    mode = next;
    document.querySelectorAll("button.mode").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    rebuildHighlights();
  }

  function updateHud(state) {
    const turnEl = document.getElementById("turn");
    const bannerEl = document.getElementById("banner");

    if (state.terminal) {
      if (state.winner) {
        turnEl.textContent = "Game over";
        bannerEl.textContent = `${state.winner} wins!`;
        bannerEl.style.color = state.winner === "Black" ? "#c9cbd4" : "#f2e9d8";
      } else {
        turnEl.textContent = "Game over";
        bannerEl.textContent = "No moves left — draw.";
        bannerEl.style.color = "#e8e8ec";
      }
    } else {
      turnEl.textContent = `${state.player} to move`;
      bannerEl.textContent = "";
    }

    document.getElementById("hand-black").textContent =
      `Black — ${state.hand_black.sarsens} sarsens, ${state.hand_black.lintels} lintels`;
    document.getElementById("hand-white").textContent =
      `White — ${state.hand_white.sarsens} sarsens, ${state.hand_white.lintels} lintels`;
  }

  // --- Top-down board minimap (upper-right HUD) ---

  let minimapDpr = 1;

  function setupMinimap() {
    const canvas = document.getElementById("minimap-canvas");
    if (!canvas) return;
    const css = 236;
    minimapDpr = window.devicePixelRatio || 1;
    canvas.style.width = css + "px";
    canvas.style.height = css + "px";
    canvas.width = Math.round(css * minimapDpr);
    canvas.height = Math.round(css * minimapDpr);
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  // Taller stacks read as slightly brighter — a height cue that survives
  // the drop from 3D to a flat top-down view.
  function shadeForHeight(piece, height) {
    const t = Math.min(1, height / 12);
    const base = piece === "Black" ? [58, 61, 70] : [242, 233, 216];
    const lit = piece === "Black" ? [112, 120, 142] : [255, 253, 246];
    const c = base.map((v, i) => Math.round(v + (lit[i] - v) * t));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }

  function playerAccent(player) {
    return player === "Black" ? "#9aa2b8" : "#f2e9d8";
  }

  // BFS for a border-to-border path of `color`'s cells (Black: top row to
  // bottom row; White: left column to right column), moving through
  // 4-adjacent cells the player owns. Returns the cell indices along one
  // winning route, or null if no connection exists.
  function findWinningPath(board, w, h, color) {
    const idx = (x, y) => y * w + x;
    const owned = (i) => board[i] && board[i].piece === color;
    const starts = [];
    const goal = new Set();
    if (color === "Black") {
      for (let x = 0; x < w; x++) starts.push(idx(x, 0));
      for (let x = 0; x < w; x++) goal.add(idx(x, h - 1));
    } else {
      for (let y = 0; y < h; y++) starts.push(idx(0, y));
      for (let y = 0; y < h; y++) goal.add(idx(w - 1, y));
    }

    const prev = new Map();
    const queue = starts.filter(owned);
    queue.forEach((i) => prev.set(i, -1));

    let reached = -1;
    for (let head = 0; head < queue.length && reached < 0; head++) {
      const cur = queue[head];
      if (goal.has(cur)) {
        reached = cur;
        break;
      }
      const cx = cur % w;
      const cy = Math.floor(cur / w);
      const neighbors = [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]];
      for (const [nx, ny] of neighbors) {
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = idx(nx, ny);
        if (!owned(ni) || prev.has(ni)) continue;
        prev.set(ni, cur);
        queue.push(ni);
      }
    }
    if (reached < 0) return null;

    const path = [];
    for (let i = reached; i >= 0; i = prev.get(i)) path.push(i);
    return path.reverse();
  }

  function updateMinimap(state) {
    const canvas = document.getElementById("minimap-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(minimapDpr, 0, 0, minimapDpr, 0, 0);
    const cssW = canvas.width / minimapDpr;
    const cssH = canvas.height / minimapDpr;
    ctx.clearRect(0, 0, cssW, cssH);

    const { w, h } = state.size;
    const pad = 14;
    const gap = 3;
    const cell = Math.max(
      3,
      Math.min(
        (cssW - pad * 2 - (w - 1) * gap) / w,
        (cssH - pad * 2 - (h - 1) * gap) / h
      )
    );
    const gridW = cell * w + gap * (w - 1);
    const gridH = cell * h + gap * (h - 1);
    const ox = (cssW - gridW) / 2;
    const oy = (cssH - gridH) / 2;

    // Graph connectors: thick same-color links between 4-adjacent cells,
    // drawn underneath cells so only the stubs in the inter-cell gaps show
    // — the board's connection graph at a glance.
    ctx.lineCap = "round";
    for (let i = 0; i < state.board.length; i++) {
      const sq = state.board[i];
      if (!sq.piece) continue;
      const x = i % w;
      const y = Math.floor(i / w);
      const cx = ox + x * (cell + gap) + cell / 2;
      const cy = oy + y * (cell + gap) + cell / 2;
      ctx.strokeStyle = shadeForHeight(sq.piece, sq.height);
      ctx.lineWidth = cell * 0.85;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      if (x + 1 < w && state.board[i + 1].piece === sq.piece) {
        ctx.lineTo(ox + (x + 1) * (cell + gap) + cell / 2, cy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
      }
      if (y + 1 < h && state.board[i + w].piece === sq.piece) {
        ctx.lineTo(cx, oy + (y + 1) * (cell + gap) + cell / 2);
        ctx.stroke();
      }
    }

    // Cells
    for (let i = 0; i < state.board.length; i++) {
      const sq = state.board[i];
      const x = i % w;
      const y = Math.floor(i / w);
      const px = ox + x * (cell + gap);
      const py = oy + y * (cell + gap);
      roundRect(ctx, px, py, cell, cell, Math.max(2, cell * 0.22));
      ctx.fillStyle = sq.piece ? shadeForHeight(sq.piece, sq.height) : "#23252c";
      ctx.fill();
    }

    // Winning connection, when one exists: a glowing route through the
    // winner's cells from one border to the other.
    if (state.terminal && state.winner) {
      const path = findWinningPath(state.board, w, h, state.winner);
      if (path) {
        const glowColor = state.winner === "Black" ? "#8f9bff" : "#ffd98a";
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

    // Turn ring around the grid; falls back to winner / neutral when over.
    const ringColor = state.terminal
      ? state.winner
        ? playerAccent(state.winner)
        : "#6b6e78"
      : playerAccent(state.player);
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 2;
    roundRect(ctx, ox - 4, oy - 4, gridW + 8, gridH + 8, 6);
    ctx.stroke();

    // Turn dot in the panel title, matching the hand colors in the main HUD.
    const dot = document.getElementById("minimap-turn-dot");
    dot.style.background = state.terminal
      ? state.winner
        ? state.winner === "Black" ? "#3a3d46" : "#f2e9d8"
        : "#6b6e78"
      : state.player === "Black" ? "#3a3d46" : "#f2e9d8";
  }

  function setBusy(value) {
    busy = value;
    // Only disable the buttons that would actually race a request in
    // flight. "New Game" (and the dialog it opens) and the autoplay toggle
    // must stay usable even mid-AI-turn -- during AI-vs-AI play `busy` is
    // true almost continuously (each move immediately chains into the
    // next), so gating those on it made them unclickable in practice.
    document
      .querySelectorAll("#modes button, #ai-move")
      .forEach((btn) => (btn.disabled = value));
    rebuildHighlights();
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async function refresh({ rebuildBoard } = {}) {
    const [state, legalMoves] = await Promise.all([
      fetchJson("/api/state"),
      fetchJson("/api/legal_moves"),
    ]);
    currentState = state;
    currentLegalMoves = legalMoves;

    if (rebuildBoard) {
      buildBoard(state.size);
      initLayers(state);
    }
    buildPieces(state);
    rebuildHighlights();
    updateHud(state);
    updateMinimap(state);
  }

  async function postMove(move) {
    const owner = currentState.player;
    const epoch = gameEpoch;
    setBusy(true);
    try {
      await fetchJson("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(move),
      });
      if (epoch !== gameEpoch) return; // a new game started while this was in flight
      applyMoveToLayers(move, owner);
      await refresh();
    } catch (err) {
      console.error("move rejected", err);
    } finally {
      setBusy(false);
    }
    if (epoch !== gameEpoch) return;
    // Must run after setBusy(false) -- maybeTriggerAiTurn bails out while busy.
    await maybeTriggerAiTurn();
  }

  // `preset` picks which AI config plays this one move. Used both for
  // seat-driven auto-play and for the manual "AI Move" button.
  async function aiMove(preset) {
    const owner = currentState.player;
    const epoch = gameEpoch;
    setBusy(true);
    document.getElementById("banner").textContent = "AI is thinking…";
    try {
      const result = await fetchJson("/api/ai_move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset }),
      });
      if (epoch !== gameEpoch) return; // a new game started while this was in flight
      applyMoveToLayers(result.last_move, owner);
      await refresh();
    } catch (err) {
      console.error("AI move failed", err);
    } finally {
      setBusy(false);
    }
    if (epoch !== gameEpoch) return;
    // Must run after setBusy(false) -- maybeTriggerAiTurn bails out while busy.
    await maybeTriggerAiTurn();
  }

  // If it's a non-human seat's turn, play it automatically. Chains on its
  // own for AI-vs-AI (aiMove calls this again after its own refresh).
  async function maybeTriggerAiTurn() {
    if (busy || autoplayPaused || !currentState || currentState.terminal) return;
    const seat = seats[currentState.player];
    if (seat === "human") return;
    await aiMove(seat);
  }

  function setAutoplayPaused(value) {
    autoplayPaused = value;
    const btn = document.getElementById("autoplay-toggle");
    btn.textContent = autoplayPaused ? "Resume" : "Pause";
    btn.classList.toggle("paused", autoplayPaused);
  }

  function toggleAutoplay() {
    setAutoplayPaused(!autoplayPaused);
    if (!autoplayPaused) maybeTriggerAiTurn();
  }

  // The preset the manual "AI Move" button uses: the current seat's own
  // preset if it's AI-controlled, otherwise a reasonable one-off "take over
  // for me" strength.
  function presetForManualMove() {
    const seat = seats[currentState.player];
    return seat === "human" ? "strong" : seat;
  }

  function populateSeatSelectors() {
    ["seat-black", "seat-white"].forEach((id) => {
      const sel = document.getElementById(id);
      const previous = sel.value;
      sel.innerHTML = "";
      sel.appendChild(new Option("Human", "human"));
      aiPresets.forEach((p) => sel.appendChild(new Option(`AI: ${p.label}`, p.id)));
      if (previous) sel.value = previous;
    });
  }

  async function loadAiPresets() {
    try {
      aiPresets = await fetchJson("/api/ai_presets");
      populateSeatSelectors();
    } catch (err) {
      console.error("failed to load AI presets", err);
    }
  }

  function openNewGameDialog() {
    document.getElementById("seat-black").value = seats.Black;
    document.getElementById("seat-white").value = seats.White;
    document.getElementById("new-game-dialog").showModal();
  }

  async function startNewGame() {
    const [w, h] = document
      .getElementById("new-size")
      .value.split("x")
      .map(Number);
    seats.Black = document.getElementById("seat-black").value;
    seats.White = document.getElementById("seat-white").value;
    gameEpoch++; // invalidate any in-flight move/AI request from the old game

    await fetchJson("/api/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ size: { w, h } }),
    });
    document.getElementById("new-game-dialog").close();
    setAutoplayPaused(false);
    await refresh({ rebuildBoard: true });
    await maybeTriggerAiTurn();
  }

  const HOTKEYS = { 1: "sarsen", 2: "lintelH", 3: "lintelV" };

  function onKeyDown(event) {
    if (busy) return;
    const tag = event.target.tagName;
    if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
    const next = HOTKEYS[event.key];
    if (next) setMode(next);
  }

  function initUi() {
    document.querySelectorAll("button.mode").forEach((btn) => {
      btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
    document.getElementById("new-game").addEventListener("click", openNewGameDialog);
    document.getElementById("new-game-cancel").addEventListener("click", () => {
      document.getElementById("new-game-dialog").close();
    });
    document.getElementById("new-game-form").addEventListener("submit", (event) => {
      event.preventDefault();
      startNewGame();
    });
    document
      .getElementById("ai-move")
      .addEventListener("click", () => aiMove(presetForManualMove()));
    document.getElementById("autoplay-toggle").addEventListener("click", toggleAutoplay);
    window.addEventListener("keydown", onKeyDown);
  }

  initScene();
  initUi();
  setupMinimap();
  loadAiPresets();
  refresh({ rebuildBoard: true });
})();
