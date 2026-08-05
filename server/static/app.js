(function () {
  const CUBE = 0.92; // visual cube size, < 1.0 so stacked blocks read as distinct layers
  const LEVEL_H = 1.0; // vertical spacing per stacked layer

  const BLACK_COLOR = 0x3a3d46;
  const WHITE_COLOR = 0xf2e9d8;
  const SARSEN_HILITE = 0xffcf5c;
  const LINTEL_HILITE = 0x63d3ff;

  let scene, camera, renderer, controls, raycaster, mouse;
  let boardGroup, piecesGroup, highlightGroup;
  let pickables = [];
  let mode = "sarsen"; // "sarsen" | "lintelH" | "lintelV"
  let currentState = null;
  let currentLegalMoves = [];

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
    scene.add(boardGroup, piecesGroup, highlightGroup);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer.domElement.addEventListener("click", onClick);
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
      child.geometry && child.geometry.dispose();
      child.material && child.material.dispose();
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
      color: 0x4a4d58,
      transparent: true,
      opacity: 0.6,
    });
    boardGroup.add(new THREE.LineSegments(gridGeo, gridMat));

    const center = new THREE.Vector3((w - 1) / 2, 0, (h - 1) / 2);
    controls.target.copy(center);
    camera.position.set(center.x - w * 0.3, Math.max(w, h) * 1.3, center.z + h * 0.9);
    camera.lookAt(center);
  }

  // Piece stacks render every layer up to `height` in the *current* owner's
  // color. The server-side state model only tracks the current top owner per
  // cell (older layers can be built over by lintels), so there is no
  // per-layer history to render even if we wanted to.
  function buildPieces(state) {
    clearGroup(piecesGroup);
    const { w } = state.size;
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 });

    state.board.forEach((square, idx) => {
      if (square.height === 0 || !square.piece) return;
      const x = idx % w;
      const z = Math.floor(idx / w);
      const color = square.piece === "Black" ? BLACK_COLOR : WHITE_COLOR;
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 });
      const geo = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
      const edges = new THREE.EdgesGeometry(geo);

      for (let level = 0; level < square.height; level++) {
        const cube = new THREE.Mesh(geo, mat);
        cube.position.set(x, level * LEVEL_H + LEVEL_H / 2, z);
        piecesGroup.add(cube);
        const line = new THREE.LineSegments(edges, edgeMat);
        line.position.copy(cube.position);
        piecesGroup.add(line);
      }
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
    if (!currentState || currentState.terminal) return;

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
    if (!pickables.length) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    if (hits.length === 0) return;
    postMove(hits[0].object.userData.move);
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

    if (rebuildBoard) buildBoard(state.size);
    buildPieces(state);
    rebuildHighlights();
    updateHud(state);
  }

  async function postMove(move) {
    try {
      await fetchJson("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(move),
      });
      await refresh();
    } catch (err) {
      console.error("move rejected", err);
    }
  }

  async function newGame() {
    await fetchJson("/api/new", { method: "POST" });
    await refresh({ rebuildBoard: true });
  }

  function initUi() {
    document.querySelectorAll("button.mode").forEach((btn) => {
      btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
    document.getElementById("new-game").addEventListener("click", newGame);
  }

  initScene();
  initUi();
  refresh({ rebuildBoard: true });
})();
