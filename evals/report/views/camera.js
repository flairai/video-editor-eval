// The camera-path tasks' analysis (evals/report/views/camera.py): the turn at each step, requested against what
// VGGT-Omega recovered from the grid; the evidence images drawn from the reconstruction; and the reconstruction
// itself in 3D (three.js, loaded from a CDN when opened).
(function () {
  let three = null, viewer = null;

  function lineChart(C, box, xs, series, opt) {
    box.innerHTML = "";
    const W = Math.max(box.clientWidth, 320), H = opt.h || 200, m = {l: 40, r: 8, t: 8, b: 26};
    const X = i => m.l + (W - m.l - m.r) * (i / Math.max(1, xs.length - 1)), Y = v => m.t + (H - m.t - m.b) * (1 - (v - opt.min) / (opt.max - opt.min));
    const svg = C.svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.label}, box);
    for (const tv of opt.ticks) { C.svgEl("line", {x1: m.l, x2: W - m.r, y1: Y(tv), y2: Y(tv), stroke: "var(--line)"}, svg); C.txt(svg, m.l - 8, Y(tv) + 4, opt.fmt(tv), {"text-anchor": "end"}); }
    xs.forEach((c, i) => C.txt(svg, X(i), H - 6, c, {"text-anchor": "middle"}));
    for (const sr of series) {
      C.svgEl("polyline", {points: sr.v.map((v, i) => `${X(i)},${Y(Math.min(v, opt.max))}`).join(" "), fill: "none", stroke: sr.c, "stroke-width": 1.5, "stroke-linejoin": "round"}, svg);
      sr.v.forEach((v, i) => C.hover(C.svgEl("circle", {cx: X(i), cy: Y(Math.min(v, opt.max)), r: 4, fill: sr.c, stroke: "var(--bg)", "stroke-width": 2}, svg), `<b>Step ${xs[i]}</b><br>${series.map(s => `${s.n} ${opt.fmt(s.v[i])}`).join("<br>")}`));
    }
  }

  async function explore(C, el, task, v) {
    el.innerHTML = '<p class="faint" style="padding:16px;margin:0">Loading</p>';
    try {
      if (!three) { const T = await import("three"), O = await import("three/addons/controls/OrbitControls.js"); three = {T, OrbitControls: O.OrbitControls}; }
      const c = await (await fetch(v.cloud)).json();
      draw3d(C, el, task, c, three.T, three.OrbitControls);
    } catch (err) {
      el.innerHTML = `<p class="faint" style="padding:16px;margin:0">The 3D view needs the page served over HTTP (${C.esc(err.message || err)}).</p>`;
    }
  }

  function draw3d(C, el, task, c, THREE, OrbitControls) {
    if (viewer) viewer.stop();
    el.innerHTML = "";
    const E = task.viz.view.extent, W = el.clientWidth, H = 480;
    const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(devicePixelRatio); renderer.setSize(W, H); el.append(renderer.domElement);
    renderer.setClearColor(new THREE.Color("#0b0b0d"));
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 1000); camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    const BLUES = ["#9cc4ff", "#86b6ff", "#70a8ff", "#5b9dff", "#4a8cf0", "#3c7cdc", "#306cc6", "#265cb0", "#1d4d99"].map(h => new THREE.Color(h));
    const OR = [new THREE.Color("#2a2a2e"), new THREE.Color("#ff6b63")];
    const dec = (str, T) => { const b = atob(str); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
    const q16 = dec(c.xyz, Int16Array), xyz = new Float32Array(q16.length); for (let i = 0; i < q16.length; i++) xyz[i] = c.centre[i % 3] + q16[i] * c.step;
    const rgb = dec(c.rgb, Uint8Array), fr = dec(c.frame, Uint8Array), er = dec(c.err, Uint8Array), n = xyz.length / 3;
    const cols = {real: new Float32Array(n * 3), cell: new Float32Array(n * 3), err: new Float32Array(n * 3)};
    for (let i = 0; i < n; i++) { for (let k = 0; k < 3; k++) cols.real[3 * i + k] = rgb[3 * i + k] / 255; const b = BLUES[fr[i]] || BLUES[8]; cols.cell.set([b.r, b.g, b.b], 3 * i); const o = OR[0].clone().lerp(OR[1], er[i] / 255); cols.err.set([o.r, o.g, o.b], 3 * i); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(xyz, 3)); const ca = new THREE.BufferAttribute(cols.real.slice(), 3); geo.setAttribute("color", ca);
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({size: 0.0045 * E, vertexColors: true})));
    const grey = new THREE.Color("#5c5c64"), req = c.requested.map(p => new THREE.Vector3(...p));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(req), new THREE.LineBasicMaterial({color: grey})));
    const labels = [], cs = [];
    c.cams.forEach((cam, k) => {
      const Cn = new THREE.Vector3(...cam.c); cs.push(Cn); const Rm = cam.r, ax = j => new THREE.Vector3(Rm[0][j], Rm[1][j], Rm[2][j]);
      const f = ax(2), r = ax(0), u = ax(1).multiplyScalar(-1), L = 0.045 * E, w = 0.03 * E;
      const q = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => Cn.clone().add(f.clone().multiplyScalar(L)).add(r.clone().multiplyScalar(a * w)).add(u.clone().multiplyScalar(b * w * 0.6)));
      const seg = []; q.forEach(p => seg.push(Cn, p)); for (let i = 0; i < 4; i++) seg.push(q[i], q[(i + 1) % 4]);
      scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({color: BLUES[k] || BLUES[8]})));
      const lab = document.createElement("div"); lab.textContent = String(k + 1);
      lab.style.cssText = "position:absolute;font:500 11px var(--mono);color:var(--text);pointer-events:none;transform:translate(6px,-14px)";
      el.append(lab); labels.push([lab, Cn]);
    });
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(cs), new THREE.LineBasicMaterial({color: new THREE.Color("#9a9aa2")})));
    const T = new THREE.Vector3(...task.viz.view.target); controls.target.copy(T); camera.position.set(T.x - 0.55 * E, T.y - 0.75 * E, T.z + 0.6 * E); controls.update();
    const tools = document.createElement("div"); tools.className = "tools seg";
    tools.innerHTML = [["real", "Colour"], ["cell", "Frame"], ["err", "Disagreement"]].map(([m, l], i) => `<button type="button" data-mode="${m}" aria-pressed="${i === 0}">${l}</button>`).join("");
    el.append(tools);
    tools.querySelectorAll("button").forEach(b => b.onclick = () => { tools.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); ca.array.set(cols[b.dataset.mode]); ca.needsUpdate = true; });
    let live = true; const v3 = new THREE.Vector3();
    (function loop() { if (!live || !el.isConnected) { renderer.dispose(); return; } controls.update(); renderer.render(scene, camera); for (const [lab, P] of labels) { v3.copy(P).project(camera); lab.style.left = ((v3.x + 1) / 2 * W) + "px"; lab.style.top = ((1 - v3.y) / 2 * H) + "px"; lab.style.display = v3.z < 1 ? "block" : "none"; } requestAnimationFrame(loop); })();
    viewer = {stop() { live = false; }};
  }

  const NAMES = {"path.png": "Path", "cloud_top.png": "From above", "cloud_overview.png": "Overview", "cloud_frames.png": "By frame",
                 "cloud_error.png": "Disagreement", "worst_pair.png": "Worst pair", "closure_pair.png": "First and last"};
  VIEWS["camera-path"] = {render(box, task, trial, C) {
    const v = trial.viz;
    if (viewer) { viewer.stop(); viewer = null; }
    if (!v) { box.innerHTML = '<p class="dim" style="margin:0">Not reconstructed: the grid failed its gate.</p>'; return; }
    const steps = v.turnReq.map((_, i) => String(i + 1)), mean = v.rotErr.reduce((a, b) => a + b, 0) / v.rotErr.length;
    const facts = [["Turn error", `${mean.toFixed(1)}°`], ["Path fit", `${(v.fit * 100).toFixed(0)}%`], ["Disagreeing pixels", `${Math.round(v.warp * 100)}%`]];
    if (v.closure) facts.push(["Loop gap", `${(v.closure[0] * 100).toFixed(1)}%`]);
    const ev = Object.keys(NAMES).filter(k => (v.evidence || {})[k]).map(k => [k, v.evidence[k]]);
    const wide = k => ["path.png", "worst_pair.png", "closure_pair.png"].includes(k);
    box.innerHTML = `<dl class="facts">${facts.map(([k, x]) => `<div><dt>${C.esc(k)}</dt><dd>${C.esc(x)}</dd></div>`).join("")}</dl>
      <h3 style="margin-top:28px">Turn per step</h3><div class="chart" id="turnChart"></div>
      <div class="legend"><span><i class="sw" style="background:var(--text-3)"></i>Requested</span><span><i class="sw" style="background:var(--c1)"></i>Recovered</span></div>
      <h3>Reconstruction</h3><div class="evidence">${ev.map(([k, src]) => `<figure class="${wide(k) ? "wide" : ""}"><img src="${src}" alt="${C.esc(NAMES[k])}" loading="lazy"><figcaption>${C.esc(NAMES[k])}</figcaption></figure>`).join("")}</div>
      ${v.cloud ? '<h3>In 3D</h3><div class="canvas3d" id="viewer3d"><button class="seg" type="button" id="load3d" style="margin:16px"><span style="padding:4px 10px">Open</span></button></div>' : ""}`;
    box.querySelectorAll(".evidence img").forEach(img => { img.onclick = () => { const i = new Image(); i.src = img.src; C.zoom(i); }; });
    lineChart(C, box.querySelector("#turnChart"), steps, [{n: "Requested", v: v.turnReq, c: C.css("--text-3")}, {n: "Recovered", v: v.turnRec, c: C.css("--c1")}],
              {min: 0, max: 180, ticks: [0, 45, 90, 135, 180], fmt: x => `${Math.round(x)}°`, label: "Turn per step"});
    const b = box.querySelector("#load3d");
    if (b) b.onclick = () => explore(C, box.querySelector("#viewer3d"), task, v);
  }};
})();
