/* A small, hand-drawn pixel scene. All geometry uses the same low-resolution
   isometric space; CSS enlarges the pixels without smoothing. */
(() => {
  "use strict";
  const canvas = document.querySelector("#fusion-reactor");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const ink = getComputedStyle(canvas).getPropertyValue("--navy").trim() || "#193653";
  const colors = ["#ffffff", "#edf1f5", "#d9e2ea", "#b8c8d7", "#8ca3ba", "#5c7b98", "#365774", ink];
  const TAU = Math.PI * 2;

  function box(x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), Math.round(y), w, h);
  }

  // Scanline polygons and integer lines keep every edge on the pixel grid.
  function polygon(points, color) {
    const min = Math.ceil(Math.min(...points.map(p => p[1])));
    const max = Math.ceil(Math.max(...points.map(p => p[1])));
    for (let y = min; y < max; y++) {
      const intersections = [];
      for (let i = 0; i < points.length; i++) {
        const a = points[i], b = points[(i + 1) % points.length];
        if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) {
          intersections.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
        }
      }
      intersections.sort((a, b) => a - b);
      for (let i = 0; i + 1 < intersections.length; i += 2) {
        box(Math.ceil(intersections[i]), y, Math.ceil(intersections[i + 1]) - Math.ceil(intersections[i]), 1, color);
      }
    }
  }

  function line(a, b, color, width = 1) {
    const steps = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]));
    for (let i = 0; i <= steps; i++) {
      const t = steps ? i / steps : 0;
      box(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, width, width, color);
    }
  }

  function ellipsePoint(angle, rx, ry, y = 100) {
    return [160 + Math.cos(angle) * rx, y + Math.sin(angle) * ry];
  }

  function ring(rx, ry, thickness, y, color, start = 0, end = TAU) {
    const points = [];
    const steps = Math.ceil((end - start) * 24);
    for (let i = 0; i <= steps; i++) points.push(ellipsePoint(start + (end - start) * i / steps, rx, ry, y));
    for (let i = steps; i >= 0; i--) points.push(ellipsePoint(start + (end - start) * i / steps, rx - thickness, ry - thickness * .48, y));
    polygon(points, color);
  }

  function wall(rx, ry, y, height, color, start = 0, end = Math.PI) {
    const points = [];
    for (let a = start; a <= end + .001; a += (end - start) / 64) points.push(ellipsePoint(a, rx, ry, y));
    for (let a = end; a >= start - .001; a -= (end - start) / 64) points.push(ellipsePoint(a, rx, ry, y + height));
    polygon(points, color);
  }

  function coil(angle, front) {
    const outer = ellipsePoint(angle, 98, 46, 91);
    const inner = ellipsePoint(angle, 83, 38, 91);
    const [x, y] = outer;
    const cap = [[x - 5, y - 22], [x + 5, y - 25], [inner[0] + 5, inner[1] - 25], [inner[0] - 5, inner[1] - 22]];
    polygon([[x - 5, y - 22], [x + 5, y - 25], [x + 5, y + 21], [x - 5, y + 24]], colors[front ? 6 : 5]);
    polygon([[x + 5, y - 25], [inner[0] + 5, inner[1] - 25], [inner[0] + 5, inner[1] + 17], [x + 5, y + 21]], colors[7]);
    polygon(cap, colors[3]);
    for (let j = 0; j < 6; j++) line([x - 4, y - 16 + j * 6], [x + 3, y - 18 + j * 6], colors[3]);
    box(x - 2, y - 22, 2, 3, colors[0]);
  }

  function draw(time) {
    ctx.clearRect(0, 0, 320, 220);
    box(0, 0, 320, 220, colors[0]);
    const pulse = (Math.sin(time * TAU / 3.6) + 1) / 2;

    // Ground and the three faces of the reactor's isometric plinth.
    polygon([[31, 163], [153, 106], [293, 164], [173, 215]], colors[1]);
    for (let i = -2; i <= 2; i++) {
      line([49 + i * 15, 164 + i * 7], [166 + i * 15, 111 + i * 7], colors[2]);
      line([55 + i * 15, 163 - i * 7], [174 + i * 15, 214 - i * 7], colors[2]);
    }
    polygon([[43, 147], [153, 96], [279, 150], [169, 201]], colors[7]);
    polygon([[43, 138], [153, 87], [279, 141], [169, 192]], colors[2]);
    polygon([[43, 138], [169, 192], [169, 201], [43, 147]], colors[5]);
    polygon([[169, 192], [279, 141], [279, 150], [169, 201]], colors[6]);
    line([47, 138], [169, 190], colors[0]);
    line([169, 190], [275, 141], colors[3]);

    // Lower vacuum vessel, inset panels, and the rear containment magnets.
    wall(94, 44, 115, 20, colors[7]);
    ring(94, 44, 94, 115, colors[4]);
    for (let a = .2; a < Math.PI; a += .23) {
      const p = ellipsePoint(a, 94, 44, 118);
      line(p, [p[0], p[1] + 12], colors[3], 2);
    }
    ring(95, 45, 4, 132, colors[5], 0, Math.PI);
    for (let a = Math.PI + .22; a < TAU; a += .45) coil(a, false);
    wall(91, 43, 81, 30, colors[5], Math.PI, TAU);
    ring(91, 43, 12, 81, colors[3], Math.PI, TAU);
    ring(91, 43, 2, 80, colors[0], Math.PI, TAU);
    wall(79, 37, 81, 27, colors[7], Math.PI, TAU);
    ring(86, 40, 58, 113, colors[2]);
    ring(82, 38, 2, 111, colors[4]);

    // A dark, saturated plasma torus. Its halo breathes in discrete shades;
    // bright packets circulate along the field, with no blur or gradients.
    ring(75, 35, 22, 98, colors[pulse > .55 ? 3 : 2]);
    ring(71, 33, 19, 97, colors[pulse > .35 ? 5 : 4]);
    ring(68, 31, 12 + Math.round(pulse * 3), 96, colors[7]);
    ring(65, 30, 2, 94, colors[pulse > .7 ? 0 : 3]);
    for (let i = 0; i < 9; i++) {
      const angle = (time * TAU * 4 / (9 * 3.6) + i * TAU / 9) % TAU;
      ring(64, 29, 3, 96, colors[0], angle, angle + .16);
      const p = ellipsePoint(angle + .18, 62, 28, 96);
      box(p[0], p[1], 2, 2, colors[3]);
    }

    // Central solenoid, drawn over the back of the plasma for depth.
    wall(22, 11, 67, 49, colors[7]);
    for (let y = 74; y < 116; y += 5) ring(22, 11, 3, y, colors[4], 0, Math.PI);
    polygon([[145, 67], [152, 70], [152, 116], [145, 113]], colors[5]);
    ring(25, 12, 25, 67, colors[3]);
    ring(25, 12, 3, 67, colors[0]);
    ring(14, 7, 14, 66, colors[6]);
    ring(8, 4, 8, 65, colors[7]);

    // The front is cut away so the pulsing plasma remains visible.
    wall(91, 43, 112, 8, colors[6]);
    ring(91, 43, 9, 112, colors[3], 0, Math.PI);
    ring(91, 43, 2, 111, colors[0], 0, Math.PI);
    for (const a of [.15, .63, 2.51, 2.99]) coil(a, true);

    // Service pipe and a miniature control console on the plinth.
    line([208, 158], [223, 165], colors[7], 5);
    line([223, 165], [247, 153], colors[7], 5);
    line([209, 158], [223, 164], colors[3], 2);
    polygon([[99, 157], [115, 150], [139, 160], [123, 168]], colors[7]);
    polygon([[99, 157], [123, 168], [123, 180], [99, 170]], colors[5]);
    polygon([[123, 168], [139, 160], [139, 172], [123, 180]], colors[6]);
    polygon([[103, 157], [114, 153], [130, 160], [120, 164]], colors[2]);
    line([107, 157], [119, 161], colors[7]);
    for (let i = 0; i < 3; i++) box(105 + i * 5, 167 + i * 2, 2, 2, colors[pulse > .6 ? 0 : 3]);
    // Quiet registration marks make the drawing feel like a printed plate.
    for (const [x, y, sign] of [[30, 29, 1], [288, 29, -1]]) {
      line([x, y], [x + sign * 8, y], colors[3]);
      line([x, y], [x, y + 8], colors[3]);
    }
  }

  window.drawFusionReactor = draw;
  draw(0);
})();
