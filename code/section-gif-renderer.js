/* Deterministic pixel drawings for the four section GIFs. Exported offline;
   the blog itself only loads GIF/PNG images and lightweight playback controls. */
(() => {
  "use strict";
  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const C = ["#ffffff", "#edf1f5", "#b8c5d2", "#526276", "#193653", "#f6efdf", "#b69a70", "#745020"];
  const TAU = Math.PI * 2;
  function rect(x, y, w, h, color) {
    ctx.fillStyle = C[color];
    ctx.fillRect(Math.round(x), Math.round(y), w, h);
  }
  function line(x1, y1, x2, y2, color) {
    const n = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
    for (let i = 0; i <= n; i++) {
      const t = n ? i / n : 0;
      rect(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, 1, 1, color);
    }
  }
  function poly(points, color) {
    const low = Math.ceil(Math.min(...points.map(p => p[1])));
    const high = Math.ceil(Math.max(...points.map(p => p[1])));
    for (let y = low; y < high; y++) {
      const crossings = [];
      for (let i = 0; i < points.length; i++) {
        const a = points[i], b = points[(i + 1) % points.length];
        if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) {
          crossings.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
        }
      }
      crossings.sort((a, b) => a - b);
      for (let i = 0; i + 1 < crossings.length; i += 2) {
        rect(Math.ceil(crossings[i]), y, Math.ceil(crossings[i + 1]) - Math.ceil(crossings[i]), 1, color);
      }
    }
  }
  function brackets(x, y, w, h, color) {
    for (const [a, b, dx, dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]) {
      line(a,b,a+dx*4,b,color); line(a,b,a,b+dy*4,color);
    }
  }
  function vision(t) {
    // A tiny monitor: a scanning mask separates a mountain from the sky.
    rect(17,57,64,3,1);
    poly([[16,9],[78,9],[83,14],[83,49],[78,45],[16,45]],3);
    rect(13,8,65,39,4); rect(16,11,59,32,1);
    rect(22,16,5,5,6);
    const terrain = [[18,39],[34,19],[44,31],[52,23],[72,39]];
    poly(terrain,2);
    const scan = 18 + Math.round((.5 - .5 * Math.cos(t * TAU)) * 53);
    ctx.save(); ctx.beginPath(); ctx.rect(18,12,scan-18,29); ctx.clip();
    poly(terrain,6);
    for (let y=19;y<39;y+=4) for(let x=19;x<73;x+=4) {
      // Checker cells are clipped to a second copy of the mountain silhouette.
      const inLeft = x>=34-(y-19)*.8 && x<=34+(y-19)*.84;
      if(inLeft) rect(x,y,2,2,5);
    }
    ctx.restore();
    brackets(28,16,32,24,4);
    line(scan,12,scan,41,7); rect(scan-1,10,3,2,7);
    rect(41,47,10,7,3); rect(32,54,29,3,4); rect(72,44,2,1,6);
  }
  function video(t) {
    // A continuous sprocketed filmstrip with a bouncing ball in each frame.
    rect(9,52,79,4,1);
    poly([[9,12],[85,12],[88,16],[88,50],[85,46],[9,46]],3);
    rect(8,11,78,37,4);
    ctx.save(); ctx.beginPath(); ctx.rect(10,12,74,35); ctx.clip();
    const shift = Math.floor(t * 28);
    for(let x=-18-shift;x<110;x+=28) {
      rect(x,18,24,23,1);
      line(x+2,37,x+21,37,2);
      const height = Math.round(Math.sin(t * TAU) * 5);
      rect(x+9,25-height,6,6,6); rect(x+10,24-height,4,1,7);
      for(let j=0;j<3;j++) { rect(x+j*9,13,4,3,0); rect(x+j*9,43,4,3,0); }
    }
    ctx.restore();
    brackets(34,16,26,27,7);
    rect(32,53,32,2,2); rect(32,53,Math.round(t*32),2,6);
  }
  function language(t) {
    // Token tiles flow through a speech bubble into a generated reply.
    rect(14,54,68,3,1);
    poly([[18,10],[78,10],[82,14],[82,41],[78,37],[18,37]],3);
    rect(15,8,63,30,4); rect(18,11,57,24,0);
    poly([[23,37],[35,37],[23,46]],4);
    for(let i=0;i<5;i++) {
      const active = Math.floor(t*5) === i;
      rect(22+i*10,16,8,7,active ? 6 : 2);
      rect(23+i*10,17,4,1,active ? 5 : 0);
    }
    const count = Math.floor((.5-.5*Math.cos(t*TAU))*5);
    for(let i=0;i<count;i++) rect(22+i*9,28,6,2,4);
    rect(22+count*9,26,2,6,7);
    for(let i=0;i<3;i++) {
      const x=47+i*10, y=44-Math.round(Math.sin(t*TAU+i*.7)*2);
      rect(x+1,y+1,7,6,4); rect(x,y,7,5,5); rect(x+1,y+1,4,1,6);
    }
  }
  // A reproducible toy weight matrix: noise plus a changing rank-one signal.
  // Its actual WᵀW eigenvalues drive the spectrum, rather than decorative bars.
  const weightSize = 12;
  let randomState = 731;
  function random() {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return (randomState + .5) / 4294967296;
  }
  const noise = Array.from({length: weightSize}, () =>
    Array.from({length: weightSize}, () =>
      Math.sqrt(-2 * Math.log(random())) * Math.cos(TAU * random()) / Math.sqrt(weightSize)));
  const mode = Array.from({length: weightSize}, (_, i) => (i % 3 === 0 ? -1 : 1) / Math.sqrt(weightSize));
  function eigenvalues(matrix) {
    // Jacobi rotations for this tiny real symmetric matrix (offline export only).
    const a = matrix.map(row => [...row]), n = a.length;
    for (let iteration = 0; iteration < 30 * n * n; iteration++) {
      let p = 0, q = 1;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        if (Math.abs(a[i][j]) > Math.abs(a[p][q])) { p = i; q = j; }
      }
      if (Math.abs(a[p][q]) < 1e-9) break;
      const angle = .5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
      const c = Math.cos(angle), s = Math.sin(angle);
      const pp = a[p][p], qq = a[q][q], pq = a[p][q];
      for (let k = 0; k < n; k++) if (k !== p && k !== q) {
        const kp = a[k][p], kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
      a[p][p] = c*c*pp - 2*s*c*pq + s*s*qq;
      a[q][q] = s*s*pp + 2*s*c*pq + c*c*qq;
      a[p][q] = a[q][p] = 0;
    }
    return a.map((row, i) => Math.max(0, row[i])).sort((a, b) => a - b);
  }
  function research(t) {
    // A learned-direction metaphor: a gold mode emerges from the blue noise bulk.
    // The strength breathes smoothly back to zero for a seamless loop.
    const strength = 3 * (.5 - .5 * Math.cos(TAU * t));
    const w = noise.map((row, i) => row.map((x, j) => x + strength * mode[i] * mode[j]));
    const gram = w.map((_, i) => w.map((_, j) =>
      w.reduce((sum, row) => sum + row[i] * row[j], 0)));
    const spectrum = eigenvalues(gram);
    rect(5,38,27,2,1);
    brackets(4,6,27,29,4);
    for (let row = 0; row < 4; row++) for (let col = 0; col < 4; col++) {
      const value = w[row][col];
      rect(9+col*5,11+row*5,3,3,value > .22 ? 6 : value < -.22 ? 4 : 2);
    }
    line(33,22,40,22,3); line(37,19,40,22,3); line(37,25,40,22,3);
    const xs = [46,64,83], ys = [10,22,34];
    for (let layer = 0; layer < 2; layer++) {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        line(xs[layer],ys[i],xs[layer+1],ys[j],1);
      }
      // Gold pulses travel forward along one feature path.
      const path = [2,1,0], i = path[layer], j = path[layer+1];
      line(xs[layer],ys[i],xs[layer+1],ys[j],strength > 1 ? 6 : 2);
      const pulse = (t*4 - layer*.5 + 2) % 1;
      if (strength > .3) rect(xs[layer]+(xs[layer+1]-xs[layer])*pulse-1,
                            ys[i]+(ys[j]-ys[i])*pulse-1,3,3,7);
    }
    for (let layer = 0; layer < 3; layer++) for (let i = 0; i < 3; i++) {
      rect(xs[layer]-2,ys[i]-2,5,5,4);
      rect(xs[layer]-1,ys[i]-1,3,3,i === 2-layer && strength > 1 ? 6 : 0);
    }
    // Fixed spectral axis 0–12. Blue histogram excludes the largest eigenvalue;
    // the gold lollipop tracks that largest eigenvalue throughout the loop.
    line(9,60,90,60,3);
    for (let x = 9; x <= 90; x += 27) rect(x,61,1,2,2);
    const bins = Array(27).fill(0);
    for (const value of spectrum.slice(0,-1)) bins[Math.min(26, Math.floor(value/12*27))]++;
    bins.forEach((count, i) => {
      if (count) rect(10+i*3,59-count*3,2,count*3,3);
    });
    const peak = 10 + Math.round(spectrum[spectrum.length-1]/12*78);
    line(peak,47,peak,59,7);
    rect(peak-1,45,3,3,6); rect(peak,44,1,1,7);
    // A miniature lambda labels the spectrum without relying on canvas fonts.
    line(2,49,6,57,4); line(4,53,1,58,4);
  }
  const drawings = {vision, video, language, research};
  window.SECTION_PALETTE = C;
  window.drawSection = (name, t) => { rect(0,0,96,64,0); drawings[name](t); };
})();
