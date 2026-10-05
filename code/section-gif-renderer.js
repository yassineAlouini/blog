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
  function research(t) {
    // A matrix becomes a little eigenvalue histogram, echoing the RMT series.
    rect(8,54,80,3,1);
    line(10,13,10,44,4);line(10,13,14,13,4);line(10,44,14,44,4);
    line(41,13,41,44,4);line(37,13,41,13,4);line(37,44,41,44,4);
    for(let row=0;row<4;row++) for(let col=0;col<4;col++) {
      const value = Math.sin(t*TAU+(row+col)*.8);
      rect(15+col*6,17+row*6,4,4,value>.35 ? 6 : value<-.35 ? 4 : 2);
    }
    line(46,29,54,29,3);line(51,26,54,29,3);line(51,32,54,29,3);
    poly([[56,47],[75,55],[90,47],[71,39]],1);
    for(let i=0;i<5;i++) {
      const x=57+i*6, y=44+i*1;
      const height=7+Math.round(Math.sin((i+1)*Math.PI/6)*15 + Math.sin(t*TAU+i*.5)*3);
      poly([[x,y-height],[x+4,y-height+2],[x+4,y+2],[x,y]],i===2 ? 6 : 3);
      poly([[x+4,y-height+2],[x+6,y-height],[x+6,y],[x+4,y+2]],4);
      poly([[x,y-height],[x+2,y-height-2],[x+6,y-height],[x+4,y-height+2]],i===2 ? 5 : 2);
    }
  }
  const drawings = {vision, video, language, research};
  window.SECTION_PALETTE = C;
  window.drawSection = (name, t) => { rect(0,0,96,64,0); drawings[name](t); };
})();
