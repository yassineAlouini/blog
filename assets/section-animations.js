/* GIFs work without JavaScript. Controls select stills for pause, reduced
   motion, background tabs, or illustrations that have left the viewport. */
(() => {
  "use strict";
  const button = document.querySelector(".section-animation-toggle");
  const pictures = [...document.querySelectorAll(".section-icon")];
  if (!button || !pictures.length) return;
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let paused = motion.matches;
  const visible = new Set(pictures);
  function sync() {
    for (const picture of pictures) {
      picture.querySelector("source").media = !paused && !document.hidden && visible.has(picture) ? "all" : "not all";
    }
    button.textContent = paused ? "Play section animations" : "Pause section animations";
  }
  button.hidden = false;
  button.addEventListener("click", () => { paused = !paused; sync(); });
  motion.addEventListener("change", () => { paused = motion.matches; sync(); });
  document.addEventListener("visibilitychange", sync);
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      }
      sync();
    });
    pictures.forEach(picture => observer.observe(picture));
  }
  sync();
})();
