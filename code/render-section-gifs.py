"""Generate the section GIFs/stills: python3 code/render-section-gifs.py.

Optional authoring dependencies: Pillow, Playwright, and Playwright Chromium.
Serving or rebuilding the blog does not require these tools.
"""
import base64
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "sections"
OUTPUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_content('<canvas width="96" height="64"></canvas>')
    page.add_script_tag(path=str(ROOT / "code/section-gif-renderer.js"))
    colors = page.evaluate("window.SECTION_PALETTE")
    rgb = [int(c[i:i + 2], 16) for c in colors for i in (1, 3, 5)]
    palette = Image.new("P", (1, 1))
    palette.putpalette(rgb + rgb[:3] * (256 - len(colors)))
    for name in ("vision", "video", "language", "research"):
        frames = []
        for index in range(36):
            data = page.evaluate("""([name, t]) => {
                window.drawSection(name, t);
                return document.querySelector('canvas').toDataURL().split(',')[1];
            }""", [name, index / 36])
            frame = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        frames[9].save(OUTPUT / f"{name}.png")
        output = OUTPUT / f"{name}.gif"
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=100, loop=0, optimize=True, disposal=1)
        print(f"{name}: {output.stat().st_size:,} bytes")
    browser.close()
