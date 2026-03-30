"""
Render the Three.js explainer scene frame-by-frame using Playwright.

Usage:
    python video/render.py                    # full render
    python video/render.py --act 3            # render only Act 3
    python video/render.py --preview          # 720p, every 4th frame
"""

import argparse
import asyncio
import subprocess
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

WIDTH = 1920
HEIGHT = 1080
FPS = 60
TOTAL_FRAMES = 10800

ACTS = {
    1: (0, 2099),
    2: (2100, 3899),
    3: (3900, 6899),
    4: (6900, 8699),
    5: (8700, 10079),
    6: (10080, 10799),
}

OUTPUT_DIR = Path("video/frames")
FINAL_DIR = Path("video/output")


async def render(act=None, preview=False):
    w = WIDTH // 2 if preview else WIDTH
    h = HEIGHT // 2 if preview else HEIGHT
    step = 4 if preview else 1

    if act:
        start, end = ACTS[act]
    else:
        start, end = 0, TOTAL_FRAMES - 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": w, "height": h})

        # Serve the HTML file
        scene_path = Path("video/scene.html").resolve()
        await page.goto(f"file://{scene_path}")

        # Wait for Three.js to initialize
        await page.wait_for_function("window.renderFrame !== undefined")

        # Set resolution
        await page.evaluate(f"renderer.setSize({w}, {h})")

        n_frames = (end - start) // step + 1
        print(f"Rendering frames {start}-{end} (step={step}, {n_frames} frames)")

        for i, frame in enumerate(range(start, end + 1, step)):
            await page.evaluate(f"window.renderFrame({frame})")
            # Use sequential index for filenames so ffmpeg can read them
            fname = OUTPUT_DIR / f"frame_{i:06d}.png"
            canvas = page.locator("canvas")
            await canvas.screenshot(path=str(fname))

            if i % 100 == 0:
                pct = i / n_frames * 100
                print(f"  Frame {frame}/{end} ({pct:.0f}%)")

        await browser.close()

    print(f"Captured {n_frames} frames in {OUTPUT_DIR}")

    # Stitch with ffmpeg
    effective_fps = FPS // step
    output_name = f"topofeatures_act{act}_silent.mp4" if act else "topofeatures_silent.mp4"
    output_path = FINAL_DIR / output_name

    # Build ffmpeg input — frames are sequentially numbered 000000, 000001, ...
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(effective_fps),
        "-i", str(OUTPUT_DIR / "frame_%06d.png"),
        "-frames:v", str(n_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    print(f"Stitching → {output_path}")
    subprocess.run(ffmpeg_cmd, check=True)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Render only this act")
    parser.add_argument("--preview", action="store_true",
                        help="720p, every 4th frame (fast preview)")
    args = parser.parse_args()
    asyncio.run(render(act=args.act, preview=args.preview))


if __name__ == "__main__":
    main()
