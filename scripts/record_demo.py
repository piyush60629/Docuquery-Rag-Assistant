"""Record a demo GIF of the Streamlit app: load the demo documents, ask a question, open the cited
sources, then ask something the documents do not answer.

    streamlit run app.py &                       # with GEMINI_API_KEY (and GROQ_API_KEY) set
    python scripts/record_demo.py --out docs/demo.gif

Needs: pip install playwright && playwright install chromium, and ffmpeg. The GitHub workflow
"Record demo" runs exactly this with the repository secrets and commits the GIF.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1280, "height": 860}
QUESTIONS = [
    "What caused the out of memory error in the Databricks job, and how was it fixed?",
    "Who is the CEO of the retail company?",
]


def pause(page: Page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def type_slowly(page: Page, locator, text: str) -> None:
    locator.click()
    for char in text:
        page.keyboard.type(char)
        page.wait_for_timeout(25)


def follow_until(page: Page, condition, timeout_s: int) -> None:
    """Keep scrolling down (the answer grows the page) until condition() is true."""

    deadline = time.monotonic() + timeout_s
    while not condition():
        error = page.locator('[data-testid="stAlertContentError"]')
        if error.count():
            raise RuntimeError(f"The app showed an error: {error.first.inner_text()[:300]}")
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for the app")
        page.mouse.wheel(0, 300)
        pause(page, 0.8)


def record(url: str, video_dir: Path, timeout_s: int) -> Path:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir), record_video_size=VIEWPORT)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.get_by_text("DocuQuery").first.wait_for(timeout=120_000)
        page.mouse.move(VIEWPORT["width"] * 0.6, VIEWPORT["height"] * 0.6)
        pause(page, 2)

        page.get_by_role("button", name="Load included demo documents").click()
        page.get_by_text("Documents are ready").first.wait_for(timeout=timeout_s * 1000)
        pause(page, 2)

        for number, question in enumerate(QUESTIONS, start=1):
            box = page.get_by_placeholder("Ask a question about the uploaded documents")
            type_slowly(page, box, question)
            pause(page, 0.6)
            page.keyboard.press("Enter")
            expanders = page.get_by_text("Retrieved source passages")
            follow_until(page, lambda: expanders.count() >= number and not page.get_by_text("Searching the documents").count(),
                         timeout_s)
            pause(page, 3)
            if number == 1:  # show the cited passages once
                expanders.nth(0).click()
                pause(page, 1)
                for _ in range(6):
                    page.mouse.wheel(0, 120)
                    pause(page, 0.4)
                pause(page, 3)
                expanders.nth(0).click()
                pause(page, 1)
            page.mouse.wheel(0, 2000)
            pause(page, 2.5)

        video = Path(page.video.path())
        context.close()
        browser.close()
        return video


def to_gif(video: Path, out: Path, speed: float, width: int, fps: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", f"{filters},palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(palette),
                        "-lavfi", f"{filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        str(out)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=880)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        video = record(args.url, Path(tmp), args.timeout)
        to_gif(video, args.out, args.speed, args.width, args.fps)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1_000_000:.1f} MB) in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
