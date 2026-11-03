"""Capture dashboard screenshots for the README (needs a running dashboard).

Usage: ``make screenshots`` (starts the dashboard on the demo DB) or
``python scripts/screenshots.py --url http://localhost:8501``.
Requires ``pip install -e ".[docs]"`` and an installed Google Chrome
(or ``playwright install chromium`` and ``--channel chromium``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TABS = {
    "live": "Try it live",
    "overview": "Overview",
    "scatter": "Quality · cost · latency",
    "heatmap": "By category",
    "drift": "Drift",
    "drilldown": "Drill-down",
    "history": "Run history",
}


def _wait_ready(page: Page) -> None:
    page.wait_for_selector("text=LLM Model Comparison", timeout=60_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000
    )
    page.wait_for_timeout(1500)


def main() -> None:
    """Open each tab and save a full-page PNG."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        page.goto(args.url)
        _wait_ready(page)
        for name, label in TABS.items():
            page.get_by_role("tab", name=label, exact=True).click()
            page.wait_for_timeout(2500)
            if name == "drilldown":
                page.get_by_text("Failures only").click()
                page.wait_for_timeout(2000)
            page.screenshot(path=str(out / f"{name}.png"), full_page=False)
            print(f"saved {out / name}.png")
        browser.close()


if __name__ == "__main__":
    main()
