"""Download Clifford-style moral foundations vignettes CSV.

Source: https://github.com/peterkirgis/llm-moral-foundations (peterkirgis fork
of MFV/Clifford et al. 2015). 132 short third-person scenarios labeled by
foundation, with mean Wrong rating in [0, 4].
"""
from __future__ import annotations
import sys
from pathlib import Path

import httpx
from loguru import logger

URL = "https://raw.githubusercontent.com/peterkirgis/llm-moral-foundations/main/data/survey/vignettes.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "vignettes.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"GET {URL}")
    r = httpx.get(URL, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    n = sum(1 for _ in OUT.read_text().splitlines()) - 1
    logger.info(f"wrote {OUT} ({n} rows)")
    if n < 100:
        logger.error(f"expected ~132 rows, got {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
