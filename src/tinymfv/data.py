"""Dataset loading. Reads from local `data/` if available, else falls back to
HuggingFace `wassname/tiny-mcf-vignettes`."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "wassname/tiny-mcf-vignettes"


def load_vignettes(name: str = "") -> list[dict]:
    """Load rewritten vignettes for a config.

    `name=''` -> clifford (132 Clifford vignettes); `name='scifi'` -> 51 sci-fi.
    Tries local `data/vignettes[_<name>]_rewritten.jsonl` first; falls back to
    the HuggingFace dataset (https://huggingface.co/datasets/wassname/tiny-mcf-vignettes)."""
    suf = f"_{name}" if name else ""
    p = ROOT / "data" / f"vignettes{suf}_rewritten.jsonl"
    if p.exists():
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    from datasets import load_dataset
    cfg = name or "clifford"
    return list(load_dataset(HF_REPO, cfg, split="train"))
