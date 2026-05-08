"""Merge ai (grok-4-fast, post-hoc rescaled) labels into the main vignette files.

Reads `data/multilabel_<name>.jsonl` and merges the `ai_*` 
columns into `data/vignettes_<name>_{other,self}_violate.jsonl`. This prepares
the files so that `05_upload_hf.py` will upload the machine labels to HuggingFace.

Usage:
    python scripts/07a_merge_labels.py
"""
from __future__ import annotations
import json
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["classic", "scifi", "ai-actor"]
CONDITIONS = ["other_violate", "self_violate"]

def main() -> None:
    for name in NAMES:
        # Load the multilabel records
        ml_path = ROOT / "data" / f"multilabel_{name}.jsonl"
        
        if not ml_path.exists():
            logger.warning(f"missing {ml_path}, skipping config {name}")
            continue
            
        ml_lines = [json.loads(line) for line in ml_path.read_text().splitlines() if line.strip()]
        
        # We extract all ai_* keys
        # The multilabel script only runs on other_violate by default, but the labels apply
        # to the vignette ID as a whole.
        extra_by_id = {}
        for row in ml_lines:
            extra = {}
            for k, v in row.items():
                if k.startswith("ai_"):
                    extra[k] = v
            extra_by_id[row["id"]] = extra
            
        # Patch the vignette files.
        for cond in CONDITIONS:
            vig_path = ROOT / "data" / f"vignettes_{name}_{cond}.jsonl"
            if not vig_path.exists():
                logger.warning(f"missing {vig_path}, skipping")
                continue
                
            lines = vig_path.read_text().splitlines()
            out = []
            n_match = 0
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                
                extra = extra_by_id.get(rec["id"])
                if extra:
                    n_match += 1
                    for k, v in extra.items():
                        rec[k] = v
                
                out.append(json.dumps(rec))
                
            vig_path.write_text("\n".join(out) + "\n")
            logger.info(f"patched {vig_path.name}: {n_match} records merged with machine labels")

if __name__ == "__main__":
    main()
