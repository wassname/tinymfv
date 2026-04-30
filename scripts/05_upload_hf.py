"""Upload tiny-mfv to HuggingFace Hub as a dataset with three configs.

Creates / updates: wassname/tiny-mfv

- config `clifford`: 132 vignettes from Clifford et al. (2015), Wrong ratings are human Likert.
- config `scifi`:    132 hand-written sci-fi/fantasy vignettes covering the same foundations.
- config `airisk`:   132 hand-written AI-risk vignettes (deception, sandbagging, principal subversion, etc.)
                     mapped onto the MFT foundation taxonomy.

Each config has two splits:
- `other_violate` -- verbatim 3rd-person source (no LLM call).
- `self_violate`  -- 1st-person rewrite. For airisk this preserves the AI-as-actor
                     framing ("You, an AI X bot, ..."); for clifford/scifi it's a plain
                     "You ..." 1st-person shift.
"""
from __future__ import annotations
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "wassname/tiny-mfv"
ROOT = Path(__file__).resolve().parents[1]

CONFIGS = ["clifford", "scifi", "airisk"]
SPLITS = ["other_violate", "self_violate"]


def local_jsonl(cfg: str, split: str) -> Path:
    suf = "" if cfg == "clifford" else f"_{cfg}"
    return ROOT / "data" / f"vignettes{suf}_{split}.jsonl"


def local_csv(cfg: str) -> Path:
    suf = "" if cfg == "clifford" else f"_{cfg}"
    return ROOT / "data" / f"vignettes{suf}.csv"


def hf_jsonl(cfg: str, split: str) -> str:
    return f"{cfg}/vignettes_{split}.jsonl"


def hf_csv(cfg: str) -> str:
    return f"{cfg}/vignettes.csv"


def yaml_configs() -> str:
    lines = ["configs:"]
    for cfg in CONFIGS:
        lines.append(f"- config_name: {cfg}")
        lines.append("  data_files:")
        for split in SPLITS:
            lines.append(f"    - split: {split}")
            lines.append(f"      path: {hf_jsonl(cfg, split)}")
    return "\n".join(lines)


README = f"""---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- moral-foundations
- alignment
- evaluation
- ai-safety
pretty_name: Tiny Moral-Foundations Vignettes
size_categories:
- n<1K
{yaml_configs()}
---

# tiny-mfv

Fast inner-loop moral-foundations probe for tracking LLM checkpoints under steering. Three configs of 132 vignettes each, covering the same MFT taxonomy (Care, Fairness, Loyalty, Authority, Sanctity, Liberty, Social Norms).

## Configs

- **clifford**: 132 vignettes from Clifford et al. (2015) "Moral Foundations Vignettes". `wrong` is the human Likert mean (1-5).
- **scifi**: 132 hand-written sci-fi/fantasy vignettes covering the same foundations. Genre-clean cues, no real-world ethnicity/religion confounds.
- **airisk**: 132 hand-written AI-risk vignettes (deception, sandbagging, principal subversion, manipulation, surveillance) mapped onto the MFT taxonomy.

## Splits (per config)

- `other_violate` — verbatim 3rd-person source text. No LLM call. For clifford this means the verbatim text is in every LLM's training set, which is fine for tracking deltas across checkpoints (the offset is constant).
- `self_violate`  — 1st-person rewrite of the same scenario. For clifford and scifi this is a plain `"You ..."` shift. For airisk the principal IS the AI, so the rewrite preserves the AI-as-actor framing as `"You, an AI X bot, ..."` (a naive `"You ..."` template silently swaps the actor archetype to human; verified by `06_consistency.py`).

Both splits have schema: `{{id, foundation, foundation_coarse, wrong, text}}`.

## Eval

Two scalars per checkpoint:

- `wrongness = mean(s_other_violate)` over foundations — does steering shift moral-rating magnitude?
- `gap = mean(s_other_violate - s_self_violate)` over foundations — does steering shift perspective bias (harshness on others vs self)?

Per-vignette score `s ∈ [-1, +1]` from a JSON-bool dual-frame probe (`is_wrong` true vs `is_acceptable` false), which cancels JSON-true prior. Full eval: see [tiny-mfv on GitHub](https://github.com/wassname/tiny-mcf-vignettes).

## Citation

Clifford, S., Iyengar, V., Cabeza, R., & Sinnott-Armstrong, W. (2015). *Moral Foundations Vignettes: A standardized stimulus database of scenarios based on moral foundations theory.* Behavior Research Methods, 47(4), 1178-1198.

Source vignettes: https://github.com/peterkirgis/llm-moral-foundations
"""


def main():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"repo: {REPO_ID}")

    files: list[tuple[Path, str]] = []
    for cfg in CONFIGS:
        files.append((local_csv(cfg), hf_csv(cfg)))
        for split in SPLITS:
            files.append((local_jsonl(cfg, split), hf_jsonl(cfg, split)))

    for src, dst in files:
        if not src.exists():
            print(f"SKIP missing {src}")
            continue
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=REPO_ID, repo_type="dataset")
        print(f"uploaded {dst}")

    readme_p = ROOT / "_HF_README.md"
    readme_p.write_text(README)
    api.upload_file(path_or_fileobj=str(readme_p), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="dataset")
    readme_p.unlink()
    print(f"uploaded README.md")
    print(f"\nhttps://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
