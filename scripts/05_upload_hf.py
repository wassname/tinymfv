"""Upload tiny-mcf-vignettes to HuggingFace Hub as a dataset with two configs.

Creates / updates: wassname/tiny-mcf-vignettes
- config 'clifford': 132 vignettes from Clifford et al. (2015), rewritten 4 ways
- config 'scifi': 51 hand-written sci-fi/fantasy vignettes, rewritten 4 ways

Each row of the rewritten files has: id, foundation, foundation_coarse, wrong,
other_violate, other_uphold, self_violate, self_uphold.
"""
from __future__ import annotations
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "wassname/tiny-mcf-vignettes"
ROOT = Path(__file__).resolve().parents[1]


README = """---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- moral-foundations
- evaluation
- alignment
pretty_name: Tiny Moral-Foundations Vignettes
size_categories:
- n<1K
configs:
- config_name: clifford
  data_files:
    - split: train
      path: clifford/vignettes_rewritten.jsonl
- config_name: scifi
  data_files:
    - split: train
      path: scifi/vignettes_scifi_rewritten.jsonl
---

# tiny-mcf-vignettes

Fast inner-loop moral-foundations probe for steering LLM checkpoints. Two configs:

- **clifford**: 132 vignettes from Clifford et al. (2015) "Moral Foundations Vignettes" covering Care, Fairness, Loyalty, Authority, Sanctity, Liberty, plus a Social Norms negative control. Wrong ratings are human Likert (5-point).
- **scifi**: 51 hand-written sci-fi/fantasy vignettes covering the same 7 foundations. Genre-clean foundation cues (no real-world ethnicity / religion confounds). Judge-vs-original ceiling 94.1% (vs Clifford 84.9%). Wrong ratings are author-assigned.

Each row in the `rewritten` split has 4 conditions:

- `other_violate`: verbatim original (third-person violation).
- `other_uphold`: LLM-rewritten third-person upholding the foundation.
- `self_violate`: LLM-rewritten first-person violation.
- `self_uphold`: LLM-rewritten first-person upholding.

Used for the bias-cancelled dual Y/N probe in
[wassname/tiny-mcf-vignettes (GitHub)](https://github.com/wassname/tiny-mcf-vignettes).

## Citation

Clifford, S., Iyengar, V., Cabeza, R., & Sinnott-Armstrong, W. (2015).
*Moral Foundations Vignettes: A standardized stimulus database of scenarios
based on moral foundations theory.* Behavior Research Methods, 47(4), 1178-1198.

Source vignettes: https://github.com/peterkirgis/llm-moral-foundations
"""


def main():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"repo: {REPO_ID}")

    files = [
        ("data/vignettes.csv",                    "clifford/vignettes.csv"),
        ("data/vignettes_rewritten.jsonl",        "clifford/vignettes_rewritten.jsonl"),
        ("data/vignettes_scifi.csv",              "scifi/vignettes_scifi.csv"),
        ("data/vignettes_scifi_rewritten.jsonl",  "scifi/vignettes_scifi_rewritten.jsonl"),
    ]
    for src, dst in files:
        p = ROOT / src
        if not p.exists():
            print(f"SKIP missing {p}")
            continue
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst,
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
