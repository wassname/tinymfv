"""tinymfv: tiny moral-foundations vignettes eval.

High-level usage:

    from tinymfv import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").cuda()
    report = evaluate(model, tok, name="scifi")
    print(report["table"])  # tabulated per-foundation
    print(report["score"])  # headline align_other(real)

Lower-level: see `format_prompts`, `score_prompts`, `analyse`.
"""
from .core import (
    CONDITIONS,
    FRAMES,
    format_prompt,
    format_prompts,
    bool_token_ids,
    score_prompts,
    analyse,
)
from .data import load_vignettes
from .eval import evaluate

__all__ = [
    "CONDITIONS", "FRAMES",
    "format_prompt", "format_prompts", "bool_token_ids",
    "score_prompts", "analyse", "load_vignettes", "evaluate",
]
