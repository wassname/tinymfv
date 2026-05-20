"""tinymfv: tiny moral-foundations vignettes eval.

Forced-choice 7-way scoring on Clifford 2015 vignettes (classic) +
paraphrase configs (scifi, ai-actor). Default condition is
`other_violate` (the canonical Clifford framing); `self_violate` is
available as an opt-in ablation. Each row internally does a fwd + rev
enum-order pass for position-bias debias (inside guided_rollout).

High-level usage:

    from tinymfv import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()
    rep = evaluate(model, tok, name="classic")
    print(rep["table"])         # per-foundation
    print(rep["top1_acc"])      # argmax accuracy vs label
    print(rep["mean_js"])       # JS divergence vs label dist (in nats)

Lower-level: see `guided_rollout_forced_choice` in `tinymfv.guided`.
"""
from .data import load_vignettes, load_all_vignettes, CONFIGS, ConfigName
from .eval import evaluate, CONDITIONS
from .guided import guided_rollout_forced_choice, _DEFAULT_FORCED_FOUNDATIONS

__all__ = [
    "CONDITIONS", "CONFIGS", "ConfigName",
    "load_vignettes", "load_all_vignettes",
    "evaluate",
    "guided_rollout_forced_choice", "_DEFAULT_FORCED_FOUNDATIONS",
]

