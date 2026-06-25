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
from .instrument import Instrument, InstrItem, per_item_categorical
from .instruments import get as get_instrument, INSTRUMENTS, build_instrument
from .read import read_items, resolve_answer_ids, build_user_content
from .readouts import expected_score, logit_contrast, agree_logodds, entropy
from .administer import administer


def __getattr__(name: str):
    # `maps` pulls matplotlib; load it lazily so `import tinymfv` stays headless and fast for the
    # numeric-only consumers (steering-lite). `tinymfv.maps.plot_*` still works -- first access
    # triggers the import here.
    if name == "maps":
        import importlib
        return importlib.import_module(".maps", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The front door. Other symbols above stay importable (e.g. read_items for item subsets,
# per_item_categorical, build_instrument) but are plumbing, kept out of `import *`.
__all__ = [
    # entrypoints
    "evaluate", "administer", "get_instrument", "read_items",
    # ordinal readouts (pure functions of the raw answer-token logprobs)
    "expected_score", "logit_contrast", "agree_logodds", "entropy",
    # types consumers build / subset
    "Instrument", "InstrItem",
    # data API
    "load_vignettes", "load_all_vignettes", "CONFIGS", "ConfigName", "CONDITIONS",
    # lower-level rollout + lazy plotting
    "guided_rollout_forced_choice", "maps",
]

