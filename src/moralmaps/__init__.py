"""moralmaps: moral and value instruments for LLMs.

One answer-token reader, two reducer families:

- nominal MFV vignettes: answer = foundation category; `evaluate` reports a
  7-way profile plus label-match metrics.
- ordinal Likert questionnaires: answer = scale point; `administer` reports
  E, C, agree-vs-disagree log-odds, entropy, and pmass diagnostics.

High-level usage:

    from moralmaps import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").cuda()
    rep = evaluate(model, tok, name="classic")
    print(rep["table"])         # per-foundation
    print(rep["top1_acc"])      # argmax accuracy vs label
    print(rep["mean_nll_T"])    # temperature-scaled soft NLL vs label dist

Lower-level: see `guided_rollout_forced_choice` in `moralmaps.guided`.
"""
from .data import load_vignettes, load_all_vignettes, CONFIGS, ConfigName
from .eval import evaluate, CONDITIONS, EvalResult, EvalRow, EvalInfo
from .guided import guided_rollout_forced_choice, _DEFAULT_FORCED_FOUNDATIONS
from .instrument import Instrument, InstrItem, per_item_categorical, reduce_nominal, reduce_ordinal
from .instruments import get as get_instrument, INSTRUMENTS, build_instrument
from .read import read_items, resolve_answer_ids, build_user_content
from .readouts import expected_score, logit_contrast, logodds_agree, entropy
from .metrics import gated_selectivity, si_flips, clr_per_row, OFF_WEIGHT
from .administer import administer


def __getattr__(name: str):
    # `maps` pulls matplotlib; load it lazily so `import moralmaps` stays headless and fast for the
    # numeric-only consumers (steering-lite). `moralmaps.maps.plot_*` still works -- first access
    # triggers the import here.
    if name == "maps":
        import importlib
        return importlib.import_module(".maps", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The front door. Other symbols above stay importable (e.g. read_items for item subsets,
# per_item_categorical, build_instrument) but are helper internals, kept out of `import *`.
__all__ = [
    # entrypoints
    "evaluate", "administer", "get_instrument", "read_items",
    # ordinal readouts (pure functions of the raw answer-token logprobs)
    "expected_score", "logit_contrast", "logodds_agree", "entropy",
    # headline steering metrics (shared canonical defs; imported by steering-lite + j-steer)
    "gated_selectivity", "si_flips", "clr_per_row", "OFF_WEIGHT",
    # types consumers build / subset
    "Instrument", "InstrItem", "EvalResult", "EvalRow", "EvalInfo", "reduce_nominal", "reduce_ordinal",
    # data API
    "load_vignettes", "load_all_vignettes", "CONFIGS", "ConfigName", "CONDITIONS",
    # lower-level rollout + lazy plotting
    "guided_rollout_forced_choice", "maps",
]
