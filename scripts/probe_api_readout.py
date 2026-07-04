"""UAT for the sampling readout (read_api.read_items_sampled) vs the logprob readout (read.read_items).

Three checks on the same first-N mfq2 items (identical build_user_content stimulus for both paths):

  1. UNBIASEDNESS (same model, no API): draw N Monte-Carlo samples from the LOCAL model's own logprob
     p and reduce to E. E_mc must match the exact logprob E within sampling error -- proves the
     frequency -> E estimator + reduce plumbing is unbiased (the statistical core of read_api).
  2. LIVE API: run read_items_sampled on a real OpenRouter model (no logprobs). Show per-item E,
     parse rate (the sampling coherence gate), and entropy. Sanity: parse rate ~1, E in [1,5].
  3. COMPARABILITY: tabulate local logprob-E next to API sampled-E on the same items. Different
     models, so they differ by genuine opinion, but both land on the same [1,5] E scale -- which is
     the whole point: a logprob-less frontier model drops onto the same map.

  uv run python scripts/probe_api_readout.py --api-model meta-llama/llama-3.1-8b-instruct
"""
from __future__ import annotations

import argparse

import dotenv
import numpy as np
import torch

dotenv.load_dotenv()  # OPENROUTER_API_KEY from .env for read_api
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

import tinymfv as T
from tinymfv.instruments import get as get_instrument
from tinymfv.read import read_items, resolve_answer_ids, build_user_content
from tinymfv.read_api import read_items_sampled
from tinymfv.readouts import expected_score, entropy

W = np.arange(1, 6, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="mfq2")
    ap.add_argument("--api-model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--local-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    instr = get_instrument(args.instr)
    items = [it for it in instr.items if it.frame == "forward"][: args.n_items]
    logger.info(f"{len(items)} {args.instr} forward items; shared stimulus for item 0:\n"
                f"{build_user_content(instr, items[0])!r}")

    # --- local logprob readout ---
    tok = AutoTokenizer.from_pretrained(args.local_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.local_model, dtype=torch.bfloat16).to(args.device).eval()
    answer_ids = resolve_answer_ids(tok, instr.answer_space)
    rows_local = read_items(model, tok, instr, items, answer_ids,
                            max_think_tokens=args.max_think_tokens, batch_size=8,
                            n_samples=1, temperature=0.0, verbose_first=True)
    p_local = {r["id"]: np.asarray(r["p"], float) for r in rows_local}

    # --- (1) unbiasedness: MC-sample the local model's own p, reduce to E ---
    rng = np.random.default_rng(0)
    N_MC = 2000
    e_logprob = {i: expected_score(p, 5) for i, p in p_local.items()}
    e_mc = {i: float(rng.choice(5, size=N_MC, p=p).mean() + 1) for i, p in p_local.items()}
    max_mc_gap = max(abs(e_mc[i] - e_logprob[i]) for i in p_local)
    logger.info(f"(1) unbiasedness: max |E_mc(N={N_MC}) - E_logprob| = {max_mc_gap:.4f} "
                f"(should be < 0.05; MC of the model's own p reduces to the same E)")

    # --- (2)+(3) live API sampled readout ---
    rows_api = read_items_sampled(args.api_model, instr, items, n_samples=args.n_samples,
                                  temperature=args.temperature, verbose_first=True)

    table = []
    for it, ra in zip(items, rows_api):
        pl = p_local[it.id]
        pa = np.asarray(ra["p"], float)
        table.append([
            it.id, it.dimension, it.prompt[:34],
            f"{expected_score(pl, 5):.2f}", f"{e_mc[it.id]:.2f}",
            f"{expected_score(pa, 5):.2f}" if ra["n_parsed"] else "NaN",
            f"{ra['pmass_allowed']:.2f}", f"{ra['n_parsed']}/{ra['n_samples']}",
            f"{entropy(pa, 5):.2f}" if ra["n_parsed"] else "NaN",
        ])
    print("\n" + tabulate(table, headers=[
        "id", "dimension", "statement", "E_local(lp)", "E_mc", f"E_api({args.api_model.split('/')[-1]})",
        "api_pmass", "parsed", "api_H"], tablefmt="pipe"))
    print("\nSHOULD: E_local ~= E_mc (unbiased estimator); E_api on the same [1,5] scale (a "
          "logprob-less model on the same map); api_pmass near 1.0 = coherent (parse rate). "
          "E_api != E_local is genuine model disagreement, not method error.")


if __name__ == "__main__":
    main()
