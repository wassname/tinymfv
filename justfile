# smoke test: download + 5-item rewrite + 5-item eval
smoke:
    uv run python scripts/01_download.py
    uv run python scripts/02_rewrite.py --limit 5
    uv run python scripts/03_eval.py --model Qwen/Qwen3-0.6B --limit 5 2>&1 | tee /tmp/tinymcf_smoke.log

# full rewrite via OpenRouter (one-time, cached on disc)
rewrite:
    uv run python scripts/02_rewrite.py --model openai/gpt-4o-mini

# eval a checkpoint: just eval Qwen/Qwen3-0.6B step_500
eval model tag="":
    uv run python scripts/03_eval.py --model {{model}} --tag {{tag}}
