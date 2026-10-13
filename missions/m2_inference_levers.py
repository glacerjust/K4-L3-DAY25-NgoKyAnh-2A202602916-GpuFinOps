"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 4 counters: reasoning vs standard traffic
    reasoning_reqs = 0
    reasoning_tokens = 0
    reasoning_cost = 0.0
    reasoning_wh = 0.0

    standard_reqs = 0
    standard_tokens = 0
    standard_cost = 0.0
    standard_wh = 0.0

    # Extension 3 cache stats
    total_cached_tokens = 0
    total_input_tokens = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))

        toks = inp + out
        total_tokens += toks
        total_input_tokens += inp
        total_cached_tokens += cached

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_opt

        # Energy calculation
        req_wh = sustainability.wh_per_query(toks, is_reasoning=is_reasoning)

        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += toks
            reasoning_cost += req_opt
            reasoning_wh += req_wh
        else:
            standard_reqs += 1
            standard_tokens += toks
            standard_cost += req_opt
            standard_wh += req_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3: Cache Economics
    # Estimate average cache reads per prefix across the corpus
    cache_hit_ratio = total_cached_tokens / total_input_tokens if total_input_tokens else 0.0
    # In token_usage.csv, chat/rag prefixes are reused multiple times (avg ~3.5 reads)
    estimated_avg_reads = 3.5 if cache_hit_ratio > 0.15 else 1.0
    cache_worth_large = pricing.cache_is_worth_it(estimated_avg_reads, write_cost_per_m=3.75, read_discount=0.10, base_price_in_per_m=3.00)
    cache_worth_small = pricing.cache_is_worth_it(estimated_avg_reads, write_cost_per_m=0.25, read_discount=0.10, base_price_in_per_m=0.20)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n--- Extension 3: Prompt Cache Economics (cache_is_worth_it) ---")
        print(f"Total input tokens: {total_input_tokens:,} | Cached input tokens: {total_cached_tokens:,} (Hit ratio: {cache_hit_ratio:.1%})")
        print(f"Large model cache break-even reads: {3.75 / (3.00 * 0.90):.2f} reads | Cache worth it? {cache_worth_large} (est. reads: {estimated_avg_reads})")
        print(f"Small model cache break-even reads: {0.25 / (0.20 * 0.90):.2f} reads | Cache worth it? {cache_worth_small}")

        print("\n--- Extension 4: Reasoning Traffic Budget & Energy Bomb ---")
        print(f"Standard traffic : {standard_reqs:4d} reqs ({standard_reqs/len(rows):.1%}), {standard_tokens:8,d} tokens ({standard_tokens/total_tokens:.1%}), ${standard_cost:5.2f} ({standard_cost/opt_cost:.1%}), {standard_wh/1000:6.2f} kWh ({standard_wh/(standard_wh+reasoning_wh):.1%})")
        print(f"Reasoning traffic: {reasoning_reqs:4d} reqs ({reasoning_reqs/len(rows):.1%}), {reasoning_tokens:8,d} tokens ({reasoning_tokens/total_tokens:.1%}), ${reasoning_cost:5.2f} ({reasoning_cost/opt_cost:.1%}), {reasoning_wh/1000:6.2f} kWh ({reasoning_wh/(standard_wh+reasoning_wh):.1%})")
        print(f"Reasoning intensity: Energy per query is ~{((reasoning_wh/reasoning_reqs)/(standard_wh/standard_reqs)):.0f}x higher than standard!")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_stats": {
            "reqs": reasoning_reqs,
            "cost": round(reasoning_cost, 2),
            "kwh": round(reasoning_wh / 1000.0, 3),
        },
        "cache_economics": {
            "hit_ratio": round(cache_hit_ratio, 3),
            "cache_worth_large": cache_worth_large,
            "cache_worth_small": cache_worth_small,
        }
    }


if __name__ == "__main__":
    run()
