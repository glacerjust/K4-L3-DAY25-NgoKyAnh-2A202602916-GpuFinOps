"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Extension 5 carbon tracking for interruptible jobs
    carbon_baseline_g = 0.0
    carbon_clean_g = 0.0
    kwh_total = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        jdays = int(num(j["days"]))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Extension 1: pass gpu_type and job_days
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=jdays)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        # Extension 5: calculate carbon for interruptible workloads
        if interruptible:
            watts = num(c["watts"])
            kwh = (gpu_hours * watts) / 1000.0
            kwh_total += kwh
            carbon_baseline_g += sustainability.carbon_g(kwh * 1000.0, "us-east-1")
            carbon_clean_g += sustainability.carbon_g(kwh * 1000.0, "europe-north1")

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    carbon_saved_kg = (carbon_baseline_g - carbon_clean_g) / 1000.0
    carbon_saved_pct = (carbon_saved_kg / (carbon_baseline_g / 1000.0) * 100.0) if carbon_baseline_g else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n--- Extension 5: Carbon-Aware Scheduling for Interruptible Workloads ---")
        print(f"Interruptible compute: {kwh_total:,.1f} kWh / month")
        print(f"Default region (us-east-1, 380 gCO2/kWh)    : {carbon_baseline_g/1000.0:,.1f} kg CO2e")
        print(f"Clean region   (europe-north1, 30 gCO2/kWh) : {carbon_clean_g/1000.0:,.1f} kg CO2e")
        print(f"-> Potential carbon cut: {carbon_saved_kg:,.1f} kg CO2e ({carbon_saved_pct:.1f}% reduction!)")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_aware_savings_kg": round(carbon_saved_kg, 1),
    }


if __name__ == "__main__":
    run()
