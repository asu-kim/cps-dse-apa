#!/usr/bin/env python3
import json, sys, os
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Minisat22
import time

# Sections in the exact order requested
ORDERED_SECTIONS = [
    "proposed_approach",
    "area_payload",
    "cost_area",
    "payload_cost",
    "simulated_annealing",
    "bayesian",
    "pg_dse",
    "random_search",
    "genetic_algorithm",
    "discrete",
    "lengler",
    "portfolio",
]

# EPS = 1e-6

def fmt_ident(rec):
    return f"{rec.get('type','?')} {rec.get('body','?')}/{rec.get('motor','?')}/{rec.get('battery','?')} x{rec.get('quantity','?')}"

def load_data(path):
    # default to uploaded path if not provided
    if not os.path.exists(path) and os.path.exists("/mnt/data/results.json"):
        path = "/mnt/data/results.json"
    with open(path, "r") as f:
        return json.load(f)

def main(path="results.json"):
    t_total_start = time.perf_counter() # --- start total timer ---
    data = load_data(path)
    inputs = data.get("inputs", {})
    budget = float(inputs.get("budget", 0.0))
    farm_size = float(inputs.get("farm_size", 0.0))

    print(f"Inputs: budget={budget}, farm_size={farm_size}\n")

    vpool = IDPool()
    cnf = CNF()

    overall_ok = True
    saw_any_configs = False
    section_summary = {}

    for sec in ORDERED_SECTIONS:
        arr = data.get(sec, None)

        if arr is None:
            print(f"=== {sec} ===")
            print("  — missing from JSON —\n")
            section_summary[sec] = {"present": False, "total": 0, "ok": 0, "viol": 0}
            continue

        if not isinstance(arr, list):
            print(f"=== {sec} ===")
            print("  — present but not a list; skipping —\n")
            section_summary[sec] = {"present": True, "total": 0, "ok": 0, "viol": 0}
            continue

        print(f"=== {sec} (n={len(arr)}) ===")
        saw_any_configs = saw_any_configs or len(arr) > 0

        lits = []
        ok_count = 0
        viol_count = 0

        if len(arr) == 0:
            print("  — present but empty —\n")
            section_summary[sec] = {"present": True, "total": 0, "ok": 0, "viol": 0}
            continue

        for i, rec in enumerate(arr):
            var = vpool.id(f"{sec}[{i}]_ok")
            ident = fmt_ident(rec)
            tc = float(rec.get("total_cost", 0.0))
            tv = float(rec.get("total_coverage", 0.0))

            cost_ok = (tc <= budget)
            cov_ok  = (tv >= farm_size)
            ok = cost_ok and cov_ok

            # Encode as a unit clause; aggregation is enforced with a cardinality constraint.
            cnf.append([var if ok else -var])
            lits.append(var)

            if ok:
                ok_count += 1
                print(f"  ✓ {ident}  (total_cost={tc}, total_coverage={tv})")
            else:
                viol_count += 1
                overall_ok = False
                print(f"  ✗ {ident}  (total_cost={tc}, total_coverage={tv})")
                if not cost_ok:
                    print(f"     - total_cost {tc} > budget {budget}")
                if not cov_ok:
                    print(f"     - total_coverage {tv} < farm_size {farm_size}")

        # Require ALL configs in this section to satisfy both totals
        if lits:
            cnf.extend(CardEnc.atleast(lits=lits, bound=len(lits), vpool=vpool).clauses)

        section_summary[sec] = {"present": True, "total": len(arr), "ok": ok_count, "viol": viol_count}
        print()

    # Solve once for the whole instance
    t_solve_start = time.perf_counter()  #SAT solver timing only
    with Minisat22(bootstrap_with=cnf.clauses) as m:
        sat = m.solve()
    t_solve_end = time.perf_counter()

    print("=== SUMMARY ===")
    for sec in ORDERED_SECTIONS:
        s = section_summary.get(sec, {"present": False, "total": 0, "ok": 0, "viol": 0})
        if not s["present"]:
            print(f"{sec}: MISSING")
        else:
            print(f"{sec}: total={s['total']}, ok={s['ok']}, violations={s['viol']}")

    if not saw_any_configs:
        print("\nNo configurations found in JSON.")

        t_total_end = time.perf_counter() # --- end total timer before exiting ---

        print(f"\n[Timing] SAT solve time: {t_solve_end - t_solve_start:.6f} s")
        print(f"[Timing] Total runtime: {t_total_end - t_total_start:.6f} s")

        sys.exit(2)

    if sat and overall_ok:
        print("\nALL SECTIONS SATISFY: total_cost ≤ budget AND total_coverage ≥ farm_size")

        t_total_end = time.perf_counter()
        print(f"\n[Timing] SAT solve time: {t_solve_end - t_solve_start:.6f} s")
        print(f"[Timing] Total runtime: {t_total_end - t_total_start:.6f} s")

        sys.exit(0)
    else:
        print("\nVIOLATIONS DETECTED (SAT solver returned UNSAT for the aggregated constraints)")

        t_total_end = time.perf_counter()
        print(f"\n[Timing] SAT solve time: {t_solve_end - t_solve_start:.6f} s")
        print(f"[Timing] Total runtime: {t_total_end - t_total_start:.6f} s")

        sys.exit(1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SAT-based verification of totals per section in results.json.")
    ap.add_argument("results", nargs="?", default="results.json", help="Path to results.json")
    args = ap.parse_args()
    main(args.results)
