import argparse, json
from .io import load_problem
from .ga import run_ga

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="inputs/inputs_to_load.xlsx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--time", type=int, default=600)
    args = ap.parse_args()

    data = load_problem(args.excel)
    res = run_ga(data,
                 pop_size=args.pop,
                 max_generations=args.gens,
                 time_limit_sec=args.time,
                 seed=args.seed)

    best_chrom, best_val = res["best"]
    print(json.dumps({
        "best_value": best_val,
        "generations": res["generations"],
        "stall": res["stall"],
        "best_routes": best_chrom
    }, indent=2))

if __name__ == "__main__":
    main()
