# firefighting model by erdi dasdemir
# first successful run !!! March 28, 2023 - 3:35 pm
# successful run after all bugs are fixed !! March 29, 2023 - 17:00
# combinations mode is added June 15, 2023 - 17:00


# import required packages
import numpy as np
import pandas as pd
import openpyxl
import xlsxwriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import mip_setup as mip_setup
import mip_solve as mip_solve

from itertools import combinations
from datetime import datetime
import os
from random import sample


# read user inputs
user_inputs = mip_setup.UserInputsRead()
user_inputs.solution_method = user_inputs.parameters_df.loc["solution_method", "value"]
user_inputs.optimization_mode = user_inputs.parameters_df.loc["optimization_mode", "value"]
user_inputs.experiment_mode = user_inputs.parameters_df.loc["experiment_mode", "value"]

# modes
# single_run: runs MIP as a single optimization task
# combination_run: runs MIP in the combination mode (to evaluate the impact of quantity and location of initial fires)
# instance_generate: generate a new WUI scenario based case instance
if user_inputs.solution_method == "ga":

    import json
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt

    from ga.io import load_problem, load_ga_params
    from ga.ga import run_ga

    EXCEL_PATH = "inputs/inputs_to_load.xlsx"

    data = load_problem(EXCEL_PATH)
    gp = load_ga_params(EXCEL_PATH)

    from time import perf_counter

    t0 = perf_counter()

    res = run_ga(
        data,
        pop_size=gp["pop_size"],
        max_generations=gp["max_generations"],
        time_limit_sec=gp["time_limit_sec"],
        seed=gp["seed"],
        coverage_ratio=gp["coverage_ratio"],
        p_cross=gp["p_cross"],
        p_mut_swap=gp["p_mut_swap"],
        p_mut_add=gp["p_mut_add"],
        p_mut_remove=gp["p_mut_remove"],
        heuristic_seed_ratio=gp["heuristic_seed_ratio"],
        elite_ratio=gp["elite_ratio"],
        stall_ratio=gp["stall_ratio"],
        track_history = True,  # <-- make sure it's on

    )

    run_time = perf_counter() - t0
    res["run_time_sec"] = run_time

    best_chrom, best_val = res["best"]
    print(json.dumps({
        "best_value": best_val,
        "generations": res["generations"],
        "stall": res["stall"],
        "best_routes": best_chrom
    }, indent=2))

    # ---------- Visualization ----------
    hist = res["history"]
    if hist:
        # create unique run folder inside outputs/ga/
        run_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
        out_dir = os.path.join("outputs", "ga", f"ga_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        gens = [h["generation"] for h in hist]
        bests = [h["best"] for h in hist]
        means = [h["mean"] for h in hist]

        # 1) Convergence curves
        plt.figure()
        plt.plot(gens, bests, label="Best")
        plt.plot(gens, means, label="Mean")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (expected reward)")
        plt.title("GA Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
        plt.close()

        # 2) Population spread per generation
        plt.figure()
        for h in hist:
            x = [h["generation"]] * len(h["fitness"])
            plt.scatter(x, h["fitness"], alpha=0.5, s=10)
        plt.xlabel("Generation")
        plt.ylabel("Fitness (expected reward)")
        plt.title("Population Fitness per Generation")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "population_spread.png"), dpi=150)
        plt.close()
        print(f"[GA] Plots saved in {out_dir}")


        # 3) Per-generation scatter plots
        per_gen_dir = os.path.join(out_dir, "per_generation")
        os.makedirs(per_gen_dir, exist_ok=True)

        for h in hist:
            # if (h["generation"] + 1) % 5 != 0:  # keep every 5th
            #     continue
            gen_idx = h["generation"]  # 0,1,2,... (you can +1 for naming)
            plt.figure()
            # one dot per individual in this generation
            x = [gen_idx] * len(h["fitness"])
            plt.scatter(x, h["fitness"], alpha=0.7, s=12)
            plt.xlabel("Generation")
            plt.ylabel("Fitness (expected reward)")
            plt.title(f"Population Fitness — Generation {gen_idx + 1}")
            plt.grid(True, alpha=0.3)
            # save as generation_1.png, generation_2.png, ...
            plt.savefig(os.path.join(per_gen_dir, f"generation_{gen_idx + 1}.png"), dpi=150)
            plt.close()

        print(f"[GA] Per-generation plots saved in {per_gen_dir}")


        # ---------- Build Excel report ----------
        import pandas as pd

        # measure run time across the GA call (wrap your run_ga invocation with timers)
        # if you want precise timing:
        # from time import perf_counter
        # t0 = perf_counter()
        # ... run GA ...
        # run_time = perf_counter() - t0
        # Here we'll derive rough run time from history length if you didn't time it; better to time it.

        # Compute detailed results for the best chromosome
        from ga.model import evaluate_solution_details, expected_reward

        best_chrom, best_val = res["best"]

        details = evaluate_solution_details(data, best_chrom)
        scen_totals = details["per_scen_totals"]  # dict: {scenario_id: total}
        n_scen = len(scen_totals)

        # If you timed the GA externally, use that value; otherwise estimate 0 for placeholder
        try:
            run_time = res.get("run_time_sec", None)
        except Exception:
            run_time = None

        # 1) global_results sheet
        global_rows = []
        init_fires = [j for j in data.Jstar if data.nodes[j].state == 1]
        global_rows.append(("n_scenarios", n_scen))
        global_rows.append(("expected_collected_value", best_val))
        global_rows.append(("scenario_collected_value_results", str(scen_totals)))
        global_rows.append(("run_time", run_time if run_time is not None else ""))  # fill if you recorded it
        global_rows.append(("number_of_nodes", len(data.Jstar)))
        global_rows.append(("number_of_initial_fires", len(init_fires)))
        global_rows.append(("number_of_vehicles", data.n_vehicles))
        global_rows.append(("initial_fire_node_IDs", str(init_fires)))
        global_rows.append(("generations_executed", res["generations"]))
        global_rows.append(("ga_stop_reason", res.get("stop_reason", "")))
        global_rows.append(("best_makespan", res.get("best_makespan", None)))
        df_global = pd.DataFrame(global_rows, columns=["result_name", "results"])

        # 2) routes sheet (vehicle timeline)
        # columns: vehicle, step, node_id, node_type, arrival_time
        rows_routes = []
        for k, steps in details["timeline"].items():
            for step_idx, nid, ntype, arr in steps:
                rows_routes.append({
                    "vehicle": k,
                    "step": step_idx,
                    "node_id": nid,
                    "node_type": ntype,
                    "arrival_time": arr
                })
        df_routes = pd.DataFrame(rows_routes).sort_values(["vehicle", "step"]).reset_index(drop=True)

        # 3) node_details sheet
        # per scenario, for each node in J*, with arrival_time and drop_time from that scenario
        rows_nodes = []
        for scen in data.scenarios:
            dets = details["per_scen_details"][scen.id]
            for j in data.Jstar:
                d = dets[j]
                rows_nodes.append({
                    "scenario_id": scen.id,
                    "node_id": j,
                    "had_fire": d.had_fire,
                    "final_status": d.status,
                    "ts": d.ts if d.ts is not None else 0.0,
                    "tm": d.tm if d.tm is not None else 0.0,
                    "te": d.te if d.te is not None else 0.0,
                    "arrival_time": d.arrival_time if d.arrival_time is not None else 0.0,
                    "drop_time": d.drop_time if d.drop_time is not None else 0.0,
                    "collected_value": d.collected_value,
                })
        df_nodes = pd.DataFrame(rows_nodes).sort_values(["scenario_id", "node_id"]).reset_index(drop=True)

        # 4) collected_reward sheet (per-scenario totals; include probabilities if you like)
        rows_cr = []
        for scen in data.scenarios:
            rows_cr.append({
                "scenario_id": scen.id,
                "scenario_probability": scen.prob,
                "collected_value_total": details["per_scen_totals"][scen.id],
            })
        df_cr = pd.DataFrame(rows_cr).sort_values("scenario_id").reset_index(drop=True)



        # --- GA SUMMARY (one row) ---
        ga_summary_rows = [{
            "pop_size": gp["pop_size"],
            "max_generations": gp["max_generations"],
            "time_limit_sec": gp["time_limit_sec"],
            "elite_ratio": gp["elite_ratio"],
            "stall_ratio": gp["stall_ratio"],
            "coverage_ratio": gp["coverage_ratio"],
            "p_cross": gp["p_cross"],
            "p_mut_swap": gp["p_mut_swap"],
            "p_mut_add": gp["p_mut_add"],
            "p_mut_remove": gp["p_mut_remove"],
            "heuristic_seed_ratio": gp["heuristic_seed_ratio"],
            "generations_executed": res["generations"],  # how many gens GA actually ran
            "stall_counter_final": res["stall"],
            "stop_reason": res.get("stop_reason", ""),
            "best_makespan": res.get("best_makespan", None),
            "elapsed_sec_run_ga": round(res.get("elapsed_sec", 0.0), 3),
            "elapsed_sec_outer": round(res.get("run_time_sec", 0.0), 3),  # set earlier in your script
        }]
        df_ga_summary = pd.DataFrame(ga_summary_rows)

        # --- PER-GENERATION HISTORY ---
        hist = res["history"]  # list of dicts per generation
        # Required columns: generation, mean and max fitness
        # We'll include extra columns too (median, p10, p90, best_makespan) if available
        rows_hist = []
        for h in hist:
            rows_hist.append({
                "generation": h["generation"],
                "fitness_mean": h.get("mean"),
                "fitness_max": h.get("best"),
                "fitness_median": h.get("median"),
                "fitness_p10": h.get("p10"),
                "fitness_p90": h.get("p90"),
                "best_makespan": h.get("best_makespan"),
                # optional: number of individuals recorded
                "population_size": len(h.get("fitness", [])),
            })
        df_ga_history = pd.DataFrame(rows_hist).sort_values("generation").reset_index(drop=True)

        xlsx_path = os.path.join(out_dir, "results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_global.to_excel(writer, sheet_name="global_results", index=False)
            df_routes.to_excel(writer, sheet_name="routes", index=False)
            df_nodes.to_excel(writer, sheet_name="node_details", index=False)
            df_cr.to_excel(writer, sheet_name="collected_reward", index=False)
            # NEW:
            df_ga_summary.to_excel(writer, sheet_name="ga_summary", index=False)
            df_ga_history.to_excel(writer, sheet_name="ga_history", index=False)

        print(f"[GA] Excel report saved: {xlsx_path}")



else:
    # run optimization in single_run_mode
    if user_inputs.experiment_mode == "single_run":
        if user_inputs.optimization_mode == "deterministic_optimal_evaluation":
            base_output_folder = 'outputs'
            subfolder_name = user_inputs.parameters_df.loc["folder_to_be_evaluated", "value"]
            user_inputs.subfolder_path = os.path.join(base_output_folder, subfolder_name)
        mip_inputs = mip_setup.InputsSetup(user_inputs)
        mip_solve.mathematical_model_solve(mip_inputs)

    elif user_inputs.experiment_mode == "scenario_run":
        min_scenario_number = user_inputs.parameters_df.loc["min_scenario_number", "value"]
        user_inputs.max_scenario_number = user_inputs.parameters_df.loc["max_scenario_number", "value"]
        step_size_scenario_number = user_inputs.parameters_df.loc["step_size_scenario_number", "value"]
        scenario_size_set = list(range(min_scenario_number, user_inputs.max_scenario_number + 1, step_size_scenario_number))
        base_output_folder = 'outputs'

        if user_inputs.optimization_mode == "two_stage_optimization":
            current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
            subfolder_name = f"scenario_run_on_{current_time}"
            user_inputs.subfolder_path = os.path.join(base_output_folder, subfolder_name)
            os.makedirs(user_inputs.subfolder_path, exist_ok=True)
        elif user_inputs.optimization_mode == "deterministic_optimal_evaluation":
            subfolder_name = user_inputs.parameters_df.loc["folder_to_be_evaluated", "value"]
            user_inputs.subfolder_path = os.path.join(base_output_folder, subfolder_name)
        for n_scenarios in scenario_size_set:
            print("The run starts for {} scenarios.".format(n_scenarios))
            user_inputs.n_scenarios = n_scenarios
            mip_inputs = mip_setup.InputsSetup(user_inputs)
            mip_solve.mathematical_model_solve(mip_inputs)

    elif user_inputs.experiment_mode == "scenario_increasing_deviation_run":
        min_scenario_number = user_inputs.parameters_df.loc["increasing_deviation_min_scenario_number", "value"]
        max_scenario_number = user_inputs.parameters_df.loc["increasing_deviation_max_scenario_number", "value"]
        step_size_scenario_number = user_inputs.parameters_df.loc["increasing_deviation_step_size_scenario_number", "value"]
        scenario_size_set = list(range(min_scenario_number, max_scenario_number+1, step_size_scenario_number))

        min_scenario_rate = user_inputs.parameters_df.loc["increasing_deviation_min_rate", "value"]
        max_scenario_rate = user_inputs.parameters_df.loc["increasing_deviation_max_rate", "value"]
        step_size_scenario_rate = user_inputs.parameters_df.loc["increasing_deviation_step_size_rate", "value"]
        scenario_rate_set = np.round(np.arange(min_scenario_rate, max_scenario_rate+step_size_scenario_rate, step_size_scenario_rate), 1).tolist()


        current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
        base_output_folder = 'outputs'
        subfolder_name = f"scenario_run_increasing_deviation_run_on_{current_time}"
        user_inputs.subfolder_path = os.path.join(base_output_folder, subfolder_name)
        os.makedirs(user_inputs.subfolder_path, exist_ok=True)

        for n_scenarios in scenario_size_set:
            for scenario_rate in scenario_rate_set:
                print("The run starts for {} scenarios and {} rate.".format(n_scenarios, scenario_rate))
                user_inputs.n_scenarios = n_scenarios
                user_inputs.scenario_rate = scenario_rate
                mip_inputs = mip_setup.InputsSetup(user_inputs)
                mip_solve.mathematical_model_solve(mip_inputs)

    elif user_inputs.experiment_mode == "combination_run":
        fire_prone_node_list = user_inputs.problem_data_df.query("state == 0")["node_id"].tolist()
        list_combinations = list()

        for n in range(len(fire_prone_node_list) + 1):
            combn_list = list(combinations(fire_prone_node_list, n))
            if user_inputs.parameters_df.loc["n_nodes", "value"] <= 12:
                list_combinations += combn_list
            else:
                list_combinations += sample(combn_list, min(20, len(combn_list)))
        list_combinations = list_combinations[1:]
        # i=list_combinations[5]
        user_inputs.run_start_date = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
        for i in list_combinations:
            print(i)
            mip_inputs = mip_setup.InputsSetup(user_inputs, i)
            run_result = mip_solve.mathematical_model_solve(mip_inputs)



