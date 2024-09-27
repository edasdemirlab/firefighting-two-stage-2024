# firefighting model by erdi dasdemir
# first successful run !!! March 28, 2023 - 3:35 pm
# successful run after all bugs are fixed !! March 29, 2023 - 17:00
# combinations mode is added June 15, 2023 - 17:00


# import required packages
import numpy as np
import pandas as pd
import openpyxl

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
user_inputs.optimization_mode = user_inputs.parameters_df.loc["optimization_mode", "value"]
user_inputs.experiment_mode = user_inputs.parameters_df.loc["experiment_mode", "value"]

# modes
# single_run: runs MIP as a single optimization task
# combination_run: runs MIP in the combination mode (to evaluate the impact of quantity and location of initial fires)
# instance_generate: generate a new WUI scenario based case instance


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



