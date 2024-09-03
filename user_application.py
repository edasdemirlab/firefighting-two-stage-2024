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
experiment_mode = user_inputs.parameters_df.loc["mode", "value"]


# modes
# single_run: runs MIP as a single optimization task
# combination_run: runs MIP in the combination mode (to evaluate the impact of quantity and location of initial fires)
# instance_generate: generate a new WUI scenario based case instance





# run optimization in single_run_mode
if experiment_mode == "single_run":
    mip_inputs = mip_setup.InputsSetup(user_inputs)
    mip_solve.mathematical_model_solve(mip_inputs)

# run optimization in combination_mode
elif experiment_mode == "combination_run":
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

elif experiment_mode == "instance_generation":
    user_inputs.case_output_file_name = os.path.join('outputs', "wui_scenario_{0}.csv".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))
    generator.generate_grid(user_inputs)
    print("The inputs of the new instance are successfully generated! see outputs folder.")


#esther -->
# the role of default density
# how do we determine number of waters and blocks # sometimes it does not create blocks (is this case for water as well ? Should we make sure that there has to be at least 1 water and block when they are set to true?)
# what is the relationship between n and m
# initial fires
