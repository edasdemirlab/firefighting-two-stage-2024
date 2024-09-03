import gurobipy as gp
import pandas as pd
import os
from ast import literal_eval
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
class NodeSetup:
    def __init__(self, attributes_given):  # attributes=values
        self.attributes = attributes_given

    def get_node_id(self):
        return self.attributes["node_id"]

    def get_x_coordinate(self):
        return self.attributes["x_coordinate"]

    def get_y_coordinate(self):
        return self.attributes["y_coordinate"]

    def get_value_at_start(self):
        return float(self.attributes["value_at_start"])

    def get_value_degradation_rate(self):
        return float(self.attributes["value_degradation_rate"])

    def get_fire_degradation_rate(self):
        return float(self.attributes["fire_degradation_rate"])

    def get_fire_amelioration_rate(self):
        return float(self.attributes["fire_amelioration_rate"])

    def get_state(self):
        return self.attributes["state"]

    def get_neighborhood_list(self):
        return self.attributes["neighborhood_list"]

# define setup classes
class InputsSetup:
    def __init__(self, user_inputs, list_of_active_fires="NA"):

        # read problem input
        self.directory = user_inputs.directory
        # self.directory = os.path.join('inputs', 'inputs_to_load_5x5.xlsx')  # os.getcwd(),

        self.parameters_df = user_inputs.parameters_df.copy()
        # self.parameters_df = pd.read_excel(self.directory, sheet_name="parameters", index_col=0, engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')

        self.problem_data_df = user_inputs.problem_data_df.copy()
        # self.problem_data_df = pd.read_excel(self.directory, sheet_name="inputs_df", engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')

        self.scenarios_df = user_inputs.scenarios_df.copy()
        # self.problem_data_df = pd.read_excel(self.directory, sheet_name="inputs_df", engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')


        # self.experiment_mode = experiment_mode
        self.experiment_mode = self.parameters_df.loc["mode", "value"]

        if self.experiment_mode == "combination_run":
            self.problem_data_df.loc[[x - 1 for x in list(list_of_active_fires)], "state"] = 1
            self.run_start_date = user_inputs.run_start_date

        fix_df = self.problem_data_df.copy()
        self.problem_data_df["neighborhood_list"] = fix_df["neighborhood_list"].apply(literal_eval)  # make string lists of csv to python lists
        # self.parameters_df.loc["scenario_rate_changes", "value"] = literal_eval(self.parameters_df.loc["scenario_rate_changes", "value"])
        # self.parameters_df.loc["scenario_probabilities", "value"] = literal_eval(self.parameters_df.loc["scenario_probabilities", "value"])

        # problem parameters
        self.region_side_length = self.parameters_df.loc["region_side_length", "value"]
        self.node_area = self.parameters_df.loc["node_area", "value"]
        self.n_vehicles = self.parameters_df.loc["n_vehicles", "value"]
        self.vehicle_list = list(range(1, int(self.n_vehicles) + 1))
        self.vehicle_flight_speed = self.parameters_df.loc["vehicle_flight_speed", "value"]
        self.time_limit = self.parameters_df.loc["time_limit", "value"]
        self.n_nodes = self.parameters_df.loc["n_nodes", "value"]
        self.node_list = list(range(1, int(self.n_nodes) + 1))
        self.base_node_id = self.problem_data_df.query("node_state == 6")["node_id"].item()
        self.water_node_id = self.problem_data_df.query("node_state == 5")["node_id"].tolist()
        self.block_node_id = self.problem_data_df.query("node_state == 4")["node_id"].tolist()
        self.small_enough_coefficient = 10 ** -4
        # scenario parameters
        self.n_scenarios = len(self.scenarios_df)
        self.scenarios_list = self.scenarios_df["scenario_id"].tolist()
        self.scenario_probabilities = self.scenarios_df["scenario_probability"].tolist()
        self.scenario_rate_changes = self.scenarios_df["scenario_rate_change"].tolist()



        # define empty lists for classes, i.e. each element of the list will be an element of the corresponding class
        self.node_object_dict = {}  # dictionary of elements of node attribute class
        self.links_multidict_input = {}  # multi-dictionary input for the transportation cost information for each available arc in the network

        # set neighborhood df
        # node states --> 0: without forest fire, 1: with forest fire, 2: rescued, 3: burned down, 4: fire proof, 5: water, 6:home/base
        fire_proof_states = list(range(2, 7))
        fire_proof_nodes = self.problem_data_df.query("node_state in @fire_proof_states").loc[:,"node_id"].tolist()
        self.fire_proof_node_list = fire_proof_nodes
        self.neighborhood_links_df = self.problem_data_df[['node_id', 'neighborhood_list']].copy()
        self.neighborhood_links_df = self.neighborhood_links_df.explode("neighborhood_list")
        self.neighborhood_links_df = self.neighborhood_links_df.rename(columns={"node_id": "from", "neighborhood_list": "to"})
        self.neighborhood_links_df = self.neighborhood_links_df.query("`from` not in @fire_proof_nodes & `to` not in @fire_proof_nodes")
        self.neighborhood_links_df = self.neighborhood_links_df.reset_index(drop=True)
        # Replicate each row for the number of scenarios
        # Repeat each row according to the number of scenarios
        self.neighborhood_links_df_deplicated = self.neighborhood_links_df.loc[self.neighborhood_links_df.index.repeat(self.n_scenarios)].reset_index(drop=True)
        # Add the scenario_id column
        self.neighborhood_links_df_deplicated['scenario_id'] = self.scenarios_list * len(self.neighborhood_links_df)
        self.neighborhood_links = gp.tuplelist(list(self.neighborhood_links_df_deplicated.itertuples(index=False, name=None)))

        # set active fire nodes
        self.set_of_active_fires_at_start = self.problem_data_df.query("`node_state`==1")["node_id"].tolist()

        # create pairwise distance data frame
        coordinate_array = self.problem_data_df[["x_coordinate", "y_coordinate"]].values
        euclidean_distances_array = euclidean_distances(coordinate_array, coordinate_array)
        self.distance_matrix = pd.DataFrame(
            euclidean_distances_array,
            columns=self.problem_data_df["node_id"],
            index=self.problem_data_df["node_id"]
        )
        self.distance_matrix.index.name = 'from'
        self.distance_df = self.distance_matrix.stack().reset_index()
        self.distance_df .columns = ["from", "to", "distance"]

        # eliminate non-existing connections
        # 1 - no inner connections
        index_drop = self.distance_df[(self.distance_df['from'] == self.distance_df['to'])].index
        self.distance_df.drop(index_drop, inplace=True)

        # 2 - no connections from base to water nodes (we assume that UAVs wait at base with full tank)
        index_drop = self.distance_df[(self.distance_df['from'] == self.base_node_id) & (self.distance_df['to'].isin(self.water_node_id))].index
        self.distance_df.drop(index_drop, inplace=True)

        # 3 - no connections from water nodes to base (UAV returns directly to the base after its final shot without refilling)
        index_drop = self.distance_df[(self.distance_df['from'].isin(self.water_node_id)) & (self.distance_df['to'] == self.base_node_id)].index
        self.distance_df.drop(index_drop, inplace=True)

        # 4 - no connections between water nodes
        index_drop = self.distance_df[(self.distance_df['from'].isin(self.water_node_id)) & (self.distance_df['to'].isin(self.water_node_id))].index
        self.distance_df.drop(index_drop, inplace=True)

        # 5 - no connections from and to the blocking nodes
        index_drop = self.distance_df[(self.distance_df['from'].isin(self.block_node_id)) | (self.distance_df['to'].isin(self.block_node_id))].index
        self.distance_df.drop(index_drop, inplace=True)

        # reset index
        self.distance_df.reset_index(drop=True)


        # start creating links
        flow_active_states = [0, 1, 5, 6]
        flow_active_nodes = self.problem_data_df.query("`node_state` in @flow_active_states")["node_id"]
        self.flow_active_distance_df = self.distance_df.query("`from` in @flow_active_nodes & `to`  in @flow_active_nodes").copy().reset_index(drop=True)

        # setup links and their distances
        for i in range(len(self.flow_active_distance_df)):
            values = self.flow_active_distance_df .loc[i, :]
            for k in range(1, int(self.n_vehicles) + 1):
                self.links_multidict_input[int(values["from"]), int(values["to"]), k] = (values["distance"] / self.vehicle_flight_speed)

        # setup multi dictionaries --> if you are unfamiliar with multidict and want to learn about, go to below link
        # https://www.gurobi.com/documentation/8.1/refman/py_multidict.html
        self.links, self.links_durations = gp.multidict(self.links_multidict_input)

        # create links for water resources
        self.fire_ready_nodes_data_df = self.problem_data_df.query("`node_id` not in @fire_proof_nodes").copy().reset_index(drop=True)

        self.fire_ready_node_ids = list(self.fire_ready_nodes_data_df["node_id"])
        self.fire_ready_node_ids_and_base = [self.base_node_id] + self.fire_ready_node_ids

        self.s_ijkw_links = gp.tuplelist()
        for i in self.fire_ready_node_ids:
            to_j_list = [x for x in self.fire_ready_node_ids if x != i]
            for j in to_j_list:
                for k in self.vehicle_list:
                    for w in self.water_node_id:
                        self.s_ijkw_links.append((i, j, k, w))


        # Creating fire_ready_nodes_stage_1_df
        self.fire_ready_nodes_stage_1_df = self.fire_ready_nodes_data_df[
            ['node_id', 'x_coordinate', 'y_coordinate', 'initial_value', 'node_state', 'neighborhood_list']]

        # Converting to dictionary
        self.fire_ready_nodes_stage_1_dict = self.fire_ready_nodes_stage_1_df.set_index('node_id').apply(
            lambda row: row.tolist(), axis=1).to_dict()
        self.node_ids, self.x_coodinates, self.y_coordinates, self.initial_values, self.node_states, self.neighborhood_lists = gp.multidict(self.fire_ready_nodes_stage_1_dict)



        # Creating fire_ready_nodes_stage_2_df
        self.fire_ready_nodes_stage_2_df = self.fire_ready_nodes_data_df[['node_id', 'initial_value', 'fire_spread_rate', 'fire_amelioration_rate']]
        # self.fire_ready_nodes_stage_2_tupledict = gp.tuplelist()

        # Replicate each row in df according to the number of scenarios
        df_replicated = self.fire_ready_nodes_stage_2_df.loc[self.fire_ready_nodes_stage_2_df.index.repeat(self.n_scenarios)].reset_index(drop=True)

        # Add the scenario_id column using scenario_list
        df_replicated['scenario_id'] = self.scenarios_list * len(self.fire_ready_nodes_stage_2_df)

        # Merge with the scenario DataFrame to get scenario_probability and scenario_rate_change
        df_extended = pd.merge(df_replicated, self.scenarios_df, on='scenario_id', how='left')

        # Calculate scenario_fire_spread_rate and scenario_fire_amelioration_rate
        df_extended['scenario_fire_spread_rate'] = (df_extended['fire_spread_rate'] * (1 + df_extended['scenario_rate_change'])).round(4)
        df_extended['scenario_fire_amelioration_rate'] = (df_extended['fire_amelioration_rate'] * (1 + df_extended['scenario_rate_change'])).round(4)

        # Calculate value_degradation_rate and add it as a new column
        df_extended['scenario_value_degradation_rate'] = (
                df_extended['initial_value'] / (
                (self.node_area / df_extended['scenario_fire_spread_rate']) +
                (self.node_area / df_extended['scenario_fire_amelioration_rate'])
        )
        ).round(4)

        # Creating fire_ready_nodes_stage_2_df
        self.fire_ready_nodes_stage_2_df = df_extended[
            ['node_id', 'scenario_id', 'scenario_value_degradation_rate', 'scenario_fire_spread_rate', 'scenario_fire_amelioration_rate']]

        # Converting to dictionary
        self.fire_ready_nodes_stage_2_df.set_index(['node_id', 'scenario_id'], inplace=True)
        # Convert to a dictionary with tuple values
        self.fire_ready_nodes_stage_2_dict = self.fire_ready_nodes_stage_2_df.apply(lambda row: row.tolist(), axis=1).to_dict()

        self.ns_pair, self.ns_value_degradation_rate, self.ns_fire_spread_rate, self.ns_fire_amelioration_rate = gp.multidict(self.fire_ready_nodes_stage_2_dict)


        # ----------------------------------------------------------------------------
        # Defining big-M values

        self.big_m_augmentation_for_rounding_errors = 1

        self.M_3_1 = dict()
        self.M_3_2 = dict()

        for j in self.fire_ready_node_ids:
            for w in self.scenarios_list:
                self.M_3_1[j, w] = self.ns_value_degradation_rate[j, w] * (self.time_limit - self.links_durations[(j, self.base_node_id, 1)]) + self.big_m_augmentation_for_rounding_errors
                # self.M_3_1[j, w] = 999
                self.M_3_2[j, w] = self.ns_value_degradation_rate[j, w] * (self.time_limit - self.links_durations[(j, self.base_node_id, 1)]) - self.initial_values[j] +self.big_m_augmentation_for_rounding_errors
                # self.M_3_2[j, w] = 999

        self.M_6 = dict()
        for j in self.fire_ready_node_ids:
            self.M_6[j] = (1/self.links_durations[(self.base_node_id, j, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_6[j] = 999


        self.M_16 = dict()
        for j in self.fire_ready_node_ids:
            self.M_16[j] = (self.time_limit - self.links_durations[
                (j, self.base_node_id, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_16[j] = 999


        self.M_19 = dict()
        for i in self.fire_ready_node_ids:
            t_max = self.time_limit
            d_i_h = self.links_durations[(i, self.base_node_id, 1)]
            max_d_i_w = max([self.links_durations[(i, w, 1)] for w in self.water_node_id])
            to_j_list = [x for x in self.fire_ready_node_ids if x != i]
            for j in to_j_list:
                max_d_w_j = max([self.links_durations[(w, j, 1)] for w in self.water_node_id])
                self.M_19[(i, j)] = (t_max - d_i_h + max_d_i_w + max_d_w_j) + self.big_m_augmentation_for_rounding_errors
                # self.M_19[(i, j)] = 999


        # self.M_19 = 6 * 30 * 24

        self.M_23 = dict()
        for i in self.fire_ready_node_ids:
            self.M_23[i] = (1 / self.links_durations[
                (self.base_node_id, i, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_23[i] = 999

        self.M_24 = dict()
        for i in self.fire_ready_node_ids:
            self.M_24[i] = (self.time_limit - self.links_durations[
                (i, self.base_node_id, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_24[i] = 999

        self.M_25 = dict()
        for i in self.fire_ready_node_ids:
            for w in self.scenarios_list:
                self.M_25[i, w] = (self.time_limit - self.links_durations[
                (i, self.base_node_id, 1)]) - self.node_area/self.ns_fire_spread_rate[i, w] + self.big_m_augmentation_for_rounding_errors
                # self.M_25[i,w] = 999

        self.M_26 = 1 * 30 * 24

        self.M_27 = dict()
        for i in self.fire_ready_node_ids:
            self.M_27[i] = (self.time_limit - self.links_durations[
                (self.base_node_id, i, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_27[i] = 999

        self.M_28 = dict()
        for i in self.fire_ready_node_ids:
            self.M_28[i] = (self.time_limit - self.links_durations[
                (i, self.base_node_id, 1)]) + self.small_enough_coefficient + self.big_m_augmentation_for_rounding_errors
            # self.M_28[i] = 999

        self.M_29 = dict()
        for i in self.fire_ready_node_ids:
            self.M_29[i] = (self.time_limit - self.links_durations[(i, self.base_node_id, 1)]) + self.big_m_augmentation_for_rounding_errors
            # self.M_29[i] = 999

        self.M_30 = dict()
        for i in self.fire_ready_node_ids:
            # self.M_30[i] = (self.time_limit - self.links_durations[(i, self.base_node_id, 1)]) + self.small_enough_coefficient + self.big_m_augmentation_for_rounding_errors
            self.M_30[i] = (self.time_limit + (self.node_area / self.ns_fire_spread_rate[i, w]) + (self.node_area / self.self.ns_fire_amelioration_rate[i, w]) - self.links_durations[(self.base_node_id, i, 1)]) + self.small_enough_coefficient + self.big_m_augmentation_for_rounding_errors
            # self.M_30[i] = 999


        # M_26 = dict()
        # for j in mip_inputs.fire_ready_node_ids:
        #     M_26[j] = len([l for l in mip_inputs.neighborhood_links if l[1] == j])
        #     # M_26[j] = 999

        self.M_44 = 1 * 30 * 24


# def list_combinations():
#     # read problem input
#     input_directory = os.path.join('inputs', 'inputs_to_load_5x5.xlsx')  # os.getcwd(),
#     input_parameters_df = pd.read_excel(self.directory, sheet_name="parameters", index_col=0, engine='openpyxl').dropna(
#         axis=0, how='all').dropna(axis=1, how='all')
#     self.problem_data_df = pd.read_excel(self.directory, sheet_name="inputs_df", engine='openpyxl').dropna(axis=0,
#                                                                                                            how='all').dropna(
#         axis=1, how='all')
#     # self.experiment_mode = experiment_mode
#     self.experiment_mode = self.parameters_df.loc["mode", "value"]

class UserInputsRead:
    def __init__(self):

        # read problem input
        self.directory = os.path.join('inputs', 'inputs_to_load.xlsx')  # os.getcwd(),
        self.parameters_df = pd.read_excel(self.directory, sheet_name="parameters", index_col=0, engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')
        self.problem_data_df = pd.read_excel(self.directory, sheet_name="problem_input", engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')
        self.scenarios_df = pd.read_excel(self.directory, sheet_name="scenarios_input", engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')

        # self.distance_df = pd.read_excel(self.directory, sheet_name="distance_df", engine='openpyxl').dropna(axis=0, how='all').dropna(axis=1, how='all')
