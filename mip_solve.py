import gurobipy as gp
import numpy as np
from gurobipy import GRB
import re
import pandas as pd
import os
from datetime import datetime
import time

def callback(model, where):
    if where == GRB.Callback.MESSAGE:
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        m = re.search("Best objective ([^,]+), best bound ([^,]+), gap (.*)%$", msg)
        if m:
            obj_best, obj_bound, gap = m.group(1, 2, 3)
            obj_best = float(obj_best)
            obj_bound = float(obj_bound)
            gap = float(gap) / 100
            print(f"obj_best={obj_best}, obj_bound={obj_bound}, gap={gap}")
            model._bests.append(obj_best)
            model._bounds.append(obj_bound)
            model._gaps.append(gap)




def model_organize_results(var_values, var_df):
    counter = 0
    for v in var_values:
        # if(v.X>0):
        current_var = re.split("\[|,|]", v.varName)[:-1]
        current_var.append(round(v.X, 4))
        var_df.loc[counter] = current_var
        counter = counter + 1
        # with open("./math_model_outputs/" + 'mip-results.txt',
        #           "w") as f:  # a: open for writing, appending to the end of the file if it exists
        #     f.write(','.join(map(str, current_var)) + '\n')
        # print(','.join(map(str,current_var )))
    return var_df


def mathematical_model_solve(mip_inputs):
    # the formulation is available at below link:
    # https://docs.google.com/document/d/1cCx4SCTII76LPAp1McpIxybUQPRcqfJZxiNHsSsYXQ8/

    model = gp.Model("two_stage_firefighting")  # Carvana Supply Chain Optimizer

    big_M_temp = 999

    # add stage 1 decision variables
    # add link variables - if the vehicle k moves from region i to j; 0, otherwise.
    x_ijk = model.addVars(
        mip_inputs.links,
        vtype=GRB.BINARY,
        name="x_ijk",
    )

    tv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        # ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_j",
    )

    tv_h = model.addVar(
        lb=0,
        # ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="tv_h",
    )

    #
    lv_j = model.addVars(
        mip_inputs.fire_ready_node_ids,
        lb=0,
        # ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_j",
    )

    lv_h = model.addVar(
        lb=0,
        # ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="lv_h",
    )

    w_ijkl = model.addVars(
        mip_inputs.s_ijkw_links,
        vtype=GRB.BINARY,
        name="w_ijkl",
    )

    # stage 2 decision variables
    z_ijw = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="z_ijw",
    )

    q_ijw = model.addVars(
        mip_inputs.neighborhood_links,
        vtype=GRB.BINARY,
        name="q_ijw",
    )

    y_jw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="y_jw",
    )

    b_jw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="b_jw",
    )


    s1_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s1_iw",
    )

    s2_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s2_iw",
    )

    s3_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s3_iw",
    )

    s4_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s4_iw",
    )
    s5_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s5_iw",
    )

    s6_iw = model.addVars(
        mip_inputs.ns_pair,
        vtype=GRB.BINARY,
        name="s6_iw",
    )

    ts_jw = model.addVars(
        mip_inputs.ns_pair,
        lb=0,
        # ub=mip_inputs.time_limit,
        vtype=GRB.CONTINUOUS,
        name="ts_jw",
    )

    tm_jw = model.addVars(
        mip_inputs.ns_pair,
        lb=0,
        # ub=mip_inputs.time_limit*2,
        vtype=GRB.CONTINUOUS,
        name="tm_jw",
    )

    te_jw = model.addVars(
        mip_inputs.ns_pair,
        lb=0,
        # ub=mip_inputs.time_limit * 2,
        vtype=GRB.CONTINUOUS,
        name="te_jw",
    )


    r_jw = model.addVars(
        mip_inputs.ns_pair,
        lb=0,
        # ub=[value for value in mip_inputs.initial_values.values() for _ in range(mip_inputs.n_scenarios)],
        vtype=GRB.CONTINUOUS,
        name="r_jw",
    )

    # model.update()

    # set objective
    # cost_first_party_routes = s_tr_1.prod({key: mip_inputs.hauler_miles_multiplier_first_party * (value + mip_inputs.hauler_fixed_cost_first_party) for key, value in mip_inputs.routes_first_party_distance.items()})
    # cost_third_party_routes = s_tr_3.prod(mip_inputs.routes_third_party_cost_per_hauler)


    expected_collected_value = r_jw.prod({(j, w): mip_inputs.scenario_probabilities[w - 1] for (j, w) in r_jw.keys()})
    # penalty_coef_return_time = 0
    # penalty_return_time = penalty_coef_return_time * tv_h

    # model.setObjective(obj_max - penalty_coef_spread * obj_penalize_fire_spread - penalty_coef_return_time * obj_penalize_operation_time) #
    # model.setObjective(expected_collected_value - penalty_return_time)

    # set objectives
    model.NumObj = 2

    model.setObjectiveN(expected_collected_value, index=0, priority=2, weight=1, name='expected_collected_value')
    model.setObjectiveN(tv_h, index=1, priority=1, weight=-1, name='tv_h')


    # equations for prize collection
    # constraint 3-6 - determines collected prizes from at each node
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            # j = 2
            # w = 1
            model.addConstr(r_jw[j, w] <= mip_inputs.initial_values[j] - mip_inputs.ns_value_degradation_rate[j, w] * tv_j[j] - mip_inputs.initial_values[j]*b_jw[j, w] + mip_inputs.M_3_1[j, w]*(1 - y_jw[j, w]) + mip_inputs.M_3_2[j, w]*s6_iw[j, w])
            model.addConstr(r_jw[j, w] <= mip_inputs.initial_values[j] * (2 - s6_iw[j,w] - y_jw[j,w]))
            model.addConstr(r_jw[j, w] <= mip_inputs.initial_values[j] * (1 - s5_iw[j, w]))
            model.addConstr(b_jw[j, w] >= y_jw[j, w] - mip_inputs.M_6[j] * tv_j[j])
            # model.addConstr(r_jw[j, w] <= 10)


    # equations for scheduling and routing decisions
    # Constraint 7 - a vehicle that leaves the base must return to the base
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(mip_inputs.fire_ready_node_ids, mip_inputs.base_node_id,  k))

    # Constraint 8 - each vehicle can leave the base only once
    for k in mip_inputs.vehicle_list:
        model.addConstr(x_ijk.sum(mip_inputs.base_node_id, mip_inputs.fire_ready_node_ids, k) <= 1)

    # Constraint 9 - flow balance equation -- incoming vehicles must be equal to the outgoing vehicles at each node
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, k) == x_ijk.sum(j, mip_inputs.fire_ready_node_ids_and_base, k))

    # Constraint 10 - at most one vehicle can visit a node
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') <= 1)

    # Constraint 11 - water resource selection for refilling
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(x_ijk.sum(i[0], i[1], i[2]) == w_ijkl.sum(i[0], i[1], i[2], '*'))

    # Constraint 12 - water resource connections for refilling
    for i in mip_inputs.s_ijkw_links:
        model.addConstr(2 * w_ijkl[i] <= x_ijk.sum(i[0], i[3], i[2]) + x_ijk.sum(i[3], i[1], i[2]) )

    # Constraint 13 - water resource connections for refilling
    for i in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(i, mip_inputs.fire_ready_node_ids, k) == x_ijk.sum(i, mip_inputs.water_node_id, k))

    # Constraint 14 - water resource connections for refilling
    for j in mip_inputs.fire_ready_node_ids:
        for k in mip_inputs.vehicle_list:
            model.addConstr(x_ijk.sum(mip_inputs.fire_ready_node_ids, j, k) == x_ijk.sum(mip_inputs.water_node_id, j, k))

    # Constraint 15 - time limitation
    model.addConstr(tv_h <= mip_inputs.time_limit)

    # Constraint 16 - determines return time to the base, considering the time of vehicle with maximum return time
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_h >= tv_j[j] +
                        mip_inputs.links_durations[(j, mip_inputs.base_node_id, 1)] * x_ijk.sum(j, mip_inputs.base_node_id, '*') -
                        mip_inputs.M_16[j] * (1 - x_ijk.sum(j, mip_inputs.base_node_id, '*')))

    # Constraint 17 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] <= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') + mip_inputs.M_16[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    model.addConstr(lv_h == 0)
    # model.addConstr(x_ijk[1, 3, 1] == 1)
    # model.addConstr(x_ijk[1, 8, 2] == 1)

    # Constraint 18 - determines arrival times to the nodes
    for j in mip_inputs.fire_ready_node_ids:
        home_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                          k[0] == mip_inputs.base_node_id and k[1] == j}
        model.addConstr(tv_j[j] >= lv_h + x_ijk.prod(home_to_j_coef, mip_inputs.base_node_id, j, '*') - mip_inputs.M_16[j] * (
                1 - x_ijk.sum(mip_inputs.base_node_id, j, '*')))

    # Constraint 19 - determines arrival times to the nodes
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] == i and k[1] in mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] in mip_inputs.water_node_id and k[1] == j}
            model.addConstr(
                tv_j[j] <= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
                    water_to_j_coef, mip_inputs.water_node_id, j, '*') + mip_inputs.M_19[i, j] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 20 - determines arrival times to the nodes
    for i in mip_inputs.fire_ready_node_ids:
        to_j_list = [x for x in mip_inputs.fire_ready_node_ids if x != i]
        for j in to_j_list:
            i_to_water_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] == i and k[1] in mip_inputs.water_node_id}
            water_to_j_coef = {k: v for k, v in mip_inputs.links_durations.items() if
                               k[0] in mip_inputs.water_node_id and k[1] == j}
            model.addConstr(
                tv_j[j] >= tv_j[i] + lv_j[i] + x_ijk.prod(i_to_water_coef, i, mip_inputs.water_node_id, '*') + x_ijk.prod(
                    water_to_j_coef, mip_inputs.water_node_id, j, '*') - mip_inputs.M_19[i, j] * (1 - x_ijk.sum(i, j, '*')))

    # Constraint 21 - no arrival time at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(tv_j[j] <= mip_inputs.M_16[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))

    # Constraint 22 - no loitering at unvisited nodes
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(lv_j[j] <= mip_inputs.M_16[j] * x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))


    # # Constraint 20 - vehicle arrival has to be after fire arrival (start)
    # for j in mip_inputs.fire_ready_node_ids:
    #     model.addConstr(tv_j[j] - ts_j[j] >= mip_inputs.M_19 * (x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*') - 1))
    #
    # # Constraint 21 - vehicle can not arrive after the fire finished
    # for j in mip_inputs.fire_ready_node_ids:
    #     model.addConstr(tv_j[j] <= te_j[j])

    # stage 2
    # equations linking fire arrivals and scheduling decisions
    # Constraint 23 - fire spread case 1: t_v =0 --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(mip_inputs.M_23[i] * tv_j[i] >= (1 - s1_iw[i, w]))

    # Constraint 24 - fire spread case that allows case 2 and 3: t_v > 0
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tv_j[i] <= mip_inputs.M_24[i] * s4_iw[i, w])

    # Constraint 25 - fire spread case  2: t_v > 0 and t_v >= t_m --> fire spreads
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tv_j[i] - tm_jw[i, w] + mip_inputs.small_enough_coefficient <= mip_inputs.M_25[i, w] * s2_iw[i, w])

    # Constraint 26 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tm_jw[i, w] - tv_j[i] <= mip_inputs.M_26 * (s1_iw[i, w] + s3_iw[i, w]))

    # Constraint 27 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(ts_jw[i, w] - tv_j[i] <= mip_inputs.M_27[i] * s5_iw[i, w])

    # Constraint 28 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tv_j[i] - ts_jw[i, w] + mip_inputs.small_enough_coefficient <= mip_inputs.M_28[i] * (1 - s5_iw[i, w]))

    # Constraint 29 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tv_j[i] - te_jw[i, w] <= mip_inputs.M_29[i] * s6_iw[i, w])

    # Constraint 30 - fire spread case  3: t_v > 0 and t_v < t_m --> fire does not spread
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(te_jw[i, w] - tv_j[i] + mip_inputs.small_enough_coefficient <= mip_inputs.M_30[i] * (1 - s6_iw[i, w]))

    # Constraint 31 - fire spread cases: only one of case 1, i.e. t_v=0, and case 2, i.e. t_v>0, can occur
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(s1_iw[i, w] + s4_iw[i, w] == 1)

    # Constraint 32 - fire spread cases: only one of case 3, i.e. t_v>=t_m, and case 4, i.e. t_v<t_m, can occur
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(s4_iw[i, w] >= s2_iw[i, w] + s3_iw[i, w])

    # Constraint 33 - fire spread cases: if there is no fire in node i, it cannot spread to the adjacent nodes
    for i in mip_inputs.fire_ready_node_ids:
        # i=5
        # w=1
        for w in mip_inputs.scenarios_list:
            i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i and l[2] == w]
            i_neighborhood_size = len(i_neighborhood)
            model.addConstr(z_ijw.sum(i, '*', w) <= i_neighborhood_size * y_jw[i, w])

    # Constraint 34 - fire spread cases:  there is fire in node i, but no vehicle process it, i.e. t_v=0
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i and l[2] == w]
            i_neighborhood_size = len(i_neighborhood)
            model.addConstr(z_ijw.sum(i, '*', w) >= i_neighborhood_size * (s1_iw[i, w] + y_jw[i, w] - 1))

    # Constraint 35 - fire spread cases:  there is fire in node i, a vehicle process it after it max point, i.e. t_v>=t_m --> it must spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i and l[2] == w]
            i_neighborhood_size = len(i_neighborhood)
            model.addConstr(z_ijw.sum(i, '*', w) >= i_neighborhood_size * (s2_iw[i, w] + y_jw[i, w] - 1))

    # Constraint 36 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i and l[2] == w]
            i_neighborhood_size = len(i_neighborhood)
            model.addConstr(z_ijw.sum(i, '*', w) <= i_neighborhood_size * (1 - s3_iw[i, w] + s5_iw[i, w]))

    # Constraint 37 - fire spread cases:  there is fire in node i, a vehicle process it after it before max point, i.e. t_v<t_m --> it cant spread to the adjacent cells
    for i in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            i_neighborhood = [l for l in mip_inputs.neighborhood_links if l[0] == i and l[2] == w]
            i_neighborhood_size = len(i_neighborhood)
            model.addConstr(z_ijw.sum(i, '*', w) >= i_neighborhood_size * s5_iw[i, w])

    # Constraint 38 - if a fire spreads to an adjacent node, the adjacent node must have a fire.
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j and l[2] == w])
            model.addConstr(j_neighborhood_size * y_jw[j, w] >= z_ijw.sum('*', j, w))

    # Constraint 39 - if there is no fire spreads to a node j, there cant be a fire at the node j.
    # exclude nodes with fires at start

    # Convert lists to sets and find the difference
    fire_ready_node_ids_without_initial_fires = list(set(mip_inputs.fire_ready_node_ids) - set(mip_inputs.set_of_active_fires_at_start))
    for j in fire_ready_node_ids_without_initial_fires:
        for w in mip_inputs.scenarios_list:
            j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j and l[2] == w])
            model.addConstr(z_ijw.sum('*', j, w) >= y_jw[j, w])

    # Constraint 40 - a node is visited only if it has a fire, i.e. if a node is visited, then it must have fire
    for j in mip_inputs.fire_ready_node_ids:
        model.addConstr(y_jw.sum(j, '*') >= x_ijk.sum(mip_inputs.fire_ready_node_ids_and_base, j, '*'))  # x_ijk.sum(mip_inputs.node_list, j, '*'))

    # Constraint 41 - active fires at start
    for w in mip_inputs.scenarios_list:
        model.addConstr(gp.quicksum(y_jw[j, w] for j in mip_inputs.set_of_active_fires_at_start) == len(
            mip_inputs.set_of_active_fires_at_start))

    # Constraint 42 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            j_neighborhood_size = len([l for l in mip_inputs.neighborhood_links if l[1] == j and l[2] == w])
            model.addConstr(j_neighborhood_size * q_ijw.sum('*', j, w) >= z_ijw.sum('*', j, w))

    # Constraint 43 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(q_ijw.sum('*', j, w) <= 1)

    # Constraint 44-46 - determine fire spread
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            temp_neighborhood_list = [l[0] for l in mip_inputs.neighborhood_links if l[1] == j and l[2] == w]
            # temp_neighborhood_list = [x for x in mip_inputs.node_object_dict[j].get_neighborhood_list() if x not in mip_inputs.fire_proof_node_list]
            for i in temp_neighborhood_list:  #for i in mip_inputs.node_object_dict[j].get_neighborhood_list():
                #constraint 43
                model.addConstr(q_ijw[i, j, w] <= z_ijw[i, j, w])
                #constraint 44
                model.addConstr(ts_jw[j, w] <= tm_jw[i, w] + mip_inputs.M_45 * (1 - z_ijw[i, j, w]))
                # constraint 45
                if j in mip_inputs.set_of_active_fires_at_start:
                    model.addConstr(ts_jw[j, w] >= tm_jw[i, w] - mip_inputs.M_45 * (2 - z_ijw[i, j, w] - q_ijw[i, j, w] + 1))
                else:
                    model.addConstr(ts_jw[j, w] >= tm_jw[i, w] - mip_inputs.M_45 * (2 - z_ijw[i, j, w] - q_ijw[i, j, w] + 0))

    # Constraint 47- determine fire arrival (spread) time
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(ts_jw[j, w] <= mip_inputs.M_45 * z_ijw.sum('*', j, w))

    # Constraint 48 - start time of active fires
    for w in mip_inputs.scenarios_list:
        model.addConstr(gp.quicksum(ts_jw[j, w] for j in mip_inputs.set_of_active_fires_at_start) == 0)

    # Constraint 49 - determine fire spread time (the time at which the fire reaches its maximum size)
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(tm_jw[j, w] == ts_jw[j, w] + (mip_inputs.node_area / mip_inputs.ns_fire_spread_rate[j, w]))

    # Constraint 50 - fire end time when it is not processed and burned down by itself
    for j in mip_inputs.fire_ready_node_ids:
        for w in mip_inputs.scenarios_list:
            model.addConstr(te_jw[j, w] == tm_jw[j, w] + (mip_inputs.node_area / mip_inputs.ns_fire_amelioration_rate[j, w]))


    # Constraint xx - valid inequality cuts
    #
    # for i in mip_inputs.fire_ready_node_ids:
    #     print(i)
    #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
    #     i_neighborhood_size = len(i_neighborhood)
    #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * b_j[i])
    #
    # for i in mip_inputs.fire_ready_node_ids:
    #     i_neighborhood = [l[1] for l in mip_inputs.neighborhood_links if l[0] == i]
    #     i_neighborhood_size = len(i_neighborhood)
    #     model.addConstr(gp.quicksum(y_j[j] for j in i_neighborhood) >= i_neighborhood_size * s2_i[i])

    # Define the file path and sheet name
    if mip_inputs.optimization_mode == "two_stage_optimization":
        if mip_inputs.experiment_mode == "single_run":
            file_name = mip_inputs.start_solution_file
            start_sol_path = os.path.join("inputs", "start_solution", file_name) if file_name is not None and not pd.isna(file_name) else ""
        else:
            file_name = "scenario_run_{0}_scenarios.xlsx".format(mip_inputs.n_scenarios-2)
            start_sol_path = os.path.join(mip_inputs.subfolder_path, file_name) if mip_inputs.subfolder_path else ""
        x_ijk_sheet_name = 'x_ijk_results'
        w_ijkl_sheet_name = 'w_ijkl_results'

        # Step 2: Check if the file exists
        if os.path.exists(start_sol_path):
            # Step 3: Read the Excel file and the specific sheet if it exists
            x_ijk_start_df = pd.read_excel(start_sol_path, sheet_name=x_ijk_sheet_name)
            w_ijkl_start_df = pd.read_excel(start_sol_path, sheet_name=w_ijkl_sheet_name)

            # Step 4: Set the start solution for x_ijk from the DataFrame
            for _, row in x_ijk_start_df.iterrows():
                var_name = row['var_name']  # You can use this if you want variable names (e.g., 'x_ijk')
                i = row['from_node_id']
                j = row['to_node_id']
                k = row['vehicle_id']
                value = row['value']

                # Set the starting solution for the variable x_ijk[i, j, k] if it exists
                x_ijk[(i, j, k)].start = value

            # Step 5: Set the start solution for w_ijkl from the DataFrame
            for _, row in w_ijkl_start_df.iterrows():
                var_name = row['var_name']  # You can use this if you want variable names (e.g., 'x_ijk')
                i = row['from_node_id']
                j = row['to_node_id']
                k = row['vehicle_id']
                l = row['water_node_id']
                value = row['value']

                # Set the starting solution for the variable x_ijk[i, j, k] if it exists
                w_ijkl[(i, j, k, l)].start = value

        else:
            print(f"File '{file_name}' does not exist. No start solution will be fed to the model.")

    elif mip_inputs.optimization_mode == "deterministic_optimal_evaluation":
        x_ijk_sheet_name = 'x_ijk_results'
        w_ijkl_sheet_name = 'w_ijkl_results'
        if os.path.exists(mip_inputs.optimal_sol_path):
            # Step 3: Read the Excel file and the specific sheet if it exists
            x_ijk_start_df = pd.read_excel(mip_inputs.optimal_sol_path, sheet_name=x_ijk_sheet_name)
            w_ijkl_start_df = pd.read_excel(mip_inputs.optimal_sol_path, sheet_name=w_ijkl_sheet_name)

            # Step 4: Set the start solution for x_ijk from the DataFrame
            for _, row in x_ijk_start_df.iterrows():
                var_name = row['var_name']  # You can use this if you want variable names (e.g., 'x_ijk')
                i = row['from_node_id']
                j = row['to_node_id']
                k = row['vehicle_id']
                value = row['value']
                # Set the starting solution for the variable x_ijk[i, j, k] if it exists
                model.addConstr(x_ijk[i, j, k] == value)
            print(f"x_ijk variables have been succesfully set!")

            # Step 5: Set the start solution for w_ijkl from the DataFrame
            for _, row in w_ijkl_start_df.iterrows():
                var_name = row['var_name']  # You can use this if you want variable names (e.g., 'x_ijk')
                i = row['from_node_id']
                j = row['to_node_id']
                k = row['vehicle_id']
                l = row['water_node_id']
                value = row['value']
                # Set the starting solution for the variable x_ijk[i, j, k] if it exists
                model.addConstr(w_ijkl[i, j, k, l] == value)
            print(f"w_ijkl variables have been succesfully set!")

        else:
            print(f"Deterministic optimal solution file does not exist. No start solution will be fed to the model.")



    model.ModelSense = -1  # set objective to maximization
    # model.params.TimeLimit = 60
    model.params.MIPGap = 0.03
    model.params.Presolve = 2
    model.params.Cuts = 2
    model.params.MIPFocus = 2
    # model.params.Threads = 1
    # model.params.RINS = 0

    # model.params.LogFile = "gurobi_log"
    # model.params.Heuristics = 0.2

    env0 = model.getMultiobjEnv(0)
    env0.setParam('TimeLimit', 10800)
    # env0.setParam('NoRelHeurTime', 120)
    # env0.setParam('MIPFocus', 3)

    env1 = model.getMultiobjEnv(1)
    env1.setParam('TimeLimit', 10)

    # model.update()
    # model.write("model_hand2.lp")
    # (23.745 - 23.39) == (24.1-23.745)
    # 23.745 - 23.390
    # 0.455*0.355
    model.update()
    model.printStats()
    #p=model.presolve()
    # p.printStats()

    model._bounds = []
    model._bests = []
    model._gaps = []



    start_time = time.time()
    model.optimize(callback)
    end_time = time.time()
    run_time_cpu = round(end_time - start_time, 2)

    # for c in model.getConstrs():
    #     if c.Slack < 1e-6:
    #         print('Constraint %s is active at solution point' % (c.ConstrName))


    if model.Status == GRB.Status.INFEASIBLE:
        max_dev_result = None
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("Go check infeasible_model.ilp file")
    else:

        x_ijk_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'value'])
        x_ijk_results_df = model_organize_results(x_ijk.values(), x_ijk_results_df)


        y_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        y_jw_results_df = model_organize_results(y_jw.values(), y_jw_results_df)

        z_ijw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'scenario_id', 'value'])
        z_ijw_results_df = model_organize_results(z_ijw.values(), z_ijw_results_df)
        #z_ijw_results_df.loc[z_ijw_results_df["to_node_id"]=="24",]

        q_ijw_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id',  'scenario_id', 'value'])
        q_ijw_results_df = model_organize_results(q_ijw.values(), q_ijw_results_df)

        b_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        b_jw_results_df = model_organize_results(b_jw.values(), b_jw_results_df)

        ts_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        ts_jw_results_df = model_organize_results(ts_jw.values(), ts_jw_results_df)

        tm_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        tm_jw_results_df = model_organize_results(tm_jw.values(), tm_jw_results_df)

        te_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        te_jw_results_df = model_organize_results(te_jw.values(), te_jw_results_df)

        tv_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        tv_j_results_df = model_organize_results(tv_j.values(), tv_j_results_df)
        tv_j_results_df.loc[len(tv_j_results_df.index)] = [tv_h.varName, mip_inputs.base_node_id, tv_h.X]

        tl_j_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'value'])
        tl_j_results_df = model_organize_results(lv_j.values(), tl_j_results_df)
        tl_j_results_df.loc[len(tl_j_results_df.index)] = [lv_h.varName, mip_inputs.base_node_id, lv_h.X]

        r_jw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        r_jw_results_df = model_organize_results(r_jw.values(), r_jw_results_df)


        w_ijkl_results_df = pd.DataFrame(columns=['var_name', 'from_node_id', 'to_node_id', 'vehicle_id', 'water_node_id', 'value'])
        w_ijkl_results_df = model_organize_results(w_ijkl.values(), w_ijkl_results_df)

        s1_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s2_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s3_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s4_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s5_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s6_iw_results_df = pd.DataFrame(columns=['var_name', 'node_id', 'scenario_id', 'value'])
        s1_iw_results_df = model_organize_results(s1_iw.values(), s1_iw_results_df)
        s2_iw_results_df = model_organize_results(s2_iw.values(), s2_iw_results_df)
        s3_iw_results_df = model_organize_results(s3_iw.values(), s3_iw_results_df)
        s4_iw_results_df = model_organize_results(s4_iw.values(), s4_iw_results_df)
        s5_iw_results_df = model_organize_results(s5_iw.values(), s5_iw_results_df)
        s6_iw_results_df = model_organize_results(s6_iw.values(), s6_iw_results_df)
        s_ciw_results_df = pd.concat([s1_iw_results_df, s2_iw_results_df, s3_iw_results_df, s4_iw_results_df, s5_iw_results_df, s6_iw_results_df], axis=0)


        # model global results
        # expected_collected_value_result = model.objval + (penalty_coef_spread * sum(z_ijw_results_df.loc[:, 'value'])) + penalty_coef_return_time * tv_h.X
        # expected_collected_value_result = model.objval + penalty_coef_return_time * tv_h.X
        expected_collected_value_result = sum({key: r_jw[key].X * mip_inputs.scenario_probabilities[key[1]-1] for key in r_jw}.values())
        scenario_collected_value_results = {
            j: sum(r_jw[key].X for key in r_jw if key[1] == j)
            for j in set(key[1] for key in r_jw)
        }




        # Convert the dictionary to a DataFrame
        scenario_collected_value_results_df = pd.DataFrame(list(scenario_collected_value_results.items()), columns=['scenario_id', 'collected_value'])


        global_results_df = pd.DataFrame(columns=['n_scenarios', 'expected_collected_value', 'scenario_collected_value_results', 'model_obj_1_value', 'model_obj_2_value', 'model_obj_1_bound', 'model_obj_2_bound', 'gap_1', 'gap_2', 'gurobi_time', 'python_time'])

        global_results_df.loc[len(global_results_df.index)] = [mip_inputs.n_scenarios, expected_collected_value_result,
                                                               scenario_collected_value_results, *model._bests,
                                                               *model._bounds, *model._gaps,
                                                               model.runtime, run_time_cpu]


        # global_results_df.loc[len(global_results_df.index)] = [expected_collected_value_result, scenario_collected_value_results, model.objval, model.objbound, model.mipgap,
        #                                                        model.runtime, run_time_cpu]

        global_results_df["operation_duration"] = tv_h.X
        global_results_df["number_of_nodes"] = mip_inputs.n_nodes
        # global_results_df["number_of_scenarios"] = mip_inputs.n_scenarios
        global_results_df["number_of_initial_fires"] = len(mip_inputs.set_of_active_fires_at_start)
        global_results_df["average_number_of_fires_per_scenario"] = sum(y_jw_results_df.value > 0)/mip_inputs.n_scenarios
        # global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # subtract the base return time
        global_results_df["number_of_vehicles"] = len(mip_inputs.vehicle_list)  # subtract the base return time
        mip_inputs.base_node_id_string = str(mip_inputs.base_node_id)
        global_results_df["number_of_vehicles_used"] = len(np.unique(x_ijk_results_df.query("`from_node_id` == @mip_inputs.base_node_id_string & `value` > 0")["vehicle_id"].tolist()))  # subtract the base return time
        global_results_df["initial_fire_node_IDs"] = ','.join(map(str, mip_inputs.set_of_active_fires_at_start))


        global_results_df_row = global_results_df.copy()


        # Convert to a two-column DataFrame
        global_results_df = pd.DataFrame({
            'result_Name': global_results_df.columns,
            f'{mip_inputs.n_scenarios}_scenarios': global_results_df.iloc[0].values
        })




        current_time = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
        base_output_folder = 'outputs'

        if mip_inputs.experiment_mode == "single_run":
            writer_global_file_name = ""
            if mip_inputs.optimization_mode == "deterministic_optimal_evaluation":
                writer_file_name = os.path.join(base_output_folder, "single_run_deterministic_eval_{0}_nodes_{1}.xlsx".format(mip_inputs.n_nodes, current_time))
            else:
                writer_file_name = os.path.join(base_output_folder,  "single_run_{0}_nodes_{1}.xlsx".format(mip_inputs.n_nodes, current_time))

        elif mip_inputs.experiment_mode == "scenario_run":
            if mip_inputs.optimization_mode == "deterministic_optimal_evaluation":
                writer_file_name = os.path.join(mip_inputs.subfolder_path,"scenario_run_deterministic_eval_{0}_scenarios.xlsx".format(mip_inputs.n_scenarios))
                writer_global_file_name = os.path.join(mip_inputs.subfolder_path,"scenario_run_deterministic_eval_global_results.csv".format(mip_inputs.max_scenario_number))
            else:
                writer_file_name = os.path.join(mip_inputs.subfolder_path,"scenario_run_{0}_scenarios.xlsx".format(mip_inputs.n_scenarios))
                writer_global_file_name = os.path.join(mip_inputs.subfolder_path,"scenario_run_global_results.csv".format(mip_inputs.max_scenario_number))

        elif mip_inputs.experiment_mode == "scenario_increasing_deviation_run":
            if mip_inputs.optimization_mode == "deterministic_optimal_evaluation":
                writer_file_name = os.path.join(mip_inputs.subfolder_path, "scenario_incr_dev_run_deterministic_eval_{0}_scenarios_{1}_rate.xlsx".format(mip_inputs.n_scenarios, mip_inputs.scenario_rate))
                writer_global_file_name = os.path.join(mip_inputs.subfolder_path, "scenario_incr_dev_run__deterministic_eval_global_results.csv")
            else:
                writer_file_name = os.path.join(mip_inputs.subfolder_path, "scenario_incr_dev_run_{0}_scenarios_{1}_rate.xlsx".format(mip_inputs.n_scenarios, mip_inputs.scenario_rate))
                writer_global_file_name = os.path.join(mip_inputs.subfolder_path, "scenario_incr_dev_run_global_results.csv")

            global_results_df_row["scenario_rate"] = mip_inputs.scenario_rate

        if mip_inputs.experiment_mode != "single_run":
            if os.path.isfile(writer_global_file_name):
                global_results_df_row.to_csv(writer_global_file_name, mode="a", index=False, header=False)
            else:
                global_results_df_row.to_csv(writer_global_file_name, mode="a", index=False, header=True)

        writer = pd.ExcelWriter(writer_file_name)
        global_results_df.to_excel(writer, sheet_name='global_results')
        scenario_collected_value_results_df.to_excel(writer, sheet_name='scenario_collected_value')
        x_ijk_results_df.to_excel(writer, sheet_name='x_ijk_results')
        y_jw_results_df.to_excel(writer, sheet_name='y_jw_results')
        z_ijw_results_df.to_excel(writer, sheet_name='z_ijw_results')
        q_ijw_results_df.to_excel(writer, sheet_name='q_ijw_results')
        b_jw_results_df.to_excel(writer, sheet_name='b_jw_results')
        ts_jw_results_df.to_excel(writer, sheet_name='ts_jw_results')
        tm_jw_results_df.to_excel(writer, sheet_name='tm_jw_results')
        te_jw_results_df.to_excel(writer, sheet_name='te_jw_results')
        tv_j_results_df.to_excel(writer, sheet_name='tv_j_results')
        tl_j_results_df.to_excel(writer, sheet_name='tl_j_results')
        r_jw_results_df.to_excel(writer, sheet_name='r_jw_results')
        w_ijkl_results_df.to_excel(writer, sheet_name='w_ijkl_results')
        s_ciw_results_df.to_excel(writer, sheet_name='s_ciw_results')

        mip_inputs.problem_data_df.to_excel(writer, sheet_name='inputs_problem_data')
        mip_inputs.distance_df["flight_time"] = mip_inputs.distance_df["distance"] / mip_inputs.vehicle_flight_speed
        mip_inputs.distance_df.to_excel(writer, sheet_name='inputs_distances')
        mip_inputs.parameters_df.to_excel(writer, sheet_name='inputs_parameters')
        mip_inputs.scenarios_df.to_excel(writer, sheet_name='inputs_scenario_settings')
        mip_inputs.fire_ready_nodes_stage_2_df.to_excel(writer, sheet_name='inputs_scenario_rates')
        writer.close()

        print("The run is completed succesfully and the results are printed to the output folder!")

        # subfolder_name = f"max_scenario_{mip_inputs.max_scenario_number}_on_{current_time}"



        #
        # elif mip_inputs.experiment_mode == "combination_run":
        #     writer_file_name = os.path.join('outputs', "combination_results_{0}_nodes_{1}.csv".format(mip_inputs.n_nodes, mip_inputs.run_start_date))
        #     if os.path.isfile(writer_file_name):
        #         global_results_df.to_csv(writer_file_name, mode="a", index=False, header=False)
        #     else:
        #         global_results_df.to_csv(writer_file_name, mode="a", index=False, header=True)
        #
        # # global_results_df["operation_time"] = tv_h.X
        # global_results_df["number_of_jobs_arrived"] = sum(ts_j_results_df.value > 0) + len(mip_inputs.set_of_active_fires_at_start)
        # global_results_df["number_of_job_processed"] = sum(tv_j_results_df.value > 0) - 1  # substract the base return time

        return global_results_df



        # 24 - (24-mip_inputs.links_durations[1,7,1]) == mip_inputs.links_durations[1,7,1]