# astar/astar.py
import heapq
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

# Reuse the GA stack (single source of truth)
from ga.model import (
    travel_time,
    decode_routes,
    simulate_scenario_report,
    value_and_makespan,
    evaluate_solution_details,  # rich report for Excel
)
from ga.core import ProblemData, NodeId, STATE_INIT_FIRE
import math
from typing import Dict, Set
from ga.core import STATE_INIT_FIRE, STATE_PROOF  # if not exported, use 1 and 4

PROB_SUM_SNAP_TOL = 1e-2
EPS = 1e-9

def _prob_sum_effective(data) -> float:
    ps = sum(s.prob for s in data.scenarios)
    if math.isclose(ps, 1.0, rel_tol=0.0, abs_tol=PROB_SUM_SNAP_TOL):
        return 1.0
    return ps



Chromosome = Dict[int, List[NodeId]]  # {vehicle: [nodes...]}


# ---------- Small helpers ----------

def _deepcopy_routes(routes: Chromosome) -> Chromosome:
    return {k: v[:] for k, v in routes.items()}


def reward_fixed(data: ProblemData, routes: Chromosome) -> float:
    """
    Scenario-weighted reward contributed by already scheduled nodes in `routes`.
    Uses the same simulator as GA. We sum only visited nodes' collected_value.
    """
    if all(len(v) == 0 for v in routes.values()):
        return 0.0
    dec = decode_routes(data, routes)
    V_fixed = 0.0
    for scen in data.scenarios:
        scen_total, details = simulate_scenario_report(data, dec, scen)
        for k in routes:
            for j in routes[k]:
                V_fixed += scen.prob * float(details[j].collected_value)
    return V_fixed

def UB_simple(data: ProblemData, routes: Chromosome, U: Set[NodeId]) -> float:
    """
    Admissible optimistic upper bound on expected value:
        UB = reward_fixed + sum_{j in U} (prob_sum * initial_value_j)
    Teleports & ignores future spread → optimistic but safe (admissible).
    """
    V_fixed = reward_fixed(data, routes)
    prob_sum = sum(s.prob for s in data.scenarios)
    if abs(prob_sum - 1.0) < 1e-6:   # <-- normalize
        prob_sum = 1.0
    sum_pi = sum(data.nodes[j].initial_value for j in U)
    return V_fixed + prob_sum * sum_pi


def _vehicle_end_state_from_decoded(data: ProblemData, routes: Chromosome):
    """
    For travel-aware UB: return per-vehicle (end_time, end_location) given partial routes.
    """
    dec = decode_routes(data, routes)
    end_state = {}
    for k, seq in dec.sequence.items():
        t = 0.0
        if seq:
            for a, b in zip(seq, seq[1:]):
                t += travel_time(data, a, b)
            end_state[k] = (t, seq[-1])
        else:
            end_state[k] = (0.0, data.base_id)
    return end_state

def UB_travel_aware(data: ProblemData, routes: Dict[int, List[NodeId]], U: Set[NodeId]) -> float:
    """
    Admissible tighter UB:
      UB = reward_fixed
           + sum_omega p^omega * sum_{j in U} UB_j^omega,  where

      UB_j^omega =
        - if j has INITIAL FIRE: value at earliest achievable arrival (decay from ts=0)
        - elif j is FORCED to ignite due to already-scheduled spreads in this scenario:
              if we can arrive by forced ts → full value,
              else decay from that forced ts using scenario-scaled rates
        - else (j not certainly burning): full initial value (may remain cold)

    Notes:
      * "Forced" is computed one hop from scheduled initial fires whose drop time
        is already after their tm (cannot be undone by future insertions).
      * We still honor mandatory water hops when computing earliest arrivals.
      * We snap probability mass to 1.0 within a small tolerance to avoid 0.99 deflation.
    """
    V_fixed = reward_fixed(data, routes)
    if not U:
        return V_fixed

    # Per-vehicle end state for earliest-arrival calc
    end_state = _vehicle_end_state_from_decoded(data, routes)
    a_side = math.sqrt(data.node_area)

    UB_rem = 0.0
    for scen in data.scenarios:
        # Compute forced ignitions for this scenario based on fixed part of the plan
        forced_ts = _forced_ignitions_from_fixed(data, routes, scen, a_side)  # {node: ts_forced}

        for j in U:
            n = data.nodes[j]

            # Default optimistic contribution: may stay cold
            contrib = n.initial_value

            # Determine earliest achievable arrival at j (from any vehicle)
            best_arrival = float("inf")
            for t_k, loc_k in end_state.values():
                t_direct = t_k + travel_time(data, loc_k, j)
                if data.water_ids:
                    t_via_w = min(
                        t_k + travel_time(data, loc_k, w) + travel_time(data, w, j)
                        for w in data.water_ids
                    )
                    best_arrival = min(best_arrival, t_direct, t_via_w)
                else:
                    best_arrival = min(best_arrival, t_direct)

            # Scenario-scaled rates for j
            spread_eff = n.spread_rate * (1.0 + scen.rate_change)
            amel_eff   = n.amel_rate   * (1.0 + scen.rate_change)
            d_sm = a_side / spread_eff if spread_eff > 0 else float("inf")
            d_me = a_side / amel_eff   if amel_eff   > 0 else float("inf")
            beta = n.initial_value / (d_sm + d_me) if math.isfinite(d_sm + d_me) else 0.0

            if n.state == STATE_INIT_FIRE:
                # certainly burning at ts=0
                ts = 0.0
                te = ts + d_sm + d_me
                if math.isfinite(best_arrival):
                    if best_arrival >= te:
                        contrib = 0.0
                    else:
                        t_drop = max(best_arrival, ts)
                        decay  = beta * (t_drop - ts)
                        contrib = max(0.0, min(n.initial_value, n.initial_value - decay))
                else:
                    contrib = n.initial_value

            elif j in forced_ts:
                ts = forced_ts[j]              # forced ignition start
                te = ts + d_sm + d_me
                if math.isfinite(best_arrival):
                    if best_arrival <= ts + EPS:
                        # can arrive before (or at) forced ignition → full value
                        contrib = n.initial_value
                    elif best_arrival >= te:
                        contrib = 0.0
                    else:
                        t_drop = best_arrival
                        decay  = beta * (t_drop - ts)
                        contrib = max(0.0, min(n.initial_value, n.initial_value - decay))
                else:
                    contrib = n.initial_value
            else:
                # not certainly burning → keep full initial value (optimistic)
                contrib = n.initial_value

            UB_rem += scen.prob * contrib

    # Snap total probability to 1.0 if within tolerance
    prob_sum = sum(s.prob for s in data.scenarios)
    prob_eff = _prob_sum_effective(data)
    if prob_sum > 0 and not math.isclose(prob_sum, prob_eff, abs_tol=1e-12):
        UB_rem *= (prob_eff / prob_sum)

    return V_fixed + UB_rem


def _optimistic_leg_time(data: ProblemData, loc: NodeId, j: NodeId) -> float:
    """Minimum travel time from loc to j, either direct or via best water (refill time = 0)."""
    direct = travel_time(data, loc, j)
    if data.water_ids:
        via_w = min(travel_time(data, loc, w) + travel_time(data, w, j) for w in data.water_ids)
        return min(direct, via_w)
    return direct

def _order_candidates_by_fire_then_distance(
    data: ProblemData, routes: Chromosome, Uset: Set[NodeId]
) -> List[NodeId]:
    """
    Priority 1: nodes with initial fire (state==STATE_INIT_FIRE) first.
    Priority 2: within each group, ascending by shortest optimistic leg time from any vehicle's end state.
    Tie-breaker: by node id.
    """
    # where are vehicles now?
    end_state = _vehicle_end_state_from_decoded(data, routes)  # {veh: (t_end, loc_end)}
    end_locs = [loc for (_t, loc) in end_state.values()]

    def key_fn(j: NodeId):
        has_init_fire = (data.nodes[j].state == STATE_INIT_FIRE)
        # best distance from any current vehicle end location
        if end_locs:
            best_dt = min(_optimistic_leg_time(data, loc, j) for loc in end_locs)
        else:
            best_dt = _optimistic_leg_time(data, data.base_id, j)
        # we want initial fires first → use (0, …) for them; (1, …) otherwise
        primary = 0 if has_init_fire else 1
        return (primary, best_dt, j)

    return sorted(Uset, key=key_fn)



def _order_candidates_by_fire_then_threshold(data: ProblemData,
                                             routes: Chromosome,
                                             Uset: Set[NodeId]) -> List[NodeId]:
    """
    Order remaining nodes by:
      1) initial-fire nodes first (state == STATE_INIT_FIRE),
      2) ascending estimated threshold duration d_sm = a_side / (spread_rate * avg_scenario_scale),
      3) tiebreaker: node id.

    Notes:
      - For initial-fire nodes, tm = 0 + d_sm; we don't include ts since it's 0.
      - For non-initial nodes, ts is unknown; we still rank by d_sm (smaller means faster spread if ignited).
      - avg_scenario_scale = sum_ω p_ω * (1 + rate_change_ω).
    """
    a_side = math.sqrt(data.node_area)
    avg_scale = sum(s.prob * (1.0 + s.rate_change) for s in data.scenarios)
    if avg_scale <= 0:
        avg_scale = 1.0  # fallback, should not happen

    def tm_est(j: NodeId) -> float:
        n = data.nodes[j]
        eff = n.spread_rate * avg_scale
        if eff <= 0:
            return float("inf")
        return a_side / eff  # duration to threshold (ts omitted)

    def key_fn(j: NodeId):
        is_init = (data.nodes[j].state == STATE_INIT_FIRE)
        return (0 if is_init else 1, tm_est(j), j)

    return sorted(Uset, key=key_fn)



def _forced_ignitions_from_fixed(
    data: ProblemData,
    routes: Dict[int, List[NodeId]],
    scen,
    a_side: float
) -> Dict[NodeId, float]:
    """
    For scenario `scen`, return {node_id: forced_ts} of nodes that are guaranteed
    to ignite due to already-scheduled initial-fire nodes that will spread.

    Logic:
      - Decode partial plan to get planned arrival times at already-scheduled nodes.
      - For each scheduled INITIAL-FIRE node m:
          * Compute tm_m under scenario scaling.
          * Planned drop time at m is t_drop = max(arrival[m], 0).
          * If t_drop > tm_m + EPS, spread at tm_m is unavoidable.
          * Then each eligible neighbor nb (in J*, not fire-proof, not initial-fire)
            is forced to ignite at ts = tm_m in this scenario.
      - Only one-hop forcing (no multi-hop cascade here).
    """
    from ga.model import decode_routes

    dec = decode_routes(data, routes)
    forced: Dict[NodeId, float] = {}

    for k, seq in routes.items():
        for m in seq:
            nm = data.nodes[m]
            if nm.state != STATE_INIT_FIRE:
                continue

            # planned arrival at m (exists because m is in routes)
            a_m = dec.arrival_fire.get(m, None)
            if a_m is None:
                continue

            # scenario-effective thresholds for m
            spread_eff_m = nm.spread_rate * (1.0 + scen.rate_change)
            amel_eff_m   = nm.amel_rate   * (1.0 + scen.rate_change)
            d_sm_m = a_side / spread_eff_m if spread_eff_m > 0 else float("inf")
            # d_me_m not needed here; tm is enough to check spreading
            tm_m = 0.0 + d_sm_m         # ts=0 for initial fire

            t_drop_m = max(a_m, 0.0)    # drop cannot occur before ignition start (0)
            if t_drop_m > tm_m + EPS:
                # spread at tm_m is unavoidable; mark eligible neighbors as forced at ts=tm_m
                for nb in nm.neighbors:
                    if nb not in data.Jstar:
                        continue
                    nn = data.nodes[nb]
                    if nn.state in (STATE_PROOF, STATE_INIT_FIRE):
                        continue
                    # keep the earliest (minimum) forced ts if multiple parents force nb
                    prev = forced.get(nb, float("inf"))
                    forced[nb] = min(prev, tm_m)

    return forced


from ga.model import decode_routes_with_waits, value_and_makespan_with_waits

def _zero_waits_for(routes: Chromosome) -> Dict[int, List[float]]:
    return {k: [0.0]*len(routes[k]) for k in routes}

def _ensure_waits_len(waits: Dict[int, List[float]], routes: Chromosome):
    """Extend waits[k] with zeros if routes grew (append) or ensure lengths match."""
    for k, lst in routes.items():
        if k not in waits:
            waits[k] = [0.0]*len(lst)
        elif len(waits[k]) < len(lst):
            waits[k].extend([0.0]*(len(lst)-len(waits[k])))

def _earliest_ignition_after_arrival(data: ProblemData, dec, node_j: NodeId) -> Optional[float]:
    """
    Given a decoded partial plan (dec) that includes node_j with a planned arrival,
    return min ts_j^ω over scenarios where node_j ignites AND ts_j^ω > arrival_j.
    If none, return None.
    """
    a_j = dec.arrival_fire.get(node_j, None)
    if a_j is None or not math.isfinite(a_j):
        return None
    ts_candidates = []
    for scen in data.scenarios:
        _, details = simulate_scenario_report(data, dec, scen)
        nd = details[node_j]
        if nd.had_fire == 1 and nd.ts is not None and nd.ts > a_j + EPS:
            ts_candidates.append(nd.ts)
    if not ts_candidates:
        return None
    return min(ts_candidates)


from typing import Optional, List, Dict, Set, Tuple

# ---- NEW: candidate ordering (init-fire first → smaller t_m → shorter optimistic leg) ----
def _order_candidates_by_fire_then_tm_then_dist(
    data: ProblemData,
    routes: Chromosome,
    Uset: Set[NodeId],
) -> List[NodeId]:
    # representative scenario for tm (expected rate change)
    sbar = sum(s.prob * s.rate_change for s in data.scenarios)
    a_side = math.sqrt(data.node_area)

    def tm_of(j: NodeId) -> float:
        n = data.nodes[j]
        ts = 0.0 if n.state == STATE_INIT_FIRE else float("inf")
        spread_eff = n.spread_rate * (1.0 + sbar)
        if spread_eff <= 0 or not math.isfinite(spread_eff):
            return float("inf")
        return ts + a_side / spread_eff

    end_state = _vehicle_end_state_from_decoded(data, routes)
    end_locs = [loc for (_t, loc) in end_state.values()] or [data.base_id]

    def best_leg(j: NodeId) -> float:
        def leg_from(loc):
            return _optimistic_leg_time(data, loc, j)
        return min(leg_from(loc) for loc in end_locs)

    def key(j: NodeId):
        is_init = (data.nodes[j].state == STATE_INIT_FIRE)
        return (0 if is_init else 1, tm_of(j), best_leg(j), j)

    return sorted(Uset, key=key)


# ---- NEW: fast greedy completion with inline “wait if wasted” repair ----
def greedy_rollout_complete(
    data: ProblemData,
    routes_partial: Chromosome,
    waits_partial: Dict[int, List[float]],
    Uset_in: Set[NodeId],
) -> Tuple[Chromosome, Dict[int, List[float]]]:
    routes = {k: v[:] for k, v in routes_partial.items()}
    waits  = {k: w[:] for k, w in waits_partial.items()}
    Uset   = set(Uset_in)

    _ensure_waits_len(waits, routes)

    while Uset:
        ordered = _order_candidates_by_fire_then_tm_then_dist(data, routes, Uset)
        j = ordered[0]

        # choose vehicle with earliest arrival if we append j to its tail
        best = None
        for k in routes.keys():
            trial_routes = {kk: vv[:] for kk, vv in routes.items()}
            trial_waits  = {kk: ww[:] for kk, ww in waits.items()}
            trial_routes[k].append(j)
            _ensure_waits_len(trial_waits, trial_routes)
            dec_trial = decode_routes_with_waits(data, trial_routes, trial_waits)
            a_j = dec_trial.arrival_fire.get(j, float("inf"))
            if best is None or a_j < best[0]:
                best = (a_j, k)

        _, kstar = best
        routes[kstar].append(j)
        _ensure_waits_len(waits, routes)

        # “wait if wasted” repair for this newly appended j
        dec_now = decode_routes_with_waits(data, routes, waits)
        a_j = dec_now.arrival_fire.get(j, None)
        ts_min = _earliest_ignition_after_arrival(data, dec_now, j)  # earliest forced/initial ignition ≥ a_j
        if a_j is not None and ts_min is not None and ts_min > a_j + EPS:
            # add wait before the leg to this new j
            idx_leg = len(routes[kstar]) - 1
            waits[kstar][idx_leg] += (ts_min - a_j)

        Uset.remove(j)

    return routes, waits

# ---------- A* main ----------

@dataclass
class AStarHistoryRow:
    step: int
    incumbent_best: float
    top_ub: float
    open_size: int
    pruned_cum: int


# ---- UPDATED: run_astar with incumbent seeding + per-pop rollout + new ordering ----
def run_astar(
    data: ProblemData,
    candidate_nodes: Optional[List[NodeId]] = None,
    time_limit_sec: int = 600,
    gap_tol: float = 0.05,
    use_travel_aware_ub: bool = True,
    branch_all_insert_positions: bool = False,
    log_every: int = 50,
):
    """
    Best-upper-bound-first search (A* in maximize form), now with:
      - Feasible incumbent seeding via greedy rollout,
      - Greedy completion at each pop (anytime improvement),
      - Stronger branching order (init-fire → t_m → optimistic leg).
    """
    t0 = time.time()
    if candidate_nodes is None:
        candidate_nodes = list(data.Jstar)

    def UB(routes: Chromosome, Uset: Set[NodeId]) -> float:
        return UB_travel_aware(data, routes, Uset) if use_travel_aware_ub else UB_simple(data, routes, Uset)

    # root state
    routes0: Chromosome = {k: [] for k in range(data.n_vehicles)}
    waits0: Dict[int, List[float]] = _zero_waits_for(routes0)
    U0: Set[NodeId] = set(candidate_nodes)
    ub0 = UB(routes0, U0)

    # --- Feasible incumbent seed via greedy rollout ---
    best_value = -math.inf
    best_plan: Optional[Chromosome] = None
    best_makespan: Optional[float] = None
    best_decoder = None
    best_waits: Dict[int, List[float]] = {}

    try:
        seed_routes, seed_waits = greedy_rollout_complete(data, routes0, waits0, U0)
        V0, M0, routes_pruned0, decw0 = value_and_makespan_with_waits(data, seed_routes, seed_waits)
        best_value, best_makespan, best_plan = V0, M0, routes_pruned0
        best_decoder, best_waits = decw0, seed_waits
    except Exception:
        # If anything goes wrong in rollout, proceed without incumbent
        pass

    OPEN = [(-ub0, 0, routes0, waits0, U0)]
    heapq.heapify(OPEN)
    next_id = 1

    expanded = 0
    pruned = 0
    step = 0
    stop_reason = "open_exhausted"

    # history for plots/Excel
    history: List[AStarHistoryRow] = []
    history.append(AStarHistoryRow(step=step, incumbent_best=best_value, top_ub=ub0, open_size=len(OPEN), pruned_cum=pruned))

    while OPEN:
        if time.time() - t0 >= time_limit_sec:
            stop_reason = "time_limit"
            break

        negub, _, routes, waits, Uset = heapq.heappop(OPEN)
        ub = -negub

        # incumbent pruning by UB
        if best_value > -math.inf and ub <= best_value * (1.0 + gap_tol):
            pruned += 1
            step += 1
            top_next = (-OPEN[0][0]) if OPEN else -math.inf
            history.append(AStarHistoryRow(step=step, incumbent_best=best_value, top_ub=top_next, open_size=len(OPEN), pruned_cum=pruned))
            continue

        # --- Anytime improvement: greedy complete this partial and update incumbent ---
        try:
            gr_routes, gr_waits = greedy_rollout_complete(data, routes, waits, Uset)
            Vg, Mg, rp_g, decw_g = value_and_makespan_with_waits(data, gr_routes, gr_waits)
            if (Vg > best_value) or (abs(Vg - best_value) <= 1e-9 and (best_makespan is None or Mg < best_makespan)):
                best_value, best_makespan, best_plan = Vg, Mg, rp_g
                best_decoder, best_waits = decw_g, gr_waits
        except Exception:
            pass

        # Goal test
        if not Uset:
            V, M, routes_pruned, decw = value_and_makespan_with_waits(data, routes, waits)
            if (V > best_value) or (abs(V - best_value) <= 1e-9 and (best_makespan is None or M < best_makespan)):
                best_value = V
                best_makespan = M
                best_plan = routes_pruned
                best_decoder = decw
                best_waits = waits
            expanded += 1
            step += 1
            top_next = (-OPEN[0][0]) if OPEN else -math.inf
            history.append(AStarHistoryRow(step=step, incumbent_best=best_value, top_ub=top_next, open_size=len(OPEN), pruned_cum=pruned))
            if log_every and (expanded % log_every == 0):
                print(f"[A*] expanded={expanded} pruned={pruned} best={best_value:.4f} topUB={top_next:.4f} open={len(OPEN)}")
            continue

        # Branching order: init-fire → small t_m → short leg
        ordered_candidates = _order_candidates_by_fire_then_tm_then_dist(data, routes, Uset)

        # Branch over ALL remaining nodes (keeps completeness)
        for j in ordered_candidates:
            if branch_all_insert_positions:
                for k in range(data.n_vehicles):
                    rlen = len(routes[k])
                    for pos in range(rlen + 1):
                        child_routes = _deepcopy_routes(routes)
                        child_routes[k].insert(pos, j)
                        child_waits = {kk: vv[:] for kk, vv in waits.items()}
                        _ensure_waits_len(child_waits, child_routes)
                        child_U = set(Uset); child_U.remove(j)

                        # Normal child
                        child_ub = UB(child_routes, child_U)
                        if not (best_value > -math.inf and child_ub <= best_value * (1.0 + gap_tol)):
                            heapq.heappush(OPEN, (-child_ub, next_id, child_routes, child_waits, child_U))
                            next_id += 1

                        # Waited child (if wasted)
                        dec_child = decode_routes_with_waits(data, child_routes, child_waits)
                        ts_min = _earliest_ignition_after_arrival(data, dec_child, j)
                        if ts_min is not None:
                            a_j = dec_child.arrival_fire.get(j, None)
                            if a_j is not None and ts_min > a_j + EPS:
                                Delta = ts_min - a_j
                                waited_routes = _deepcopy_routes(child_routes)
                                waited_waits = {kk: vv[:] for kk, vv in child_waits.items()}
                                waited_waits[k][pos] += Delta  # wait before leg to j at 'pos'
                                waited_U = set(child_U)
                                waited_ub = UB(waited_routes, waited_U)
                                if not (best_value > -math.inf and waited_ub <= best_value * (1.0 + gap_tol)):
                                    heapq.heappush(OPEN, (-waited_ub, next_id, waited_routes, waited_waits, waited_U))
                                    next_id += 1
            else:
                # append-to-tail + optional waited child if arrival is before ignition
                for k in range(data.n_vehicles):
                    # normal child
                    child_routes = _deepcopy_routes(routes)
                    child_routes[k].append(j)
                    child_waits = {kk: vv[:] for kk, vv in waits.items()}
                    _ensure_waits_len(child_waits, child_routes)
                    child_U = set(Uset); child_U.remove(j)

                    child_ub = UB(child_routes, child_U)
                    if not (best_value > -math.inf and child_ub <= best_value * (1.0 + gap_tol)):
                        heapq.heappush(OPEN, (-child_ub, next_id, child_routes, child_waits, child_U))
                        next_id += 1

                    # waited child (if wasted)
                    dec_child = decode_routes_with_waits(data, child_routes, child_waits)
                    ts_min = _earliest_ignition_after_arrival(data, dec_child, j)
                    if ts_min is not None:
                        a_j = dec_child.arrival_fire.get(j, None)
                        if a_j is not None and ts_min > a_j + EPS:
                            Delta = ts_min - a_j
                            waited_routes = _deepcopy_routes(child_routes)
                            waited_waits = {kk: vv[:] for kk, vv in child_waits.items()}
                            idx_leg = len(waited_routes[k]) - 1
                            waited_waits[k][idx_leg] += Delta
                            waited_U = set(child_U)
                            waited_ub = UB(waited_routes, waited_U)
                            if not (best_value > -math.inf and waited_ub <= best_value * (1.0 + gap_tol)):
                                heapq.heappush(OPEN, (-waited_ub, next_id, waited_routes, waited_waits, waited_U))
                                next_id += 1

        expanded += 1
        step += 1
        top_next = (-OPEN[0][0]) if OPEN else -math.inf
        history.append(AStarHistoryRow(step=step, incumbent_best=best_value, top_ub=top_next, open_size=len(OPEN), pruned_cum=pruned))
        if log_every and (expanded % log_every == 0):
            print(f"[A*] expanded={expanded} pruned={pruned} best={best_value:.4f} topUB={top_next:.4f} open={len(OPEN)}")

    runtime = time.time() - t0
    return {
        "best": (best_plan, best_value, best_makespan),
        "best_decoder": best_decoder,
        "best_waits": best_waits,
        "runtime_sec": runtime,
        "nodes_expanded": expanded,
        "nodes_pruned": pruned,
        "stop_reason": stop_reason,
        "ub_type": "travel_aware" if use_travel_aware_ub else "simple",
        "branching": "all_insert_positions" if branch_all_insert_positions else "append_tail",
        "history": [h.__dict__ for h in history],
    }

# ---------- Results writer (plots + Excel) ----------

def _make_outdir_astar(base_dir: str = "outputs/astar") -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("astar_%Y_%m_%d_%H_%M")
    out_dir = os.path.join(base_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_astar_outputs(out_dir: str, data: ProblemData, res: dict):
    """
    Save plots and results.xlsx for A* in `out_dir`.
    Expects res from run_astar (contains best plan/value/makespan, history, counts).
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- PLOTS ----
    hist = res.get("history", [])
    if hist:
        steps = [h["step"] for h in hist]
        inc = [h["incumbent_best"] if math.isfinite(h["incumbent_best"]) else float("nan") for h in hist]
        top = [h["top_ub"] if math.isfinite(h["top_ub"]) else float("nan") for h in hist]

        plt.figure()
        plt.plot(steps, inc, label="Incumbent best (value)")
        plt.plot(steps, top, label="Top UB")
        plt.xlabel("Expansion step")
        plt.ylabel("Value / UB")
        plt.title("A*: Incumbent vs Upper Bound")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "astar_convergence.png"), dpi=160)
        plt.close()

        plt.figure()
        opens = [h["open_size"] for h in hist]
        prunes = [h["pruned_cum"] for h in hist]
        plt.plot(steps, opens, label="OPEN size")
        plt.plot(steps, prunes, label="Pruned (cumulative)")
        plt.xlabel("Expansion step")
        plt.ylabel("Count")
        plt.title("A*: Search Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "astar_search_stats.png"), dpi=160)
        plt.close()

    # ---- EXCEL ----
    best_plan, best_val, best_mk = res["best"]

    # If no feasible plan found (best_plan None), write minimal summary
    if best_plan is None:
        df_sum = pd.DataFrame([{
            "best_value": None, "best_makespan": None,
            "runtime_sec": res.get("runtime_sec", None),
            "stop_reason": res.get("stop_reason", ""),
            "nodes_expanded": res.get("nodes_expanded", 0),
            "nodes_pruned": res.get("nodes_pruned", 0),
            "ub_type": res.get("ub_type", ""),
            "branching": res.get("branching", "")
        }])
        with pd.ExcelWriter(os.path.join(out_dir, "results.xlsx"), engine="xlsxwriter") as writer:
            df_sum.to_excel(writer, sheet_name="astar_summary", index=False)
        return

    # Rich evaluation for sheets
    # Rich evaluation for sheets (use decoder with waits if available)
    dec_override = res.get("best_decoder")
    report = evaluate_solution_details(data, best_plan, dec_override=dec_override)
    dec = report["decoder"]
    best_waits = res.get("best_waits", {})

    per_scen_totals = report["per_scen_totals"]       # dict scen_id -> value
    per_scen_details = report["per_scen_details"]     # dict scen_id -> {node -> NodeDetail}
    timeline = report["timeline"]                     # per-vehicle steps

    # --- global_results sheet ---
    n_scen = len(data.scenarios)
    n_nodes = len(data.nodes)
    n_init_fires = sum(1 for j in data.Jstar if data.nodes[j].state == 1)
    init_fire_ids = [j for j in data.Jstar if data.nodes[j].state == 1]

    global_rows = [
        ("n_scenarios", n_scen),
        ("expected_collected_value", float(best_val) if best_val is not None else None),
        ("scenario_collected_value_results", str(per_scen_totals)),
        ("run_time", round(res.get("runtime_sec", 0.0), 3)),
        ("number_of_nodes", n_nodes),
        ("number_of_initial_fires", n_init_fires),
        ("number_of_vehicles", data.n_vehicles),
        ("initial_fire_node_IDs", str(init_fire_ids)),
        ("best_makespan", float(best_mk) if best_mk is not None else None),
    ]
    df_global = pd.DataFrame(global_rows, columns=["result_name", "results"])

    # --- routes sheet ---
    # --- routes sheet (with waits) ---
    route_rows = []
    for k, fires in best_plan.items():
        waits_k = best_waits.get(k, [])
        order = 1
        for idx, j in enumerate(fires):
            wait_val = float(waits_k[idx]) if idx < len(waits_k) else 0.0
            route_rows.append({
                "vehicle": k,
                "order": order,
                "node_id": j,
                "arrival_time": dec.arrival_fire.get(j, None),
                "wait_before_leg": wait_val
            })
            order += 1
    df_routes = pd.DataFrame(route_rows).sort_values(["vehicle", "order"]).reset_index(drop=True)


    # --- node_details sheet ---
    node_rows = []
    for scen_id, details in per_scen_details.items():
        for j, nd in details.items():
            node_rows.append({
                "scenario_id": scen_id,
                "node_id": j,
                "had_fire": nd.had_fire,
                "status": nd.status,
                "ts": nd.ts,
                "tm": nd.tm,
                "te": nd.te,
                "planned_arrival": dec.arrival_fire.get(j, None),
                "drop_time": nd.drop_time,
                "collected_value": nd.collected_value
            })
    df_nodes = pd.DataFrame(node_rows).sort_values(["scenario_id", "node_id"]).reset_index(drop=True)

    # --- collected_reward sheet ---
    df_cr = pd.DataFrame(
        [{"scenario_id": sid, "collected_value": float(val)} for sid, val in per_scen_totals.items()]
    ).sort_values("scenario_id").reset_index(drop=True)

    # --- astar_summary sheet ---
    df_astar_summary = pd.DataFrame([{
        "runtime_sec": round(res.get("runtime_sec", 0.0), 3),
        "stop_reason": res.get("stop_reason", ""),
        "nodes_expanded": res.get("nodes_expanded", 0),
        "nodes_pruned": res.get("nodes_pruned", 0),
        "ub_type": res.get("ub_type", ""),
        "branching": res.get("branching", ""),
        "candidate_nodes": str(sorted(set().union(*best_plan.values())) if best_plan else "[]"),
    }])

    # --- astar_history sheet ---
    df_hist = pd.DataFrame(res.get("history", []))

    xlsx_path = os.path.join(out_dir, "results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="global_results", index=False)
        df_routes.to_excel(writer, sheet_name="routes", index=False)
        df_nodes.to_excel(writer, sheet_name="node_details", index=False)
        df_cr.to_excel(writer, sheet_name="collected_reward", index=False)
        df_astar_summary.to_excel(writer, sheet_name="astar_summary", index=False)
        df_hist.to_excel(writer, sheet_name="astar_history", index=False)

    # also drop the best plan as JSON-ish txt for quick inspection
    try:
        import json
        with open(os.path.join(out_dir, "best_plan.json"), "w", encoding="utf-8") as f:
            json.dump({"best_routes": best_plan, "best_value": best_val, "best_makespan": best_mk}, f, indent=2)
    except Exception:
        pass
