import pandas as pd
import math
import ast
from typing import Dict, List, Optional
from .core import (
    ProblemData, Node, Scenario, NodeId,
    STATE_BASE, STATE_WATER, STATE_FIRE_PRONE, STATE_INIT_FIRE
)

def load_problem(excel_path: str) -> ProblemData:
    """Read inputs_to_load.xlsx and return ProblemData."""
    xls = pd.ExcelFile(excel_path)
    df_nodes = pd.read_excel(xls, "problem_input")
    df_scen  = pd.read_excel(xls, "scenarios_input")
    df_par   = pd.read_excel(xls, "parameters")

    # parameters -> dict
    params = {str(r["parameter"]).strip(): r["value"] for _, r in df_par.iterrows()}
    # Required parameters for GA
    required = ["n_nodes", "n_vehicles", "vehicle_flight_speed", "node_area"]
    for p in required:
        if p not in params:
            raise ValueError(f"Parameter '{p}' missing in 'parameters' sheet.")
    n_nodes_param = int(params["n_nodes"])

    # Build nodes
    nodes: Dict[NodeId, Node] = {}
    base_id: Optional[NodeId] = None
    water_ids: List[NodeId] = []
    Jstar: List[NodeId] = []

    for _, r in df_nodes.iterrows():
        nid = int(r["node_id"])
        state = int(r["node_state"])

        # safely parse "[6, 2]" into [6, 2]
        raw = str(r["neighborhood_list"])
        try:
            neigh = ast.literal_eval(raw) if raw and raw != "nan" else []
        except Exception:
            neigh = []
        neigh = [int(x) for x in neigh]

        node = Node(
            id=nid,
            x=float(r["x_coordinate"]),
            y=float(r["y_coordinate"]),
            initial_value=float(r["initial_value"]),
            spread_rate=float(r["fire_spread_rate"]),
            amel_rate=float(r["fire_amelioration_rate"]),
            state=state,
            neighbors=neigh
        )
        nodes[nid] = node
        if state == STATE_BASE:
            if base_id is not None:
                raise ValueError("Multiple base nodes found (state=6). Exactly one required.")
            base_id = nid
        if state == STATE_WATER:
            water_ids.append(nid)
        if state in (STATE_FIRE_PRONE, STATE_INIT_FIRE):
            Jstar.append(nid)

    if base_id is None:
        raise ValueError("No base node found (state=6).")
    if len(water_ids) == 0:
        raise ValueError("At least one water node (state=5) is required.")
    if len(nodes) != n_nodes_param:
        raise ValueError(f"n_nodes parameter={n_nodes_param} but problem_input has {len(nodes)} rows.")

    # Scenarios (allow rounding error; normalize if close)
    # --- Scenarios: use probabilities as given (no checks, no normalization) ---
    scenarios: List[Scenario] = []
    for _, r in df_scen.iterrows():
        scenarios.append(Scenario(
            id=str(r["scenario_id"]),
            prob=float(r["scenario_probability"]),  # used as-is
            rate_change=float(r["scenario_rate_change"])
        ))

    # Quick field checks for J*
    for j in Jstar:
        n = nodes[j]
        if n.initial_value < 0 or n.spread_rate <= 0 or n.amel_rate <= 0:
            raise ValueError(f"Node {j} has invalid value/spread/amel rates.")

    return ProblemData(
        nodes=nodes,
        base_id=base_id,
        water_ids=water_ids,
        Jstar=Jstar,
        scenarios=scenarios,
        vehicle_speed=float(params["vehicle_flight_speed"]),
        n_vehicles=int(params["n_vehicles"]),
        node_area=float(params["node_area"])
    )

def load_ga_params(excel_path: str) -> dict:
    """
    Reads 'ga_parameters' sheet (parameter, value, note) and returns a dict with defaults.
    If the sheet is missing or a parameter absent, sensible defaults are used.
    """
    defaults = {
        "seed": 42,
        "pop_size": 40,
        "max_generations": 300,
        "time_limit_sec": 600,
        "coverage_ratio": 0.6,
        "p_cross": 0.9,
        "p_mut_swap": 0.2,
        "p_mut_add": 0.15,
        "p_mut_remove": 0.15,
        "heuristic_seed_ratio": 0.1,   # NEW: fraction of pop_size seeded heuristically
        "elite_ratio": 0.05,
        "stall_ratio": 0.15,
    }

    try:
        xls = pd.ExcelFile(excel_path)
        if "ga_parameters" not in xls.sheet_names:
            return defaults
        df = pd.read_excel(xls, "ga_parameters")
    except Exception:
        return defaults

    # Build map from sheet
    kv = {}
    for _, r in df.iterrows():
        k = str(r.get("parameter", "")).strip()
        if not k:
            continue
        v = r.get("value", None)
        kv[k] = v

    # Coerce types & fill defaults
    out = defaults.copy()

    def _as_int(name):
        if name in kv and kv[name] is not None and str(kv[name]).strip() != "":
            out[name] = int(float(kv[name]))
    def _as_float(name):
        if name in kv and kv[name] is not None and str(kv[name]).strip() != "":
            out[name] = float(kv[name])

    _as_int("seed")
    _as_int("pop_size")
    _as_int("max_generations")
    _as_int("time_limit_sec")

    _as_float("coverage_ratio")
    _as_float("p_cross")
    _as_float("p_mut_swap")
    _as_float("p_mut_add")
    _as_float("p_mut_remove")

    _as_float("heuristic_seed_ratio")
    out["heuristic_seed_ratio"] = min(max(out["heuristic_seed_ratio"], 0.0), 1.0)

    _as_float("elite_ratio")
    out["elite_ratio"] = min(max(out["elite_ratio"], 0.0), 1.0)

    _as_float("stall_ratio")
    out["stall_ratio"] = min(max(out["stall_ratio"], 0.0), 1.0)

    # Clamp a couple of values to safe ranges
    out["coverage_ratio"] = min(max(out["coverage_ratio"], 0.0), 1.0)
    for p in ("p_cross", "p_mut_swap", "p_mut_add", "p_mut_remove"):
        out[p] = min(max(out[p], 0.0), 1.0)

    return out

