import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
from .core import (
    ProblemData, NodeId, Scenario,
    STATE_INIT_FIRE, STATE_PROOF,
    COLD, BURNING, PROCESSED, BURNED_SIM
)

# ---------- Geometry (Euclidean) ----------
def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

def travel_time(data: ProblemData, i: NodeId, j: NodeId) -> float:
    if i == j:
        return 0.0
    ni, nj = data.nodes[i], data.nodes[j]
    dist_km = _euclidean(ni.x, ni.y, nj.x, nj.y)
    return dist_km / data.vehicle_speed

# ---------- Decoder (BASE → FIRE → WATER → FIRE …) ----------
Chromosome = Dict[int, List[NodeId]]  # {vehicle: [fire1, fire2, ...]}, union == J*

@dataclass
class DecodeResult:
    sequence: Dict[int, List[NodeId]]                 # route incl. base and waters
    arrival_fire: Dict[NodeId, float]                 # nominal arrival per fire node
    arrival_water: Dict[int, List[Tuple[NodeId,float]]]

def _best_water_for_next(data: ProblemData, j: NodeId, nextj: NodeId) -> NodeId:
    best, best_cost = None, float("inf")
    for w in data.water_ids:
        c = travel_time(data, j, w) + travel_time(data, w, nextj)
        if c < best_cost:
            best, best_cost = w, c
    assert best is not None
    return best

def decode_routes(data: ProblemData, chrom: Chromosome) -> DecodeResult:
    """Builds BASE→FIRE→WATER schedule and nominal arrival times. Refill time = 0."""
    arrival_fire: Dict[NodeId, float] = {}
    arrival_water: Dict[int, List[Tuple[NodeId, float]]] = {k: [] for k in chrom}
    sequence: Dict[int, List[NodeId]] = {}

    for k, fires in chrom.items():
        seq = [data.base_id]
        t, loc = 0.0, data.base_id

        for idx, j in enumerate(fires):
            t += travel_time(data, loc, j)
            arrival_fire[j] = t
            seq.append(j)

            if idx < len(fires) - 1:
                nxt = fires[idx + 1]
                w = _best_water_for_next(data, j, nxt)
                t += travel_time(data, j, w)  # refill time = 0
                arrival_water[k].append((w, t))
                seq.append(w)
                loc = w
            else:
                loc = j
        sequence[k] = seq
    return DecodeResult(sequence=sequence, arrival_fire=arrival_fire, arrival_water=arrival_water)

# ---------- Scenario simulator ----------
@dataclass
class NodeStateSim:
    status: int = COLD
    ts: Optional[float] = None
    tm: Optional[float] = None
    te: Optional[float] = None
    beta: Optional[float] = None
    reward: Optional[float] = None  # finalized reward for this scenario

def _tm_te_beta(a_side: float, ts: float, spread_base: float, amel_base: float, s: float, pi: float):
    """Scenario scaling: both rates multiplied by (1+s)."""
    spread_eff = spread_base * (1.0 + s)
    amel_eff   = amel_base   * (1.0 + s)
    d_sm = a_side / spread_eff
    d_me = a_side / amel_eff
    tm = ts + d_sm
    te = tm + d_me
    beta = pi / (d_sm + d_me)  # linear decay slope
    return tm, te, beta

def simulate_scenario(data: ProblemData, dec: DecodeResult, scen: Scenario) -> float:
    """Event-driven simulation with always-drop rule and water between fires."""
    a_side = math.sqrt(data.node_area)
    st: Dict[NodeId, NodeStateSim] = {j: NodeStateSim() for j in data.Jstar}

    # initial fires at ts = 0
    for j in data.Jstar:
        n = data.nodes[j]
        if n.state == STATE_INIT_FIRE:
            s = st[j]
            s.status = BURNING
            s.ts = 0.0
            s.tm, s.te, s.beta = _tm_te_beta(a_side, 0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value)

    # events: list of (time, type, node_id)
    events: List[Tuple[float, str, NodeId]] = []
    for j, s in st.items():
        if s.status == BURNING:
            events += [(s.tm, "SPREAD", j), (s.te, "BURNOUT", j)]
    for j, a_fire in dec.arrival_fire.items():
        events.append((a_fire, "DROP", j))
    events.sort(key=lambda x: x[0])

    i = 0
    while i < len(events):
        t, ev, j = events[i]
        s = st[j]; n = data.nodes[j]

        if ev == "SPREAD":
            if s.status == BURNING and (s.te is None or t < s.te):
                for nb in n.neighbors:
                    if nb in st:  # only J*
                        sn = st[nb]; nn = data.nodes[nb]
                        # cannot ignite fire-proof; don't re-ignite initial fires
                        if sn.status == COLD and nn.state != STATE_PROOF and nn.state != STATE_INIT_FIRE:
                            sn.status = BURNING
                            sn.ts = t
                            sn.tm, sn.te, sn.beta = _tm_te_beta(a_side, t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value)
                            events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
                events.sort(key=lambda x: x[0])

        elif ev == "BURNOUT":
            if s.status == BURNING:
                s.status = BURNED_SIM
                s.reward = 0.0

        elif ev == "DROP":
            # always drop at j in every scenario; if early, wait until ts
            if s.status == BURNING:
                ts = s.ts if s.ts is not None else t
                t_drop = max(t, ts)
                if s.te is None or t_drop < s.te:
                    s.status = PROCESSED
                    s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
                    # pre-threshold cancellation is implicit: future SPREAD checks status==BURNING
                else:
                    s.reward = 0.0
            elif s.status == COLD:
                # never ignites in this scenario → wasted drop, collect full value
                s.reward = n.initial_value
            # if PROCESSED or BURNED_SIM, ignore (shouldn't reoccur)

        i += 1

    # finalize rewards
    total = 0.0
    for j, s in st.items():
        if s.reward is None:
            if s.status == COLD:
                s.reward = data.nodes[j].initial_value
            elif s.status == BURNED_SIM:
                s.reward = 0.0
        total += (s.reward or 0.0)
    return total

# ---------- Fitness ----------
def expected_reward(data: ProblemData, chrom: Chromosome) -> float:
    dec = decode_routes(data, chrom)
    val = 0.0

    for scen in data.scenarios:
        val += scen.prob * simulate_scenario(data, dec, scen)
    return val


# ---- Detailed simulation for reporting ----
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math

def route_timeline(data: ProblemData, dec: DecodeResult) -> Dict[int, List[Tuple[int, NodeId, str, float]]]:
    """
    Build a per-vehicle visit timeline from decoder output, including BASE/WATER/FIRE nodes
    with arrival times. Returns {veh: [(step_idx, node_id, node_type, arrival_time), ...]}.
    """
    def node_type(nid: NodeId) -> str:
        st = data.nodes[nid].state
        if st == 6: return "base"
        if st == 5: return "water"
        return "fire"

    timeline: Dict[int, List[Tuple[int, NodeId, str, float]]] = {}
    for k, seq in dec.sequence.items():
        steps = []
        t = 0.0
        if seq:
            # first is base
            steps.append((0, seq[0], node_type(seq[0]), 0.0))
            for idx in range(1, len(seq)):
                a, b = seq[idx-1], seq[idx]
                t += travel_time(data, a, b)
                steps.append((idx, b, node_type(b), t))
        timeline[k] = steps
    return timeline

@dataclass
class NodeDetail:
    node_id: NodeId
    scenario_id: str
    had_fire: int          # 1 if ts is not None (fire arrived), else 0
    status: str            # final status: COLD/PROCESSED/BURNED
    ts: Optional[float]
    tm: Optional[float]
    te: Optional[float]
    drop_time: Optional[float]  # actual drop time if dropped while burning, else arrival/wasted
    arrival_time: Optional[float]  # planned arrival from first-stage (for fire nodes only)
    collected_value: float

def simulate_scenario_report(data: ProblemData, dec: DecodeResult, scen: Scenario) -> Tuple[float, Dict[NodeId, NodeDetail]]:
    """
    Run one scenario and return (scenario_total, {node_id: NodeDetail}).
    Mirrors simulate_scenario, but records per-node fields for reporting.
    """
    from .core import STATE_INIT_FIRE, STATE_PROOF, COLD, BURNING, PROCESSED, BURNED_SIM

    a_side = math.sqrt(data.node_area)

    @dataclass
    class S:
        status: int = COLD
        ts: Optional[float] = None
        tm: Optional[float] = None
        te: Optional[float] = None
        beta: Optional[float] = None
        drop_time: Optional[float] = None
        reward: Optional[float] = None

    st: Dict[NodeId, S] = {j: S() for j in data.Jstar}

    def tm_te_beta(ts, sr, ar, s, pi):
        spread_eff = sr * (1.0 + s)
        amel_eff   = ar * (1.0 + s)
        d_sm = a_side / spread_eff
        d_me = a_side / amel_eff
        tm = ts + d_sm
        te = tm + d_me
        beta = pi / (d_sm + d_me)
        return tm, te, beta

    # initial fires
    events: List[Tuple[float, str, NodeId]] = []
    for j in data.Jstar:
        n = data.nodes[j]
        if n.state == STATE_INIT_FIRE:
            st[j].status = BURNING
            st[j].ts = 0.0
            st[j].tm, st[j].te, st[j].beta = tm_te_beta(0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value)
            events += [(st[j].tm, "SPREAD", j), (st[j].te, "BURNOUT", j)]

    # planned drops
    for j, a in dec.arrival_fire.items():
        events.append((a, "DROP", j))
    events.sort(key=lambda x: x[0])

    i = 0
    while i < len(events):
        t, ev, j = events[i]
        s = st[j]; n = data.nodes[j]

        if ev == "SPREAD":
            if s.status == BURNING and (s.te is None or t < s.te):
                for nb in n.neighbors:
                    if nb in st:
                        sn = st[nb]; nn = data.nodes[nb]
                        if sn.status == COLD and nn.state not in (STATE_PROOF, STATE_INIT_FIRE):
                            sn.status = BURNING
                            sn.ts = t
                            sn.tm, sn.te, sn.beta = tm_te_beta(t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value)
                            events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
                events.sort(key=lambda x: x[0])

        elif ev == "BURNOUT":
            if s.status == BURNING:
                s.status = BURNED_SIM
                s.reward = 0.0

        elif ev == "DROP":
            if s.status == BURNING:
                ts = s.ts if s.ts is not None else t
                t_drop = max(t, ts)
                s.drop_time = t_drop
                if s.te is None or t_drop < s.te:
                    s.status = PROCESSED
                    s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
                else:
                    s.reward = 0.0
            elif s.status == COLD:
                s.drop_time = t
                s.reward = data.nodes[j].initial_value
        i += 1

    # finalize & collect node details
    details: Dict[NodeId, NodeDetail] = {}
    total = 0.0
    for j in data.Jstar:
        s = st[j]
        n = data.nodes[j]
        if s.reward is None:
            if s.status == 0:            # COLD
                s.reward = n.initial_value
            elif s.status == 3:          # BURNED_SIM
                s.reward = 0.0
        total += (s.reward or 0.0)

        status_str = "COLD" if s.status == 0 else ("BURNED" if s.status == 3 else ("PROCESSED" if s.status == 2 else "BURNING"))
        details[j] = NodeDetail(
            node_id=j,
            scenario_id=scen.id,
            had_fire=1 if s.ts is not None else 0,
            status=status_str,
            ts=s.ts,
            tm=s.tm,
            te=s.te,
            drop_time=s.drop_time,
            arrival_time=dec.arrival_fire.get(j, None),
            collected_value=float(s.reward or 0.0)
        )
    return total, details

def evaluate_solution_details(data: ProblemData, chrom: Chromosome):
    """
    Build a rich report dict for one solution:
      - decoded plan
      - per-scenario totals and per-node details
      - route timeline (vehicles, steps, node types, arrival times)
    """
    dec = decode_routes(data, chrom)

    per_scen_totals = {}
    per_scen_details = {}
    for scen in data.scenarios:
        tot, det = simulate_scenario_report(data, dec, scen)
        per_scen_totals[scen.id] = tot
        per_scen_details[scen.id] = det

    timeline = route_timeline(data, dec)
    return {
        "decoder": dec,
        "per_scen_totals": per_scen_totals,
        "per_scen_details": per_scen_details,
        "timeline": timeline
    }


def route_time_with_return(data, seq):
    """Sum travel time along seq, including return from last node to base."""
    if not seq:
        return 0.0
    t = 0.0
    for a, b in zip(seq, seq[1:]):
        t += travel_time(data, a, b)
    # ensure return to base
    last = seq[-1]
    if last != data.base_id:
        t += travel_time(data, last, data.base_id)
    return t

def makespan_from_decoded(data, dec):
    """Max per-vehicle route time with return to base."""
    return max((route_time_with_return(data, seq) for seq in dec.sequence.values()), default=0.0)





def simulate_scenario_with_ignited(data: ProblemData, dec: DecodeResult, scen: Scenario) -> Tuple[float, set]:
    """
    Same logic as simulate_scenario but ALSO returns the set of nodes that ever ignite
    (i.e., are BURNING at some time) under THIS scenario WITH the planned drops in `dec`.
    """
    a_side = math.sqrt(data.node_area)
    st: Dict[NodeId, NodeStateSim] = {j: NodeStateSim() for j in data.Jstar}
    ignited = set()

    # initial fires at ts = 0
    for j in data.Jstar:
        n = data.nodes[j]
        if n.state == STATE_INIT_FIRE:
            s = st[j]
            s.status = BURNING
            s.ts = 0.0
            s.tm, s.te, s.beta = _tm_te_beta(a_side, 0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value)
            ignited.add(j)

    # events: list of (time, type, node_id)
    events: List[Tuple[float, str, NodeId]] = []
    for j, s in st.items():
        if s.status == BURNING:
            events += [(s.tm, "SPREAD", j), (s.te, "BURNOUT", j)]
    for j, a_fire in dec.arrival_fire.items():
        events.append((a_fire, "DROP", j))
    events.sort(key=lambda x: x[0])

    i = 0
    while i < len(events):
        t, ev, j = events[i]
        s = st[j]; n = data.nodes[j]

        if ev == "SPREAD":
            if s.status == BURNING and (s.te is None or t < s.te):
                for nb in n.neighbors:
                    if nb in st:
                        sn = st[nb]; nn = data.nodes[nb]
                        if sn.status == COLD and nn.state != STATE_PROOF and nn.state != STATE_INIT_FIRE:
                            sn.status = BURNING
                            sn.ts = t
                            sn.tm, sn.te, sn.beta = _tm_te_beta(a_side, t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value)
                            ignited.add(nb)
                            events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
                events.sort(key=lambda x: x[0])

        elif ev == "BURNOUT":
            if s.status == BURNING:
                s.status = BURNED_SIM
                s.reward = 0.0

        elif ev == "DROP":
            if s.status == BURNING:
                ts = s.ts if s.ts is not None else t
                t_drop = max(t, ts)
                if s.te is None or t_drop < s.te:
                    s.status = PROCESSED
                    s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
                else:
                    s.reward = 0.0
            elif s.status == COLD:
                s.reward = n.initial_value
        i += 1

    # finalize rewards (to return scenario total alongside ignited set)
    total = 0.0
    for j, s in st.items():
        if s.reward is None:
            if s.status == COLD:
                s.reward = data.nodes[j].initial_value
            elif s.status == BURNED_SIM:
                s.reward = 0.0
        total += (s.reward or 0.0)

    return total, ignited


def union_ignited_over_scenarios(data: ProblemData, dec: DecodeResult) -> set:
    """
    Union of nodes that ignite in at least one scenario under THIS plan (with drops).
    """
    U = set()
    for scen in data.scenarios:
        _, ign = simulate_scenario_with_ignited(data, dec, scen)
        U |= ign
    return U










def value_and_makespan(data, chrom):
    """
    Prune dry nodes (never ignite in any scenario under this plan),
    re-decode, then return (expected_value, makespan, chrom_pruned, dec).
    """
    # first decode & find ignited union B
    dec0 = decode_routes(data, chrom)
    B = union_ignited_over_scenarios(data, dec0)
    # prune
    chrom_pruned = {k: [j for j in lst if j in B] for k, lst in chrom.items()}
    # re-decode and evaluate
    dec = decode_routes(data, chrom_pruned)
    val = 0.0
    for scen in data.scenarios:
        val += scen.prob * simulate_scenario(data, dec, scen)
    mk = makespan_from_decoded(data, dec)
    return val, mk, chrom_pruned, dec
