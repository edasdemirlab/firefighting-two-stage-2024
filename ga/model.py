import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
from ga.core import (
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

# def simulate_scenario(data: ProblemData, dec: DecodeResult, scen: Scenario) -> float:
#     """Event-driven simulation with always-drop rule and water between fires."""
#     a_side = math.sqrt(data.node_area)
#     st: Dict[NodeId, NodeStateSim] = {j: NodeStateSim() for j in data.Jstar}
#
#     # initial fires at ts = 0
#     for j in data.Jstar:
#         n = data.nodes[j]
#         if n.state == STATE_INIT_FIRE:
#             s = st[j]
#             s.status = BURNING
#             s.ts = 0.0
#             s.tm, s.te, s.beta = _tm_te_beta(a_side, 0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value)
#
#     # events: list of (time, type, node_id)
#     events: List[Tuple[float, str, NodeId]] = []
#     for j, s in st.items():
#         if s.status == BURNING:
#             events += [(s.tm, "SPREAD", j), (s.te, "BURNOUT", j)]
#     for j, a_fire in dec.arrival_fire.items():
#         events.append((a_fire, "DROP", j))
#     events.sort(key=lambda x: x[0])
#
#     i = 0
#     while i < len(events):
#         t, ev, j = events[i]
#         s = st[j]; n = data.nodes[j]
#
#         if ev == "SPREAD":
#             if s.status == BURNING and (s.te is None or t < s.te):
#                 for nb in n.neighbors:
#                     if nb in st:  # only J*
#                         sn = st[nb]; nn = data.nodes[nb]
#                         # cannot ignite fire-proof; don't re-ignite initial fires
#                         if sn.status == COLD and nn.state != STATE_PROOF and nn.state != STATE_INIT_FIRE:
#                             sn.status = BURNING
#                             sn.ts = t
#                             sn.tm, sn.te, sn.beta = _tm_te_beta(a_side, t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value)
#                             events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
#                 # events.sort(key=lambda x: x[0])
#
#
#         elif ev == "BURNOUT":
#             if s.status == BURNING:
#                 s.status = BURNED_SIM
#                 s.reward = 0.0
#
#
#         elif ev == "DROP":
#             # Drop attempt at planned arrival time.
#             if s.status == BURNING:
#                 ts = s.ts if s.ts is not None else t
#                 t_drop = max(t, ts)   # cannot process before ignition
#                 if s.te is None or t_drop < s.te:
#                     s.status = PROCESSED
#                     s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
#                 else:
#                     # arrived after natural burn-out → no value
#                     s.reward = 0.0
#             elif s.status == COLD:
#                 # Cold drop is ineffective:
#                 # - does NOT finalize reward,
#                 # - does NOT prevent future ignition/spread.
#                 # Reward will be decided at the end:
#                 #   * if never ignites → full value,
#                 #   * if ignites and not processed → 0.
#                 pass
#
#         # elif ev == "DROP":
#         #     # always drop at j in every scenario; if early, wait until ts
#         #     if s.status == BURNING:
#         #         ts = s.ts if s.ts is not None else t
#         #         t_drop = max(t, ts)
#         #         if s.te is None or t_drop < s.te:
#         #             s.status = PROCESSED
#         #             s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
#         #             # pre-threshold cancellation is implicit: future SPREAD checks status==BURNING
#         #         else:
#         #             s.reward = 0.0
#         #     elif s.status == COLD:
#         #         # never ignites in this scenario → wasted drop, collect full value
#         #         s.reward = n.initial_value
#         #     # if PROCESSED or BURNED_SIM, ignore (shouldn't reoccur)
#
#         i += 1
#
#     # finalize rewards
#     total = 0.0
#     for j, s in st.items():
#         if s.reward is None:
#             if s.status == COLD:
#                 s.reward = data.nodes[j].initial_value
#             elif s.status == BURNED_SIM:
#                 s.reward = 0.0
#         total += (s.reward or 0.0)
#     return total

# def simulate_scenario(data: ProblemData, dec: DecodeResult, scen: Scenario) -> float:
#     total, _ = simulate_scenario_report(data, dec, scen)
#     return total

def simulate_scenario(data: ProblemData, dec: DecodeResult, scen: Scenario) -> float:
    """
    Event-driven simulation consistent with simulate_scenario_report:
      - Priority at equal timestamps: SPREAD < DROP < BURNOUT.
      - Cold drop is ineffective (does not finalize reward, does not block future ignition).
      - If processed before te, collect linear-decay value; else 0.
      - Finalization at end: COLD -> full initial value, BURNED -> 0.
    """
    a_side = math.sqrt(data.node_area)

    @dataclass
    class S:
        status: int = COLD
        ts: Optional[float] = None
        tm: Optional[float] = None
        te: Optional[float] = None
        beta: Optional[float] = None
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

    # Initial burning nodes (ts=0)
    events: List[Tuple[float, str, NodeId]] = []
    for j in data.Jstar:
        n = data.nodes[j]
        if n.state == STATE_INIT_FIRE:
            st[j].status = BURNING
            st[j].ts = 0.0
            st[j].tm, st[j].te, st[j].beta = tm_te_beta(
                0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value
            )
            events += [(st[j].tm, "SPREAD", j), (st[j].te, "BURNOUT", j)]

    # Planned drops (arrivals already include waits)
    for j, a in dec.arrival_fire.items():
        events.append((a, "DROP", j))

    EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
    events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

    i = 0
    while i < len(events):
        t, ev, j = events[i]
        s = st[j]; n = data.nodes[j]

        if ev == "SPREAD":
            # Only active burning nodes can spread, and not after te
            if s.status == BURNING and (s.te is None or t < s.te):
                for nb in n.neighbors:
                    if nb in st:
                        sn = st[nb]; nn = data.nodes[nb]
                        if sn.status == COLD and nn.state not in (STATE_PROOF, STATE_INIT_FIRE):
                            sn.status = BURNING
                            sn.ts = t
                            sn.tm, sn.te, sn.beta = tm_te_beta(
                                t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value
                            )
                            events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
                # re-sort with priority after enqueuing new events
                events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

        elif ev == "BURNOUT":
            if s.status == BURNING:
                s.status = BURNED_SIM
                s.reward = 0.0

        elif ev == "DROP":
            if s.status == BURNING:
                ts = s.ts if s.ts is not None else t
                t_drop = max(t, ts)  # cannot process before ignition
                if s.te is None or t_drop < s.te:
                    s.status = PROCESSED
                    # linear decay from ts to te
                    s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
                else:
                    # arrived after natural burn-out → no value
                    s.reward = 0.0
            elif s.status == COLD:
                # Cold drop is ineffective: no reward finalized here, and
                # it does not block a future ignition/spread.
                pass

        i += 1

    # Finalize rewards for nodes that never got a reward assigned
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
    Build a per-vehicle visit timeline from decoder output USING the decoder's arrival stamps.
    We do NOT recompute times by traversing edges, so waits are correctly reflected.

    Returns {veh: [(step_idx, node_id, node_type, arrival_time), ...]}.

    Notes:
      - Base is at time 0.0.
      - Fire nodes' arrival_time comes from dec.arrival_fire[j].
      - Water nodes' arrival_time is taken in order from dec.arrival_water[k] (list of (w, t)).
    """
    def node_type(nid: NodeId) -> str:
        st = data.nodes[nid].state
        if st == 6: return "base"
        if st == 5: return "water"
        return "fire"

    timeline: Dict[int, List[Tuple[int, NodeId, str, float]]] = {}
    for k, seq in dec.sequence.items():
        steps: List[Tuple[int, NodeId, str, float]] = []
        # pointer over this vehicle's water arrivals in decoder
        water_arrs = dec.arrival_water.get(k, [])
        wi = 0

        # step 0: base at t=0
        if seq:
            steps.append((0, seq[0], node_type(seq[0]), 0.0))

        # subsequent steps: use decoder's timestamps
        for idx in range(1, len(seq)):
            nid = seq[idx]
            ntype = node_type(nid)
            if ntype == "fire":
                t = dec.arrival_fire.get(nid, 0.0)
            elif ntype == "water":
                # take next recorded water arrival (in order)
                if wi < len(water_arrs) and water_arrs[wi][0] == nid:
                    t = float(water_arrs[wi][1])
                    wi += 1
                else:
                    # fallback if list desyncs (shouldn't happen): leave 0.0
                    t = 0.0
            else:
                # base should appear only as the first element
                t = 0.0
            steps.append((idx, nid, ntype, t))
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
    from ga.core import STATE_INIT_FIRE, STATE_PROOF, COLD, BURNING, PROCESSED, BURNED_SIM

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
    # events.sort(key=lambda x: x[0])

    EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
    events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

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
                # events.sort(key=lambda x: x[0])
                EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
                events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

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
                # Cold drop: ineffective; do not finalize reward.
                # We can record planned arrival 't' somewhere else if desired,
                # but drop_time stays None for “no processing”.
                pass

        # elif ev == "DROP":
        #     if s.status == BURNING:
        #         ts = s.ts if s.ts is not None else t
        #         t_drop = max(t, ts)
        #         s.drop_time = t_drop
        #         if s.te is None or t_drop < s.te:
        #             s.status = PROCESSED
        #             s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
        #         else:
        #             s.reward = 0.0
        #     elif s.status == COLD:
        #         s.drop_time = t
        #         s.reward = data.nodes[j].initial_value
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

def evaluate_solution_details(data: ProblemData, chrom: Chromosome, dec_override: DecodeResult = None):
    """
    Build a rich report dict for one solution:
      - If dec_override is provided, use it (e.g., with waits already applied).
      - Otherwise decode from chrom (no waits).
    """
    dec = dec_override if dec_override is not None else decode_routes(data, chrom)
    # dec = decode_routes(data, chrom)

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


def makespan_from_decoded_with_waits(data: ProblemData, dec: DecodeResult) -> float:
    """
    Compute max per-vehicle route time including waits (already embedded in dec.arrival_fire),
    plus return to base from the last visited fire node.
    """
    per_k = []
    for k, seq in dec.sequence.items():
        # find last fire node in seq
        last_fire = None
        for nid in reversed(seq):
            st = data.nodes[nid].state
            if st not in (data.STATE_BASE if hasattr(data, "STATE_BASE") else 6,
                          data.STATE_WATER if hasattr(data, "STATE_WATER") else 5):
                # treat non-base, non-water as fire nodes
                last_fire = nid
                break
        if last_fire is None:
            per_k.append(0.0)
        else:
            a_last = dec.arrival_fire.get(last_fire, 0.0)
            per_k.append(a_last + travel_time(data, last_fire, data.base_id))
    return max(per_k) if per_k else 0.0



# def simulate_scenario_with_ignited(data: ProblemData, dec: DecodeResult, scen: Scenario) -> Tuple[float, set]:
#     """
#     Same logic as simulate_scenario but ALSO returns the set of nodes that ever ignite
#     (i.e., are BURNING at some time) under THIS scenario WITH the planned drops in `dec`.
#     """
#     a_side = math.sqrt(data.node_area)
#     st: Dict[NodeId, NodeStateSim] = {j: NodeStateSim() for j in data.Jstar}
#     ignited = set()
#
#     # initial fires at ts = 0
#     for j in data.Jstar:
#         n = data.nodes[j]
#         if n.state == STATE_INIT_FIRE:
#             s = st[j]
#             s.status = BURNING
#             s.ts = 0.0
#             s.tm, s.te, s.beta = _tm_te_beta(a_side, 0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value)
#             ignited.add(j)
#
#     # events: list of (time, type, node_id)
#     events: List[Tuple[float, str, NodeId]] = []
#     for j, s in st.items():
#         if s.status == BURNING:
#             events += [(s.tm, "SPREAD", j), (s.te, "BURNOUT", j)]
#     for j, a_fire in dec.arrival_fire.items():
#         events.append((a_fire, "DROP", j))
#     # events.sort(key=lambda x: x[0])
#     EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
#     events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))
#
#     i = 0
#     while i < len(events):
#         t, ev, j = events[i]
#         s = st[j]; n = data.nodes[j]
#
#         if ev == "SPREAD":
#             if s.status == BURNING and (s.te is None or t < s.te):
#                 for nb in n.neighbors:
#                     if nb in st:
#                         sn = st[nb]; nn = data.nodes[nb]
#                         if sn.status == COLD and nn.state != STATE_PROOF and nn.state != STATE_INIT_FIRE:
#                             sn.status = BURNING
#                             sn.ts = t
#                             sn.tm, sn.te, sn.beta = _tm_te_beta(a_side, t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value)
#                             ignited.add(nb)
#                             events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
#                 # events.sort(key=lambda x: x[0])
#                 EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
#                 events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))
#
#         elif ev == "BURNOUT":
#             if s.status == BURNING:
#                 s.status = BURNED_SIM
#                 s.reward = 0.0
#
#         elif ev == "DROP":
#             if s.status == BURNING:
#                 ts = s.ts if s.ts is not None else t
#                 t_drop = max(t, ts)
#                 if s.te is None or t_drop < s.te:
#                     s.status = PROCESSED
#                     s.reward = max(0.0, n.initial_value - (s.beta or 0.0) * (t_drop - ts))
#                 else:
#                     s.reward = 0.0
#             elif s.status == COLD:
#                 s.reward = n.initial_value
#         i += 1
#
#     # finalize rewards (to return scenario total alongside ignited set)
#     total = 0.0
#     for j, s in st.items():
#         if s.reward is None:
#             if s.status == COLD:
#                 s.reward = data.nodes[j].initial_value
#             elif s.status == BURNED_SIM:
#                 s.reward = 0.0
#         total += (s.reward or 0.0)
#
#     return total, ignited

def simulate_scenario_with_ignited(data: ProblemData, dec: DecodeResult, scen: Scenario) -> Tuple[float, set]:
    """
    Canonical simulator + ignited set:
      - Priority on ties: SPREAD < DROP < BURNOUT
      - Cold drop is ineffective
      - Returns (scenario_total_value, set_of_nodes_that_ever_ignited)
    """
    a_side = math.sqrt(data.node_area)

    @dataclass
    class S:
        status: int = COLD
        ts: Optional[float] = None
        tm: Optional[float] = None
        te: Optional[float] = None
        beta: Optional[float] = None
        reward: Optional[float] = None

    st: Dict[NodeId, S] = {j: S() for j in data.Jstar}
    ignited: set = set()

    def tm_te_beta(ts, sr, ar, s, pi):
        spread_eff = sr * (1.0 + s)
        amel_eff   = ar * (1.0 + s)
        d_sm = a_side / (spread_eff if spread_eff > 0 else float("inf"))
        d_me = a_side / (amel_eff   if amel_eff   > 0 else float("inf"))
        tm = ts + d_sm
        te = tm + d_me
        beta = pi / (d_sm + d_me) if math.isfinite(d_sm + d_me) else 0.0
        return tm, te, beta

    # Initial burning nodes (ts = 0)
    events: List[Tuple[float, str, NodeId]] = []
    for j in data.Jstar:
        n = data.nodes[j]
        if n.state == STATE_INIT_FIRE:
            st[j].status = BURNING
            st[j].ts = 0.0
            st[j].tm, st[j].te, st[j].beta = tm_te_beta(
                0.0, n.spread_rate, n.amel_rate, scen.rate_change, n.initial_value
            )
            ignited.add(j)
            events += [(st[j].tm, "SPREAD", j), (st[j].te, "BURNOUT", j)]

    # Planned drops (arrivals already include any waits)
    for j, a in dec.arrival_fire.items():
        events.append((a, "DROP", j))

    EVENT_PRIORITY = {"SPREAD": 0, "DROP": 1, "BURNOUT": 2}
    events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

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
                            sn.tm, sn.te, sn.beta = tm_te_beta(
                                t, nn.spread_rate, nn.amel_rate, scen.rate_change, nn.initial_value
                            )
                            ignited.add(nb)
                            events += [(sn.tm, "SPREAD", nb), (sn.te, "BURNOUT", nb)]
                # re-sort with priority
                events.sort(key=lambda x: (x[0], EVENT_PRIORITY.get(x[1], 99)))

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
                # Cold drop is ineffective; do not finalize reward here
                pass

        i += 1

    # Finalization
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
    Prune dry nodes, then greedily add per-leg waits ONLY where they improve total expected value.
    Return (expected_value, makespan, chrom_pruned, dec_with_waits, waits).
    """
    dec0 = decode_routes(data, chrom)
    B = union_ignited_over_scenarios(data, dec0)

    chrom_pruned = {k: [j for j in lst if j in B] for k, lst in chrom.items()}
    dec1 = decode_routes(data, chrom_pruned)

    # greedy loitering (returns improved value, waits dict, and the improved decoder)
    val, waits, dec_with_waits = greedy_loitering_improve(data, chrom_pruned, dec1)

    mk = makespan_from_decoded_with_waits(data, dec_with_waits)

    return val, mk, chrom_pruned, dec_with_waits, waits


def value_and_makespan_with_waits(data, chrom, waits):
    """
    Evaluate a completed plan (chrom) with explicit per-leg waits (waits),
    without greedy loitering. We still prune dry nodes first (based on dec0).
    Returns (expected_value, makespan, chrom_pruned, dec_with_waits).
    """
    # 1) Decode w/o waits to determine which nodes ever ignite under this plan; prune dry
    dec0 = decode_routes(data, chrom)
    B = union_ignited_over_scenarios(data, dec0)
    chrom_pruned = {k: [j for j in lst if j in B] for k, lst in chrom.items()}

    # 2) Rebuild waits to match pruned chromosome (drop waits aligned to removed nodes)
    waits_pruned = {k: [] for k in chrom_pruned}
    for k, fires in chrom_pruned.items():
        orig = chrom.get(k, [])
        worig = waits.get(k, [])
        # map original index → wait; keep only those fires that remain
        for j in fires:
            idx = orig.index(j) if j in orig else None
            waits_pruned[k].append((worig[idx] if idx is not None and idx < len(worig) else 0.0))

    # 3) Decode with waits
    decw = decode_routes_with_waits(data, chrom_pruned, waits_pruned)

    # 4) Expected value (no greedy loitering)
    val = 0.0
    for scen in data.scenarios:
        val += scen.prob * simulate_scenario(data, decw, scen)

    # 5) Makespan including waits (return to base)
    mk = makespan_from_decoded_with_waits(data, decw)
    return val, mk, chrom_pruned, decw



#
# def value_and_makespan(data, chrom):
#     """
#     Prune dry nodes (never ignite in any scenario under this plan),
#     re-decode, then return (expected_value, makespan, chrom_pruned, dec).
#     """
#     # first decode & find ignited union B
#     dec0 = decode_routes(data, chrom)
#     B = union_ignited_over_scenarios(data, dec0)
#     # prune
#     chrom_pruned = {k: [j for j in lst if j in B] for k, lst in chrom.items()}
#     # re-decode and evaluate
#     dec = decode_routes(data, chrom_pruned)
#     val = 0.0
#
#     for scen in data.scenarios:
#         val += scen.prob * simulate_scenario(data, dec, scen)
#     mk = makespan_from_decoded(data, dec)
#     return val, mk, chrom_pruned, dec
#

def decode_routes_with_waits(data: ProblemData,
                             chrom: Dict[int, List[NodeId]],
                             waits: Dict[int, List[float]]) -> DecodeResult:
    """
    Same as decode_routes but adds waits[k][idx] BEFORE traveling to chrom[k][idx].
    waits[k] length must equal len(chrom[k]).
    """
    arrival_fire: Dict[NodeId, float] = {}
    arrival_water: Dict[int, List[Tuple[NodeId, float]]] = {k: [] for k in chrom}
    sequence: Dict[int, List[NodeId]] = {}

    for k, fires in chrom.items():
        seq = [data.base_id]
        t, loc = 0.0, data.base_id
        wlist = waits.get(k, [])
        assert len(wlist) == len(fires), "waits[k] must match number of fires in chrom[k]"

        for idx, j in enumerate(fires):
            # wait before leaving current location toward j
            t += float(wlist[idx]) if idx < len(wlist) else 0.0

            # travel to fire
            t += travel_time(data, loc, j)
            arrival_fire[j] = t
            seq.append(j)

            # water hop if another fire remains
            if idx < len(fires) - 1:
                nxt = fires[idx + 1]
                best_w, best_c = None, float("inf")
                for w in data.water_ids:
                    c = travel_time(data, j, w) + travel_time(data, w, nxt)
                    if c < best_c:
                        best_w, best_c = w, c
                if best_w is not None:
                    t += travel_time(data, j, best_w)  # refill time = 0
                    arrival_water[k].append((best_w, t))
                    seq.append(best_w)
                    loc = best_w
                else:
                    loc = j
            else:
                loc = j
        sequence[k] = seq

    return DecodeResult(sequence=sequence, arrival_fire=arrival_fire, arrival_water=arrival_water)



import statistics

def expected_reward_greedy_loitering(data: ProblemData,
                                     chrom: Dict[int, List[NodeId]],
                                     tol: float = 1e-9) -> float:
    """
    Evaluate a chromosome with greedy loitering improvements:
    - Decode with zero waits.
    - In visit order, for each node with an early (wasted) drop:
        Δ = min_{scen with ignition and ts > arrival} (ts - arrival)
      Try inserting Δ as a wait on the leg BEFORE that node (for its vehicle).
      Accept only if total expected value increases; then keep the wait and continue.
    - Return the improved expected value. (Chromosome is NOT modified.)
    """
    # baseline decode with no waits
    waits: Dict[int, List[float]] = {k: [0.0]*len(chrom[k]) for k in chrom}
    dec = decode_routes_with_waits(data, chrom, waits)

    def exp_value(decoder_out: DecodeResult) -> float:
        val = 0.0
        for scen in data.scenarios:
            val += scen.prob * simulate_scenario(data, decoder_out, scen)
        return val

    base_val = exp_value(dec)

    # build visit list (time, veh, idx_in_route, node)
    visits = []
    for k, fires in chrom.items():
        for idx, j in enumerate(fires):
            a = dec.arrival_fire.get(j, float("inf"))
            visits.append((a, k, idx, j))
    visits.sort(key=lambda x: x[0])

    for _, k, idx, j in visits:
        a_j = dec.arrival_fire.get(j)
        if a_j is None or not math.isfinite(a_j):
            continue

        # Collect ignition start times for scenarios where j ignites AND after current arrival
        ts_candidates = []
        for scen in data.scenarios:
            _, details = simulate_scenario_report(data, dec, scen)
            nd = details[j]
            if nd.had_fire == 1 and nd.ts is not None and nd.ts > a_j + 1e-12:
                ts_candidates.append(nd.ts)

        if not ts_candidates:
            continue  # no wasted drop to fix at this node

        Delta = min(ts_candidates) - a_j  # exact minimal wait to catch the earliest ignition
        if Delta <= 0:
            continue

        # Try this single wait on the leg BEFORE j for vehicle k
        waits_try = {kk: vv[:] for kk, vv in waits.items()}
        waits_try[k][idx] += Delta

        dec_try = decode_routes_with_waits(data, chrom, waits_try)
        val_try = exp_value(dec_try)

        if val_try > base_val + tol:
            # accept improvement and continue; downstream arrivals now reflect this wait
            waits = waits_try
            dec = dec_try
            base_val = val_try
        # else: reject and keep current plan

    return base_val



import statistics

def _expected_value_from_decoded(data: ProblemData, dec: DecodeResult) -> float:
    v = 0.0
    for scen in data.scenarios:
        v += scen.prob * simulate_scenario(data, dec, scen)
    return v

def greedy_loitering_improve(data: ProblemData,
                             chrom: Dict[int, List[NodeId]],
                             dec: DecodeResult,
                             tol: float = 1e-9) -> Tuple[float, Dict[int, List[float]], DecodeResult]:
    """
    Greedy local improvement:
      - Visit nodes in planned arrival order.
      - For each node with a wasted drop (arrival < ignition start in at least one scenario that ignites),
        try exactly one wait: Δ = min_{scen with ts > arrival} (ts - arrival).
      - Accept Δ only if total expected value (across scenarios) increases.
      - Keep accepted waits and continue to next node (downstream arrivals shift).

    Returns (improved_value, waits_dict, improved_dec).
    """
    # initialize waits (all zeros)
    waits: Dict[int, List[float]] = {k: [0.0]*len(chrom[k]) for k in chrom}

    base_val = _expected_value_from_decoded(data, dec)

    # Build visit list (arrival_time, veh, idx, node) and sort by arrival
    visits = []
    for k, fires in chrom.items():
        for idx, j in enumerate(fires):
            a = dec.arrival_fire.get(j, float("inf"))
            visits.append((a, k, idx, j))
    visits.sort(key=lambda x: x[0])

    for a_j, k, idx, j in visits:
        if not math.isfinite(a_j):
            continue

        # For current plan (with current waits), recompute scenario details once
        # (We could cache per scenario, but this is still cheap.)
        ts_candidates = []
        for scen in data.scenarios:
            _, details = simulate_scenario_report(data, dec, scen)
            nd = details[j]
            if nd.had_fire == 1 and nd.ts is not None and nd.ts > a_j + 1e-12:
                ts_candidates.append(nd.ts)

        if not ts_candidates:
            continue  # no wasted drop to fix

        Delta = min(ts_candidates) - a_j
        if Delta <= 0:
            continue

        # Try this Δ inserted BEFORE the leg to node j for vehicle k
        waits_try = {kk: vv[:] for kk, vv in waits.items()}
        waits_try[k][idx] += Delta

        dec_try = decode_routes_with_waits(data, chrom, waits_try)
        val_try = _expected_value_from_decoded(data, dec_try)

        if val_try > base_val + tol:
            # accept
            waits = waits_try
            dec = dec_try
            base_val = val_try
        # else reject; keep current waits/dec/base_val

    return base_val, waits, dec
