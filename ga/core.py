from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# Excel node_state codes
STATE_FIRE_PRONE = 0     # without forest fire (fire-prone)
STATE_INIT_FIRE  = 1     # with forest fire (initial ignition)
STATE_RESCUED    = 2
STATE_BURNED     = 3
STATE_PROOF      = 4     # fire-proof
STATE_WATER      = 5     # water source
STATE_BASE       = 6     # base/home

# Internal simulator statuses
COLD, BURNING, PROCESSED, BURNED_SIM = 0, 1, 2, 3

NodeId = int

@dataclass(frozen=True)
class Node:
    id: NodeId
    x: float
    y: float
    initial_value: float
    spread_rate: float          # base rate
    amel_rate: float            # base rate
    state: int                  # 0..6 as above
    neighbors: List[NodeId]     # from neighborhood_list (IDs)

@dataclass(frozen=True)
class Scenario:
    id: str
    prob: float
    rate_change: float          # s in [0,1]; means +s proportion on BOTH rates

@dataclass
class ProblemData:
    nodes: Dict[NodeId, Node]
    base_id: NodeId
    water_ids: List[NodeId]
    Jstar: List[NodeId]         # nodes with state in {0,1}
    scenarios: List[Scenario]
    vehicle_speed: float        # km per time unit
    n_vehicles: int
    node_area: float            # km^2 (a = sqrt(node_area))
