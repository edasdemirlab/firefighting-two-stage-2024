import random, time
import statistics
from typing import Dict, List, Tuple
from .core import NodeId, ProblemData
from .model import Chromosome, expected_reward
from .core import STATE_INIT_FIRE
from .model import travel_time
from ga.model import Chromosome, value_and_makespan

# ---------------------------
# Helpers for subset encoding
# ---------------------------

def init_random_subset(candidates: List[NodeId], K: int, rng: random.Random,
                       coverage_ratio: float = 0.6) -> Chromosome:
    """
    Build a chromosome that covers only a subset of eligible nodes (candidates = data.Jstar).
    coverage_ratio in (0,1]: fraction of candidates to include overall.
    Nodes are distributed round-robin across vehicles.
    """
    nodes = candidates[:]
    rng.shuffle(nodes)
    total = max(0, min(len(nodes), int(round(len(nodes) * coverage_ratio))))
    picked = nodes[:total]

    chrom: Chromosome = {k: [] for k in range(K)}
    vi = 0
    for j in picked:
        chrom[vi].append(j)
        vi = (vi + 1) % K
    return chrom

def mutate_swap(ch: Chromosome, rng: random.Random, p: float = 0.2):
    """Swap two positions within a random vehicle (order mutation)."""
    if rng.random() < p and ch:
        k = rng.choice(list(ch.keys()))
        if len(ch[k]) >= 2:
            i, j = rng.sample(range(len(ch[k])), 2)
            ch[k][i], ch[k][j] = ch[k][j], ch[k][i]

def mutate_add_or_remove(ch: Chromosome,
                         candidates: List[NodeId],
                         rng: random.Random,
                         p_add: float = 0.15,
                         p_remove: float = 0.15):
    """
    Randomly add a candidate not present, or remove a present one.
    Ensures we never add duplicates across vehicles.
    """
    # current set
    present = set()
    for lst in ch.values():
        present.update(lst)

    # remove: drop a random node from a random vehicle
    if rng.random() < p_remove and present:
        k = rng.choice(list(ch.keys()))
        if ch[k]:
            idx = rng.randrange(len(ch[k]))
            ch[k].pop(idx)

    # add: append a missing candidate to a random vehicle
    if rng.random() < p_add:
        missing = [j for j in candidates if j not in present]
        if missing:
            j_add = rng.choice(missing)
            k = rng.choice(list(ch.keys()))
            ch[k].append(j_add)

def crossover_one_point(p1: Chromosome, p2: Chromosome, rng: random.Random):
    """
    Per-vehicle one-point crossover with safe fallbacks:
    - if one side empty: copy
    - if min(len(a), len(b)) < 2: use append-unique (no cut)
    After building children, dedupe across vehicles.
    """
    c1, c2 = {}, {}
    for k in p1.keys():
        a, b = p1[k][:], p2[k][:]
        if not a or not b:
            # if one is empty, just copy the other
            c1[k], c2[k] = a, b
            continue

        m = min(len(a), len(b))
        if m < 2:
            # singleton case: avoid randrange(1,1) -> use append-unique
            c1[k] = a + [x for x in b if x not in a]
            c2[k] = b + [x for x in a if x not in b]
        else:
            cut = rng.randrange(1, m)  # 1 .. m-1
            c1[k] = a[:cut] + [x for x in b if x not in a[:cut]]
            c2[k] = b[:cut] + [x for x in a if x not in b[:cut]]

    # ensure uniqueness across all vehicles
    c1 = dedupe_chromosome(c1)
    c2 = dedupe_chromosome(c2)
    return c1, c2

def dedupe_chromosome(ch: Chromosome) -> Chromosome:
    """
    Enforce that no node appears in more than one vehicle.
    Keeps the first occurrence in vehicle order 0..K-1, removes later duplicates.
    """
    seen = set()
    fixed: Chromosome = {}
    for k in sorted(ch.keys()):
        lst = []
        for j in ch[k]:
            if j not in seen:
                lst.append(j)
                seen.add(j)
        fixed[k] = lst
    return fixed



def _nearest_neighbor_order(data: ProblemData, start_node: NodeId, pool: List[NodeId]) -> List[NodeId]:
    """Greedy order: from current node, go to nearest next node by travel_time."""
    remaining = set(pool)
    order = []
    cur = start_node
    while remaining:
        nxt = min(remaining, key=lambda j: travel_time(data, cur, j))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order

def build_seed_initial_fire_only(data: ProblemData, rng: random.Random) -> Chromosome:
    """
    Seed chromosome that visits ONLY the initial-fire nodes, distributed across vehicles,
    with per-vehicle nearest-neighbor ordering from BASE.
    """
    base = data.base_id
    init_fires = [j for j in data.Jstar if data.nodes[j].state == STATE_INIT_FIRE]
    rng.shuffle(init_fires)  # slight randomness across seeds

    # round-robin distribute
    buckets: Dict[int, List[NodeId]] = {k: [] for k in range(data.n_vehicles)}
    for i, j in enumerate(init_fires):
        buckets[i % data.n_vehicles].append(j)

    # order each bucket by NN from base
    chrom: Chromosome = {k: [] for k in range(data.n_vehicles)}
    for k, lst in buckets.items():
        if lst:
            chrom[k] = _nearest_neighbor_order(data, base, lst)
    return chrom







# ---------------------------
# GA engine (subset-capable)
# ---------------------------

def run_ga(data: ProblemData,
           pop_size=40, p_cross=0.9,
           p_mut_swap=0.2, p_mut_add=0.15, p_mut_remove=0.15,
           max_generations=400,
           seed=42, time_limit_sec=600,
           coverage_ratio=0.6, heuristic_seed_ratio=0.1,
           elite_ratio=0.1, stall_ratio=0.2,
           track_history: bool = True):
    """
    Subset-capable GA with lexicographic objective:
      - Primary: maximize expected collected value
      - Secondary (tie-break): minimize makespan (max per-vehicle time incl. return to base)
    Evaluation uses value_and_makespan(), which also prunes dry nodes (never-ignite under plan).
    """
    import statistics
    EPS = 1e-9  # near-tie tolerance on value

    rng = random.Random(seed)
    t0 = time.time()

    elite = max(1, int(round(pop_size * elite_ratio)))
    stall_limit = max(1, int(round(max_generations * stall_ratio)))

    candidates = data.Jstar  # all eligible fire-prone nodes

    # ---- helpers ----
    def evaluate_pair(ch: Chromosome):
        # returns (chrom_pruned, (value, makespan))
        val, mk, chp, _ = value_and_makespan(data, ch)
        return chp, (val, mk)

    def better(a, b):
        """Return True if a is lexicographically better than b."""
        (va, mka) = a[1]
        (vb, mkb) = b[1]
        if va > vb + EPS:
            return True
        if abs(va - vb) <= EPS and mka < mkb:
            return True
        return False

    # ---- init population ----
    pop: List[Tuple[Chromosome, Tuple[float, float]]] = []

    n_seeds = int(round(pop_size * heuristic_seed_ratio))
    if heuristic_seed_ratio > 0 and n_seeds == 0:
        n_seeds = 1

    # Heuristic seeds: initial-fire-only, nearest-neighbor per vehicle
    for _ in range(min(n_seeds, pop_size)):
        ch = build_seed_initial_fire_only(data, rng)
        ch = dedupe_chromosome(ch)
        chp, pair = evaluate_pair(ch)
        pop.append((chp, pair))

    # Fill remainder with random subset individuals
    while len(pop) < pop_size:
        ch = init_random_subset(candidates, data.n_vehicles, rng, coverage_ratio=coverage_ratio)
        ch = dedupe_chromosome(ch)
        chp, pair = evaluate_pair(ch)
        pop.append((chp, pair))

    # Lexicographic sort: highest value first, then lowest makespan
    pop.sort(key=lambda x: (-x[1][0], x[1][1]))
    best, stall, gen = pop[0], 0, 0


    # ---- history ----
    history = []  # list of dicts per generation

    def _record(pop_list, gen_idx):
        if not track_history:
            return
        vals = [p[1][0] for p in pop_list]
        mks  = [p[1][1] for p in pop_list]
        history.append({
            "generation": gen_idx,
            "fitness": vals,                    # expected value for each individual
            "makespan": mks,                    # makespan for each individual
            "best": max(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p10": statistics.quantiles(vals, n=10)[0],
            "p90": statistics.quantiles(vals, n=10)[-1],
            "best_makespan": min(mks),
        })

    _record(pop, gen)

    # ---- evolutionary loop ----
    while gen < max_generations and (time.time() - t0) < time_limit_sec and stall < stall_limit:
        gen += 1
        # elitism: carry top 'elite' as-is
        newpop: List[Tuple[Chromosome, Tuple[float, float]]] = pop[:elite]

        while len(newpop) < pop_size:
            p1 = rng.choice(pop)[0]
            p2 = rng.choice(pop)[0]

            # crossover
            if rng.random() < p_cross:
                c1, c2 = crossover_one_point(p1, p2, rng)
            else:
                c1 = {k: v[:] for k, v in p1.items()}
                c2 = {k: v[:] for k, v in p2.items()}

            # mutations (order + add/remove) and dedupe
            mutate_swap(c1, rng, p_mut_swap)
            mutate_add_or_remove(c1, candidates, rng, p_add=p_mut_add, p_remove=p_mut_remove)
            c1 = dedupe_chromosome(c1)

            mutate_swap(c2, rng, p_mut_swap)
            mutate_add_or_remove(c2, candidates, rng, p_add=p_mut_add, p_remove=p_mut_remove)
            c2 = dedupe_chromosome(c2)

            # evaluate children (lexicographic pair)
            chp1, pair1 = evaluate_pair(c1)
            newpop.append((chp1, pair1))
            if len(newpop) < pop_size:
                chp2, pair2 = evaluate_pair(c2)
                newpop.append((chp2, pair2))

        # lexicographic sort
        newpop.sort(key=lambda x: (-x[1][0], x[1][1]))

        if gen % 5 == 0 or gen == 1 or gen == max_generations:
            best_val, best_mk = newpop[0][1]
            print(f"[Gen {gen:4d}] best={best_val:.2f}, makespan={best_mk:.2f}, stall={stall}")

        # improvement check (lexicographic with epsilon on value)
        if better(newpop[0], best):
            best, stall = newpop[0], 0
        else:
            stall += 1

        pop = newpop
        _record(pop, gen)

    # figure out stop reason
    elapsed = time.time() - t0
    if gen >= max_generations:
        stop_reason = "max_generations"
    elif elapsed >= time_limit_sec:
        stop_reason = "time_limit"
    elif stall >= stall_limit:
        stop_reason = "stall_no_improvement"
    else:
        stop_reason = "terminated"

    best_chrom = best[0]
    best_val, best_mk = best[1]
    return {
        "best": (best_chrom, best_val),
        "best_makespan": best_mk,
        "generations": gen,
        "stall": stall,
        "stop_reason": stop_reason,  # <<< NEW
        "elapsed_sec": elapsed,  # <<< NEW (raw wall time from run_ga)
        "history": history
    }

