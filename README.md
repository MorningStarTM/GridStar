# GridStar

Neural-guided A* search for power grid topology optimization. Learns a heuristic cost-to-recovery function h(s) to efficiently find optimal remedial topology actions during congestion, then distills the search results into a deployable policy network via behavioral cloning. Built on Grid2Op/L2RPN.

---

## Motivation

Power grid congestion management is a critical challenge — grid operators must reconfigure network topology (bus switching at substations) to prevent line overflows and blackouts. The action space is exponential (2^k possible topologies), making exhaustive search infeasible.

**Prior work:**
- **enliteAI (WCCI 2022, 1st place):** AlphaZero-style MCTS + policy network. Achieved 82% step survival, 60% redispatching cost reduction. But MCTS wastes simulations on random rollouts, has no optimality guarantee, and can't learn from historical data.
- **Retro\* (ICML 2020):** Neural-guided A\* on AND-OR trees for retrosynthesis. 86.8% success vs MCTS's 33.7%. But AND-OR decomposition doesn't apply to grid (actions aren't independent).

**GridStar** takes the core insight from Retro\* — best-first search with a learned heuristic — and adapts it for power grid topology, without requiring AND-OR decomposition.

---

## Core Idea

### Why A* over MCTS for Grid?

| | MCTS | A* (GridStar) |
|---|---|---|
| **Expansion** | Full root→leaf path each simulation | Single cheapest frontier node |
| **simulate() calls** | Many per simulation (path traversal) | 1 per expansion step |
| **Optimality** | No guarantee | Guaranteed (if h admissible) |
| **Exploration** | Built-in (UCB/PUCT balancing) | Greedy (relies on good h) |
| **Learns from history** | No (self-play from scratch) | Yes (offline training of h) |
| **Neural net role** | Prior P(a) for PUCT | Cost heuristic h(s) + policy π(s) for pruning |

A* directly optimizes for "find the cheapest path to a safe grid state" — no rollouts, no visit count statistics. Every simulate() call is purposeful.

---

## Algorithm Design

### Problem Framing as A*

```
f(s) = g(s) + h(s)

g(s) = accumulated overflow cost from congestion start to state s (known)
h(s) = neural network estimate of remaining cost to reach safe state (learned)
```

**START state:** Grid observation where ρ_max ≥ 0.98 (Grid State Observer triggers)

**GOAL state:** Grid observation where ρ_max < 0.98 AND the state remains safe for K steps under do-nothing (sustained recovery, not temporary relief)

**NODE:** (grid2op observation, timestep) — each node stores the full observation object required for simulate()

**EDGE:** One unitary topology action (bus reconfiguration at one substation)

### g(s) — Known Cost

Cumulative overflow cost along the path from start to current node:

```
g(s_t) = g(s_{t-1}) + cost(s_{t-1}, a, s_t)

cost(s, a, s') = max(ρ_max(s') - 0.98, 0) + 0.5 * n_offline_lines(s') + 0.01 * (a != do_nothing)
```

- Zero when grid is safe (ρ < 0.98)
- Increases with overflow severity
- Small penalty per topology change (prefer fewer interventions)

### h(s) — Neural Heuristic (the core research contribution)

A neural network that predicts "how much cost remains to reach a safe state from observation s":

```
h(s) = HeuristicNet(obs_to_vector(s)) → scalar
```

**Admissibility:** If h(s) never overestimates the true cost, A\* guarantees finding the optimal action sequence. In practice, approximate admissibility is sufficient — the goal is generating good training data for the policy, not provably optimal search.

**Training signal:** For every node visited during A\* search, the actual cost-to-recovery is known after the search completes. This gives supervised (obs, true_cost) pairs.

### Goal Test — Neural Safety Predictor

A\* needs a cheap goal test. Testing "stays safe for K steps" naively requires K simulate() calls per candidate node. Instead, we train a binary classifier:

```
SafetyPredictor(obs) → P(state remains safe for K steps)
```

- **Training data:** Run episodes, record every state and whether it stayed safe for K more steps. Simple binary classification — no search needed.
- **Usage:** Fast filter during A\* (one forward pass per candidate). Only verify the final winner with real simulate() calls.

### Action Pruning — Policy Network

Standard A\* expands ALL children of the best node. With 500-2000 topology actions, that's 500-2000 simulate() calls per expansion. Infeasible.

**Solution:** A policy network π(s) predicts which actions are promising. Only expand the **top-K actions** from π(s):

```
children = top_K(π(obs), K=20)
for action in children:
    child_obs = obs.simulate(action)
    f_child = g + cost(obs, action, child_obs) + h(child_obs)
    add child to open set
```

This policy network is also what gets **deployed at inference** — the primary output of the whole system.

---

## Architecture

### Three Networks

| Network | Input | Output | Purpose |
|---------|-------|--------|---------|
| **h(s)** — Heuristic | obs vector | scalar cost | Guide A\* search (which node to expand) |
| **π(s)** — Policy | obs vector | action distribution | Prune A\* branches (top-K) + **deployed at inference** |
| **SafetyPredictor** | obs vector | P(safe for K steps) | Cheap goal test during A\* |

### Search Procedure

```
function GridStarSearch(congested_obs):
    open_set = PriorityQueue()            # sorted by f = g + h
    root = Node(obs=congested_obs, g=0, h=h_net(congested_obs))
    open_set.push(root)

    while open_set not empty:
        node = open_set.pop()             # lowest f(s)

        if SafetyPredictor(node.obs) > 0.9:
            if verify_with_simulate(node.obs, K=10):
                return reconstruct_path(node)     # found goal

        top_actions = top_K(π(node.obs), K=20)    # policy prunes actions
        for action in top_actions:
            child_obs, reward, done, info = node.obs.simulate(action)
            if done or info["is_illegal"]:
                continue
            child_g = node.g + cost(node.obs, action, child_obs)
            child_h = h_net(child_obs)
            child = Node(obs=child_obs, g=child_g, h=child_h, parent=node)
            open_set.push(child)

    return None  # no solution found within budget
```

---

## Training Loop

### Phase 1: Bootstrap (cold start)

Use existing agents (PPO, gridGPT, or brute-force search) to collect initial training data:

```
For each congestion state encountered:
    Record (obs, best_action, cost_to_recovery)
```

Train initial h(s) and π(s) on this data.

### Phase 2: Self-Improving Loop

```
repeat for N iterations:
    1. COLLECT: Run Grid2Op episodes on training chronics
       - When congestion hits, run A* with current h(s) and π(s)
       - A* finds path to recovery

    2. EXTRACT training signal from A* results:
       - For π: at root, record action distribution from A*
         p(a) ∝ exp(-f(a) / temperature)
       - For h: for every visited node, record actual cost-to-recovery

    3. TRAIN:
       - π(s): cross-entropy loss against A*'s action distribution
       - h(s): MSE loss against actual costs
       - SafetyPredictor: BCE on (obs, survived_K_steps) pairs

    4. EVALUATE on held-out chronics → track improvement
```

As h improves → A\* searches more efficiently → finds better paths → generates better training data → h and π improve further. Same virtuous cycle as AlphaZero's self-play, but with A\* instead of MCTS.

### Phase 3: Deployment

At inference, **no A\* search is run**. Only the trained policy network:

```
function deploy(obs):
    if ρ_max < 0.98:
        return do_nothing

    top_actions = top_K(π(obs), K=5)
    for action in top_actions:
        sim_obs = obs.simulate(action)
        if sim_obs.rho.max() < 0.98:
            return action                # pick first action that resolves congestion

    return top_actions[0]                # fallback to best policy prediction
```

---

## Key Design Decisions

### Why not AND-OR tree (like Retro*)?

Retrosynthesis decomposes naturally: to make molecule M, you need reactants A AND B, each solvable independently. Grid topology has **no such decomposition** — switching bus at substation 5 affects line loads everywhere. We use plain A\* on a flat search tree instead.

### Why not learned value function (like AlphaZero)?

enliteAI tried a learned value network and found it **underperformed their hand-crafted heuristic** in their low-compute regime. Our h(s) is a middle ground — learned from data (generalizes better than hand-crafted) but trained offline with supervised learning (more stable than RL-based value learning).

### Temporal dynamics

Grid loads change every 5 minutes regardless of agent actions. Grid2Op's simulate() handles this using load forecasts. We limit search depth (D=5 steps = 25 minutes) because forecast error compounds. Beyond depth D, h(s) handles the rest — it implicitly encodes temporal patterns from training data.

### Branching factor

Even with policy pruning to top-K=20 actions, depth 5 gives 20^5 = 3.2M potential nodes. A\* won't expand all of them — a good h(s) prunes most branches. But we also set a simulation budget (max expansions) as a hard limit.

---

## Comparison with Prior Work

| | enliteAI (WCCI 2022) | GridStar (ours) |
|---|---|---|
| **Search algorithm** | MCTS (AlphaZero) | A\* with learned heuristic |
| **Value function** | Hand-crafted heuristic | Learned h(s) from data |
| **Policy training** | Behavioral cloning from MCTS | Behavioral cloning from A\* |
| **simulate() efficiency** | ~300 calls per congestion event | Potentially far fewer (best-first) |
| **Optimality** | No guarantee | Guaranteed if h admissible |
| **Learns from history** | No | Yes (offline h training) |
| **Action pruning** | Top-25 from policy | Top-K from policy (same idea) |
| **Temporal modeling** | No memory in policy | Can use sequential architectures |

---

## Project Structure

```
GridStar/
├── gridstar/
│   ├── search/
│   │   ├── astar.py              # Neural-guided A* search
│   │   ├── node.py               # Search node (obs, g, h, parent, action)
│   │   └── goal.py               # Goal test (brute-force + neural safety predictor)
│   ├── networks/
│   │   ├── heuristic.py          # h(s): cost-to-recovery estimator
│   │   ├── policy.py             # π(s): deployable policy + A* action pruner
│   │   └── safety.py             # Binary classifier for sustained safety
│   ├── env/
│   │   ├── grid_env.py           # Grid2Op environment wrapper
│   │   ├── reward.py             # g(s) cost functions
│   │   └── actions.py            # Action space reduction and filtering
│   ├── training/
│   │   ├── data_collector.py     # Collect trajectories from A* episodes
│   │   ├── trainer.py            # Self-improving training loop
│   │   └── bootstrap.py          # Bootstrap from existing agents
│   ├── utils/
│   │   ├── obs.py                # Observation processing
│   │   └── visualize.py          # Search tree visualization
│   └── config.py                 # Configuration
├── tests/
├── notebooks/
├── configs/default.yaml
├── checkpoints/
├── data/
├── main.py                       # Entry point
├── evaluate.py                   # Evaluate trained policy
└── requirements.txt
```

---

## Open Research Questions

1. **h(s) architecture:** MLP vs GNN vs Transformer — which best captures grid structure for cost prediction?
2. **Admissibility vs accuracy:** How to balance underestimation (safe but slow) vs overestimation (fast but suboptimal)?
3. **World model integration:** Can a learned world model (GridFormer) replace simulate() inside A\* for faster search?
4. **Scale:** Will this work on 118-bus grids (72,958 actions) or only case14 (178 actions)?
5. **Joint topology + redispatching:** Can A\* search over combined action spaces?

---

## References

- Dorfer et al., "Power Grid Congestion Management via Topology Optimization with AlphaZero" (NeurIPS 2022 RL4RealLife)
- Chen et al., "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search" (ICML 2020)
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
- Donnot, "Grid2Op - A testbed platform to model sequential decision making in power systems" (2020)
