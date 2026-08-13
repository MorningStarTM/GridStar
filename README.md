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

## Usage

### Environment

```python
from gridstar.grid_env.grid_env import GridStarEnv

env = GridStarEnv(
    env_name="l2rpn_case14_sandbox",
    thermal_limit=0.98,          # ρ threshold for congestion / goal test
)

obs = env.reset()
obs = env.advance_to_congestion(obs, max_steps=2000)   # fast-forward to a hot grid

print(env.action_size)          # total actions (1 do-nothing + topology)
print(env.obs_dim)              # flat observation vector length
print(env.get_rho_max(obs))     # worst-line loading (float)
print(env.is_congested(obs))    # True when ρ_max ≥ thermal_limit

obs_vector = env.obs_to_vector(obs)   # numpy float32 array, shape (obs_dim,)
```

---

### Action Space

#### Global — `ActionConverter`

Builds and indexes every unitary topology action across all substations.

```python
from gridstar.grid_env.actions import ActionConverter

converter = ActionConverter(env.env)          # pass the raw grid2op env

print(converter.n)                            # total topology actions
action = converter.act(42)                    # index → grid2op action object
idx    = converter.action_idx(action)         # grid2op action → index

# Cluster-based action mask: only allow actions for substations [0, 3, 5]
mask = converter.get_cluster_mask([0, 3, 5]) # bool array, shape (n_actions+1,)
valid_indices = converter.masked_action_indices([0, 3, 5])

# Apply mask to policy logits before softmax
import torch
from gridstar.grid_env.actions import ActionConverter
logits = torch.randn(len(converter.actions))
masked = ActionConverter.apply_action_mask(logits, mask)  # fill=-1e9 on blocked
```

#### Cluster-local — `MADiscActionConverter`

Used in multi-agent settings where each agent owns a subset of substations.

```python
from gridstar.grid_env.actions import MADiscActionConverter

sub_list = [0, 3, 5]                              # substations owned by this agent
ma_conv  = MADiscActionConverter(env.env, sub_list)

print(ma_conv.action_size())                       # actions for this cluster only
action   = ma_conv.act(0)                          # local index → grid2op action
```

---

### Substation Clustering

`ClusterUtils` applies the **Louvain** community-detection algorithm to the grid's
adjacency matrix and partitions substations into geographically coherent groups.

```python
from gridstar.grid_env.actions import ClusterUtils

# {agent_id: [sub_id, sub_id, …]}
clusters = ClusterUtils.cluster_substations(env.env)
print(clusters)
# e.g. {0: [0, 1, 4, 5], 1: [2, 3, 6], 2: [7, 8, 9, …]}

# Count topology actions available to one cluster
n_actions = ClusterUtils.cluster_action_count(clusters[0], env.env.action_space)

# Connectivity matrix (n_sub × n_sub)
matrix = ClusterUtils.create_connectivity_matrix(env.env)
```

---

### A\* Search — Random Policy (baseline)

`RandomPolicy` uses `h = 0` (admissible; reduces A\* to uniform-cost search) and
selects candidate actions by random sampling. Useful for verifying the search loop
and generating bootstrap training data before any network is trained.

```python
from gridstar.networks.policy import RandomPolicy
from gridstar.search.astar import AStarSearch
from gridstar.utils.visualize import plot_search_tree

policy   = RandomPolicy(n_actions=env.action_size, seed=42)
searcher = AStarSearch(
    env=env,
    policy=policy,
    top_k=5,             # candidate actions expanded per node
    max_expansions=300,  # hard node budget
    max_depth=8,         # maximum search depth
    thermal_limit=0.98,
)

result = searcher.search(obs)

print(result.found)           # True / False
print(result.n_expanded)      # nodes popped from the heap
print(result.n_generated)     # nodes pushed onto the heap
print(len(result.all_nodes))  # total nodes in the search tree

if result.found:
    for node in result.path:
        print(f"depth={node.depth}  action={node.action_idx}  "
              f"ρ_max={node.obs.rho.max():.4f}  g={node.g:.4f}")
```

**Edge cost** (how `g` accumulates at each step):

```
cost(s, a, s') = max(ρ_max(s') − 0.98, 0)   # congestion severity
               + 0.5 × n_offline_lines(s')    # line-disconnection penalty
               + 0.01 × (a ≠ do-nothing)      # action complexity
```

---

### Reward Function

`RewardFunction` is the single source of truth for both the A\* edge cost and the
RL training reward — the same coefficient values drive both, so tuning one tunes both.

```python
from gridstar.grid_env.reward import RewardFunction

rf = RewardFunction(
    thermal_limit=0.98,
    overflow_coeff=1.0,      # weight on per-line quadratic overflow penalty
    margin_coeff=0.1,        # weight on per-line safety-margin reward
    offline_coeff=0.5,       # penalty per disconnected line
    action_coeff=0.01,       # flat penalty for non-do-nothing actions
    goal_bonus=1.0,          # reward when ρ_max < threshold
    blackout_penalty=-10.0,  # reward when episode ends in blackout
)

# RL reward (maximise during training)
obs2, _, done, _ = env.step(action_idx=7)
r = rf(obs2, action_idx=7, done=done)

# A* edge cost (minimise during search)  — called internally by AStarSearch
cost = rf.edge_cost(obs2, action_idx=7, do_nothing_idx=0)

# Pass a custom reward function to the searcher
searcher = AStarSearch(env=env, policy=policy, reward_fn=rf)
```

**Reward components:**

| Component | Formula | Notes |
|---|---|---|
| Overflow penalty | `−Σ max(ρᵢ − θ, 0)²` | Quadratic; penalises all overloaded lines |
| Safety margin | `+mean max(θ − ρᵢ, 0)` | Rewards headroom, not just staying under limit |
| Offline lines | `−0.5 × n_offline` | Disconnected lines reduce grid redundancy |
| Action penalty | `−0.01` if non-trivial | Discourages unnecessary topology changes |
| Goal bonus | `+1.0` if ρ_max < θ | Sparse success signal |
| Blackout penalty | `−10.0` if done | Sparse failure signal |

---

### Heuristic Network

`HeuristicModel` wraps five linear-only backbones under a common interface.
Pass `net=` to select the architecture; all remaining kwargs are forwarded to
the chosen backbone.

```python
from gridstar.networks.heuristic import HeuristicModel

# Create — pick one backbone
h_net = HeuristicModel(obs_dim=env.obs_dim, net='vanilla')    # 3-layer MLP
h_net = HeuristicModel(obs_dim=env.obs_dim, net='efficient')  # 2-layer, fast
h_net = HeuristicModel(obs_dim=env.obs_dim, net='deep',       # residual blocks
                        n_blocks=6, hidden_dim=512)
h_net = HeuristicModel(obs_dim=env.obs_dim, net='wide',       # wide + LayerNorm
                        hidden_dim=512, dropout=0.1)
h_net = HeuristicModel(obs_dim=env.obs_dim, net='dueling')    # overflow + recovery heads

print(h_net)
# HeuristicModel(net='efficient', obs_dim=132, params=18,049)
```

**Available networks:**

| `net=` | Architecture | Parameters (obs=132) | Best for |
|---|---|---|---|
| `vanilla` | 256 → 256 → 128 → 1, BatchNorm | ~132 K | General; large datasets |
| `efficient` | 128 → 64 → 1, no norm | ~18 K | Fast A\* inference |
| `deep` | 256 + 4 residual blocks → 64 → 1, LayerNorm | ~530 K | High accuracy |
| `wide` | 512 → 512 → 1, LayerNorm, LeakyReLU | ~530 K | Broad feature interactions |
| `dueling` | Shared trunk → overflow head + recovery head | ~100 K | Interpretable two-stream |

All backbones end with `nn.Softplus()` so output is always ≥ 0, which is required for
A\* admissibility.

**Training batch:**

```python
import torch
import torch.nn.functional as F

x       = torch.from_numpy(obs_vectors).float()  # (batch, obs_dim)
targets = torch.tensor(costs_to_goal).float().unsqueeze(1)  # (batch, 1)

pred = h_net(x)                                  # (batch, 1)
loss = F.mse_loss(pred, targets)
loss.backward()
```

**Single-observation inference:**

```python
cost_estimate = h_net.predict(obs_vector)         # numpy array → float
```

---

### A\* Search — Neural Heuristic

Plug a trained `HeuristicModel` into the search by wrapping it with
`as_heuristic()`, which converts a raw grid2op observation to a vector before
the forward pass.

```python
from gridstar.networks.policy import NeuralHeuristicPolicy

heuristic_fn = h_net.as_heuristic(env.obs_to_vector)

policy = NeuralHeuristicPolicy(
    n_actions=env.action_size,
    heuristic_fn=heuristic_fn,
    seed=0,
)

searcher = AStarSearch(env=env, policy=policy, top_k=5, max_depth=8)
result   = searcher.search(obs)
```

The `as_heuristic()` wrapper handles the `obs → numpy → tensor → float` pipeline
so `AStarSearch` stays unaware of observation preprocessing.

---

### Search Tree Visualisation

```python
from gridstar.utils.visualize import plot_search_tree

# Save to file
plot_search_tree(result, env, save_path="doc/astar_search_tree.png", max_nodes=200)

# Or display interactively
plot_search_tree(result, env)
```

Node colours:
- **Bright green** — goal node (ρ_max < 0.98, solution found here)
- **Orange** — mild overload (0.98 ≤ ρ < 1.05)
- **Red** — severe overload (ρ ≥ 1.05)

Solution-path edges are drawn in blue; action indices are labelled on those edges only.

---

### Running the Tests

```bash
# Environment + action space sanity checks
python tests/test_grid_env.py

# A* search with random policy + saves search tree PNG to doc/
python tests/test_astar.py
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
