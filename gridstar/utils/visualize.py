from collections import defaultdict, deque

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------

def _node_color(rho_max: float, is_goal_node: bool = False) -> str:
    if is_goal_node:
        return "#2ecc71"        # bright green — solution found here
    if rho_max < 0.98:
        return "#27ae60"        # dark green   — safe but not goal (root was already safe)
    elif rho_max < 1.05:
        return "#f39c12"        # orange       — mild overload
    else:
        return "#e74c3c"        # red          — severe overload


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

def _hierarchical_layout(G: nx.DiGraph, root_id: int) -> dict:
    """
    Top-down tree layout.  Nodes at the same depth share a horizontal band;
    each band is evenly divided.  No graphviz dependency required.
    """
    depth: dict = {root_id: 0}
    layers: dict = defaultdict(list)
    layers[0].append(root_id)
    queue = deque([root_id])

    while queue:
        nid = queue.popleft()
        for child in G.successors(nid):
            if child not in depth:
                depth[child] = depth[nid] + 1
                layers[depth[child]].append(child)
                queue.append(child)

    pos: dict = {}
    for d, nodes in layers.items():
        n = max(len(nodes), 1)
        for i, nid in enumerate(nodes):
            pos[nid] = ((i + 0.5) / n, -d)

    return pos


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def plot_search_tree(result, env, save_path: str = None, max_nodes: int = 120):
    """
    Draw the A* search tree and save (or display) it.

    Node colours
    ────────────
    bright green  goal node (ρ < 0.98, solution found here)
    dark green    safe but not the goal
    orange        mild overload  (0.98 ≤ ρ < 1.05)
    red           severe overload (ρ ≥ 1.05)

    Edges on the solution path are drawn in blue; all others in light grey.
    Action indices are labelled on solution-path edges only.

    Args:
        result:     AStarResult returned by AStarSearch.search()
        env:        GridStarEnv (used only for context; not queried here)
        save_path:  if given, figure is saved there; otherwise plt.show() is called
        max_nodes:  cap on nodes rendered (large trees become unreadable)
    """
    # ── Trim to max_nodes ──────────────────────────────────────────
    visible = result.all_nodes[:max_nodes]
    valid_ids = {n.node_id for n in visible}
    edges = [
        (p, c, a)
        for p, c, a in result.edges
        if p in valid_ids and c in valid_ids
    ]

    # ── Build graph ────────────────────────────────────────────────
    G = nx.DiGraph()
    attr: dict = {}
    for node in visible:
        rho = float(node.obs.rho.max())
        G.add_node(node.node_id)
        attr[node.node_id] = {
            "rho": rho,
            "g": node.g,
            "depth": node.depth,
            "is_goal": result.found and node is result.goal_node,
        }

    for parent_id, child_id, action_idx in edges:
        G.add_edge(parent_id, child_id, action=action_idx)

    root_id = visible[0].node_id
    pos = _hierarchical_layout(G, root_id)

    # ── Solution path edge set ─────────────────────────────────────
    sol_edges: set = set()
    if result.found and result.path:
        for i in range(1, len(result.path)):
            sol_edges.add((result.path[i - 1].node_id, result.path[i].node_id))

    # ── Visual attributes ──────────────────────────────────────────
    node_colors = [
        _node_color(attr[n]["rho"], attr[n]["is_goal"]) for n in G.nodes()
    ]
    edge_colors = [
        "#2980b9" if (u, v) in sol_edges else "#bdc3c7"
        for u, v in G.edges()
    ]
    edge_widths = [
        2.5 if (u, v) in sol_edges else 0.7
        for u, v in G.edges()
    ]

    node_labels = {
        n: f"ρ={attr[n]['rho']:.2f}\ng={attr[n]['g']:.3f}"
        for n in G.nodes()
    }

    # ── Figure ─────────────────────────────────────────────────────
    fig_w = max(14, len(visible) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        labels=node_labels,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=750,
        font_size=5,
        arrows=True,
        arrowsize=10,
    )

    # Label action indices only on the solution path
    if sol_edges:
        sol_edge_labels = {
            (u, v): f"a{G.edges[u, v]['action']}"
            for u, v in G.edges()
            if (u, v) in sol_edges
        }
        nx.draw_networkx_edge_labels(
            G, pos, sol_edge_labels, font_size=7, font_color="#154360", ax=ax
        )

    # ── Legend ─────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color="#2ecc71", label="Goal  (ρ < 0.98)"),
        mpatches.Patch(color="#f39c12", label="Mild overload  (0.98 ≤ ρ < 1.05)"),
        mpatches.Patch(color="#e74c3c", label="Severe overload  (ρ ≥ 1.05)"),
        plt.Line2D([0], [0], color="#2980b9", linewidth=2.5, label="Solution path"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    # ── Title ──────────────────────────────────────────────────────
    status = "FOUND" if result.found else "NOT FOUND"
    title = (
        f"A* Search Tree  |  Solution: {status}"
        f"  |  Expanded: {result.n_expanded}"
        f"  |  Generated: {result.n_generated}"
        f"  |  Nodes shown: {len(visible)}"
    )
    if result.found:
        final_rho = float(result.goal_node.obs.rho.max())
        title += (
            f"\nPath length: {len(result.path) - 1} action(s)"
            f"  |  Final ρ_max: {final_rho:.4f}"
        )
    ax.set_title(title, fontsize=9, pad=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Search tree saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)
