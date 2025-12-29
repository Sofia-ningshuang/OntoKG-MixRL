import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import networkx as nx

from graph_rl_ini import load_graph_for_qlearning

# Q-learning 节点搜索
def find_node_by_label(G: nx.DiGraph, label: str):
    for n, data in G.nodes(data=True):
        if data.get("label") == label:
            return n
    return None

# Q-learning 动作空间
def available_actions(G: nx.DiGraph, state):
    return list(G.successors(state))

# Q-learning 图初始化
def init_q(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    Q = defaultdict(dict)
    for u, v in G.edges():
        Q[u][v] = 0.0
    return Q

# Q-learning 贪心策略
def epsilon_greedy(Q_row: Dict, actions: List, epsilon: float):
    if not actions:
        return None
    if random.random() < epsilon:
        return random.choice(actions)
    # exploit: choose action with max Q 选择拥有最大Q值的行动
    best_a = max(actions, key=lambda a: Q_row.get(a, 0.0))
    return best_a

# Q-learning 核心参数与算法步骤
def q_learning_to_target(
    G: nx.DiGraph,
    target,
    episodes: int = 3000,
    max_steps: int = 50,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
):
    """
    Train Q-values so that trajectories ending at `target` are reinforced.
    Reward model: edge weight is the immediate reward.
    Episode terminates when target is reached or max_steps exceeded.
    """
    Q = init_q(G)

    # Allow single target or iterable of targets
    if isinstance(target, (list, set, tuple)):
        targets = set(target)
    else:
        targets = {target}

    # Precompute eligible starting states: nodes that can reach any target
    R = G.reverse(copy=False)
    can_reach = set()
    dq = deque(list(targets))
    for t in targets:
        can_reach.add(t)
    while dq:
        x = dq.popleft()
        for y in R.successors(x):
            if y not in can_reach:
                can_reach.add(y)
                dq.append(y)

    # Exclude the generic 'Instance' node from starting points 从起始点中排除通用的“实例”节点
    start_nodes = [
        n for n in G.nodes()
        if n in can_reach and n not in targets and G.nodes[n].get('label') != 'Instance'
    ]
    if not start_nodes:
        raise ValueError("No eligible start nodes can reach the target.")

    # Training loop with linear epsilon decay 具有线性ε衰减的训练循环
    for ep in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))
        state = random.choice(start_nodes)

        for _ in range(max_steps):
            if state in targets:
                break
            actions = available_actions(G, state)
            # Keep only actions that are on some path to target to reduce wandering 仅保留那些指向目标路径上的操作，以减无意义操作
            actions = [a for a in actions if a in can_reach]
            if not actions:
                break

            action = epsilon_greedy(Q[state], actions, epsilon)
            if action is None:
                break

            reward = G[state][action].get("weight", 0.0)
            next_state = action

            next_actions = available_actions(G, next_state)
            # Constrain next actions to those that can reach target 将后续行动限制在能够达成目标的行动范围内
            next_actions = [a for a in next_actions if a in can_reach]
            max_next_Q = 0.0 if not next_actions else max(Q[next_state].get(a, 0.0) for a in next_actions)

            # TD update 时序差分更新
            old_q = Q[state].get(action, 0.0)
            Q[state][action] = old_q + alpha * (reward + gamma * max_next_Q - old_q)

            state = next_state

    return Q

# 枚举路径
def enumerate_paths_to_target(
    G: nx.DiGraph,
    target,
    max_depth: int = 6,
    max_paths: int = 2000,
):
    """Enumerate simple forward-direction paths that end at target (length<=max_depth)."""
    R = G.reverse(copy=False)
    paths = []

    def dfs(curr, path):
        if len(paths) >= max_paths:
            return
        if len(path) - 1 > max_depth:
            return
        if curr is None:
            return
        if curr not in R:
            return
        # Expand predecessors in forward direction: on reversed graph, successors are predecessors 向前扩展前驱：在反向图中，后继是前驱
        preds = list(R.successors(curr))
        if not preds:
            # Reached a start; record reversed path to be forward direction 已经到达起点；记录反向路径以指示正向方向
            paths.append(list(reversed(path)))
            return
        for p in preds:
            # If predecessor is the generic 'Instance', stop here and record path without including it 如果前置对象是通用的“实例”，则到此为止，并记录路径而不包含它
            if G.nodes[p].get('label') == 'Instance':
                paths.append(list(reversed(path)))
                continue
            if p in path:
                continue
            dfs(p, path + [p])

    dfs(target, [target])
    # Each path is in forward direction from some start -> ... -> target 每条路径都是从某个起点-> ... -> 目标
    return paths

# 路径得分
def path_scores(G: nx.DiGraph, Q: Dict, path: List) -> Tuple[float, float]:
    """Return (sum_Q, sum_weight) along the path."""
    q_sum = 0.0
    w_sum = 0.0
    for u, v in zip(path[:-1], path[1:]):
        q_sum += Q.get(u, {}).get(v, 0.0)
        w_sum += G[u][v].get("weight", 0.0)
    return q_sum, w_sum

# 推理类别标签
def infer_category_label(G: nx.DiGraph, node, candidate_labels=None, max_hops: int = 4):
    """Find nearest ancestor via inheritance whose label is in candidate_labels.
    If candidate_labels is None, use a default set of likely categories.
    """
    # 顶层标识为以下四类
    if candidate_labels is None:
        candidate_labels = {"HazardCondition", "TriggerEvent", "QualityDefect", "QualityRisk"}

    # BFS up the inheritance (parents via forward graph: parent -> child) BFS向上遍历继承关系（通过前向图查找父级：父级 -> 子级）
    # So to go to parents, use predecessors in forward graph 追溯到父节点，需要在前向图中利用前驱节点
    visited = {node}
    frontier = [(node, 0)]
    while frontier:
        curr, d = frontier.pop(0)
        if d > max_hops:
            break
        for parent in G.predecessors(curr):
            if parent in visited:
                continue
            visited.add(parent)
            label = G.nodes[parent].get('label')
            if label in candidate_labels:
                return label
            frontier.append((parent, d + 1))
    return None

# 标记路径
def label_path(G: nx.DiGraph, path: List) -> List[str]:
    def display_name(n):
        data = G.nodes[n]
        # prefer instance `text` when available
        ntype = (data.get('node_type') or '').lower()
        if ntype == 'instance' or data.get('label') == 'Instance':
            return data.get('text') or data.get('label') or str(n)
        # otherwise prefer label, fallback to text or id
        return data.get('label') or data.get('text') or str(n)

    labels = [display_name(n) for n in path]
    # If the path originally started from 'Instance' (now removed), we cannot detect that directly here. 如果路径最初是从“实例”（现已删除）开始的，我们无法在此处直接检测到
    # But we can add a category tag for the first node when available to replace 'Instance'. 如果可用，可以为第一个节点添加一个类别标签来替换“实例”
    if labels:
        cat = infer_category_label(G, path[0])
        if cat and cat != labels[0]:
            labels[0] = f"{cat}: {labels[0]}"
    return labels

# 输出边
def print_edges_with_category(G: nx.DiGraph, category_label: str = "HazardCondition"):
    """Print edges where either endpoint is (or belongs under) the given category."""
    print("\n==== Edges involving category:", category_label, "====")
    count = 0
    def disp(n):
        d = G.nodes[n]
        ntype = (d.get('node_type') or '').lower()
        if ntype == 'instance' or d.get('label') == 'Instance':
            return d.get('text') or d.get('label') or str(n)
        return d.get('label') or d.get('text') or str(n)

    for u, v, data in G.edges(data=True):
        u_lab = disp(u)
        v_lab = disp(v)
        # Direct label match or inferred category match 直接标签匹配或推断类别匹配
        u_match = (u_lab == category_label) or (infer_category_label(G, u) == category_label)
        v_match = (v_lab == category_label) or (infer_category_label(G, v) == category_label)
        if u_match or v_match:
            e_lab = data.get("label", data.get("edge_type", ""))
            w = data.get("weight", 0)
            where = "source" if u_match else ("target" if v_match else "either")
            print(f"  [{where}] {u_lab} -> {v_lab}  (edge: {e_lab}, weight: {w})")
            count += 1
    print(f"Total edges involving {category_label}: {count}")

# 输出q值
def print_graph_with_q(G: nx.DiGraph, Q: Dict):
    """Print the full graph: nodes, edges with weights and Q-scores."""
    print("\n==== Graph nodes ====")
    for n, data in G.nodes(data=True):
        ntype = data.get('node_type', '')
        # prefer instance text
        if (ntype or '').lower() == 'instance' or data.get('label') == 'Instance':
            label = data.get('text') or data.get('label') or str(n)
        else:
            label = data.get('label') or data.get('text') or str(n)
        print(f"  Node: {label}  (type: {ntype})")

    print("\n==== Graph edges (u -> v | edge_label | weight | Q) ====")
    for u, v, data in G.edges(data=True):
        ud = G.nodes[u]
        vd = G.nodes[v]
        if (ud.get('node_type') or '').lower() == 'instance' or ud.get('label') == 'Instance':
            u_lab = ud.get('text') or ud.get('label') or str(u)
        else:
            u_lab = ud.get('label') or ud.get('text') or str(u)
        if (vd.get('node_type') or '').lower() == 'instance' or vd.get('label') == 'Instance':
            v_lab = vd.get('text') or vd.get('label') or str(v)
        else:
            v_lab = vd.get('label') or vd.get('text') or str(v)
        e_lab = data.get("label", data.get("edge_type", ""))
        w = data.get("weight", 0)
        q = Q.get(u, {}).get(v, 0.0)
        print(f"  {u_lab} -> {v_lab}  | {e_lab} | w={w} | Q={q:.3f}")

# 启动Q-learning
def run_q_learning(
    target_label: str = "StructuralSafetyAccident",
    episodes: int = 3000,
    max_depth: int = 6,
):
    # Load weighted graph: prefer adjusted RL graph, then merged graph, then default
    import os
    base = os.path.dirname(__file__)
    rl_graph = os.path.join(base, 'rl_graph.json')
    merged_graph = os.path.join(base, 'merged_graph.json')
    if os.path.exists(rl_graph):
        G = load_graph_for_qlearning(rl_graph)
    elif os.path.exists(merged_graph):
        G = load_graph_for_qlearning(merged_graph)
    else:
        # fallback to load_graph_for_qlearning default behavior
        G = load_graph_for_qlearning()

    # Find target nodes by substring match (label or text), case-insensitive
    targets = []
    tl = target_label.lower()
    for n, data in G.nodes(data=True):
        lab = (data.get("label") or "").lower()
        text = (data.get("text") or "").lower()
        if tl in lab or tl in text:
            targets.append(n)

    if not targets:
        # fallback to exact match
        node = find_node_by_label(G, target_label)
        if node is None:
            all_labels = sorted({data.get("label", str(n)) for n, data in G.nodes(data=True)})
            raise ValueError(
                f"Target label '{target_label}' not found (substring). Available labels include: "
                + ", ".join(all_labels[:30])
                + ("..." if len(all_labels) > 30 else "")
            )
        targets = [node]

    # If any target nodes are instances, also include their class neighbors (instance_of -> class)
    tset = set(targets)
    for t in list(targets):
        for s in G.successors(t):
            tset.add(s)
    targets = list(tset)
    print(f"Using {len(targets)} target nodes after expansion.")

    # Train Q-values 训练Q值 for all targets
    Q = q_learning_to_target(G, targets, episodes=episodes)

    # Enumerate paths to each target and combine
    paths = []
    for t in targets:
        ps = enumerate_paths_to_target(G, t, max_depth=max_depth)
        paths.extend(ps)

    # Score paths 对路径进行打分
    scored = []
    for p in paths:
        q_sum, w_sum = path_scores(G, Q, p)
        scored.append({
            "path_nodes": p,
            "path_labels": label_path(G, p),
            "q_sum": q_sum,
            "weight_sum": w_sum,
            "length": len(p) - 1,
        })

    # Sort by q_sum desc, then by weight_sum desc 通过q累计值与w累计权重排序
    scored.sort(key=lambda x: (x["q_sum"], x["weight_sum"]), reverse=True)

    # Print results (top 20) 输出前20路径
    print("==== Q-learning backtrace results ====")
    for i, s in enumerate(scored[:20], 1):
        print(f"#{i}: q_sum={s['q_sum']:.3f} weight_sum={s['weight_sum']} len={s['length']}")
        print("   " + " -> ".join(s["path_labels"]))

    # Save all results to JSON 存储到JSON文件
    out = {
        "target_label": target_label,
        "episodes": episodes,
        "max_depth": max_depth,
        "paths": scored,
    }
    with open(f"q_paths_{target_label}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved all path scores to q_paths_{target_label}.json (total {len(scored)} paths)")

    # Save the trained graph with Q values and weights
    from networkx.readwrite import json_graph
    
    # Add Q values to edge attributes for trained edges
    trained_edges = 0
    for u in Q:
        for v in Q[u]:
            if G.has_edge(u, v):
                G[u][v]['q_value'] = Q[u][v]
                trained_edges += 1
    
    # Ensure all edges have q_value attribute (0.0 for untrained)
    for u, v, data in G.edges(data=True):
        if 'q_value' not in data:
            data['q_value'] = 0.0
    
    # Export complete graph to JSON with all nodes, edges, weights, and Q values
    graph_data = json_graph.node_link_data(G)
    
    # Add metadata
    graph_data['metadata'] = {
        'target_label': target_label,
        'episodes': episodes,
        'max_depth': max_depth,
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'trained_edges': trained_edges,
        'node_types': {},
        'edge_types': {}
    }
    
    # Count node and edge types
    for n, data in G.nodes(data=True):
        ntype = data.get('node_type', 'unknown')
        graph_data['metadata']['node_types'][ntype] = graph_data['metadata']['node_types'].get(ntype, 0) + 1
    
    for u, v, data in G.edges(data=True):
        etype = data.get('edge_type', 'unknown')
        graph_data['metadata']['edge_types'][etype] = graph_data['metadata']['edge_types'].get(etype, 0) + 1
    
    trained_graph_file = f"q_trained_graph_{target_label}.json"
    with open(trained_graph_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved trained graph to {trained_graph_file}:")
    print(f"  - {G.number_of_nodes()} nodes (classes, instances, etc.)")
    print(f"  - {G.number_of_edges()} edges")
    print(f"  - {trained_edges} edges with trained Q-values")
    print(f"  - All edges have 'weight' and 'q_value' attributes")
    print(f"  - All nodes have complete attribute information")

    # Print the whole graph with Q-scores as requested 输出全图q值
    print_graph_with_q(G, Q)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Q-learning path search to target label')
    parser.add_argument('--target', '-t', type=str, default=None,
                        help='Target label to search for (e.g., "Accident", "StructuralSafetyAccident")')
    parser.add_argument('--episodes', '-e', type=int, default=4000,
                        help='Number of Q-learning episodes (default: 4000)')
    parser.add_argument('--depth', '-d', type=int, default=7,
                        help='Maximum path depth (default: 7)')
    args = parser.parse_args()
    
    # Get target from argument or prompt user for input
    target_label = args.target
    if not target_label:
        target_label = input("Enter target label (e.g., 'Accident', 'StructuralSafetyAccident'): ").strip()
        if not target_label:
            target_label = "StructuralSafetyAccident"  # fallback default
            print(f"Using default target: {target_label}")
    
    run_q_learning(target_label=target_label, episodes=args.episodes, max_depth=args.depth)
