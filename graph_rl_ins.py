import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import networkx as nx
from networkx.readwrite import json_graph


def load_trained_graph(json_file: str) -> nx.DiGraph:
    """Load a Q-trained graph from JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load graph structure
    G = json_graph.node_link_graph(data, directed=True)
    
    print(f"Loaded trained graph from {json_file}")
    if 'metadata' in data:
        meta = data['metadata']
        print(f"  Target: {meta.get('target_label', 'N/A')}")
        print(f"  Nodes: {meta.get('total_nodes', G.number_of_nodes())}")
        print(f"  Edges: {meta.get('total_edges', G.number_of_edges())}")
        print(f"  Trained edges: {meta.get('trained_edges', 'N/A')}")
    
    return G


def connect_all_instances(G: nx.DiGraph, weight: float = 1.0):
    """Connect all instance nodes to each other with bidirectional edges."""
    print("\n==== Connecting all instances ====")
    
    # Get all instance nodes
    instances = [n for n, data in G.nodes(data=True) 
                if data.get('node_type', '').lower() == 'instance' 
                and data.get('label') != 'Instance']
    
    print(f"Found {len(instances)} instance nodes")
    
    if len(instances) < 2:
        print("Not enough instances to connect")
        return G
    
    # Create bidirectional connections between all instances
    added_edges = 0
    for i, inst1 in enumerate(instances):
        for inst2 in instances[i+1:]:
            # Add edge in both directions if not already present
            if not G.has_edge(inst1, inst2):
                G.add_edge(inst1, inst2, 
                          edge_type='instance_connection',
                          label='connected_to',
                          weight=weight,
                          q_value=0.0,
                          synthetic=True)
                added_edges += 1
            
            if not G.has_edge(inst2, inst1):
                G.add_edge(inst2, inst1, 
                          edge_type='instance_connection',
                          label='connected_to',
                          weight=weight,
                          q_value=0.0,
                          synthetic=True)
                added_edges += 1
    
    print(f"Added {added_edges} bidirectional edges between instances")
    return G


def spread_class_weights_to_instances(G: nx.DiGraph):
    """Spread weights from classes to their instances via instance_of/is_a relationships.
    
    Logic: If class A -> class B has an edge with weight W,
    then instance a (child of A) -> instance b (child of B) should also have an edge with weight W.
    """
    print("\n==== Spreading class weights to instances ====")
    
    # Build class -> instances mapping
    # First, determine the direction by checking node types
    class_to_instances = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', '')
        if edge_type in ['instance_of', 'is_a']:
            u_type = G.nodes[u].get('node_type', '')
            v_type = G.nodes[v].get('node_type', '')
            
            # Determine which is instance and which is class based on node_type
            if u_type == 'instance' and v_type in ['class', 'instance_class']:
                # u is instance, v is class (instance -> class direction)
                instance_node = u
                class_node = v
                class_to_instances[class_node].append(instance_node)
            elif v_type == 'instance' and u_type in ['class', 'instance_class']:
                # v is instance, u is class (class -> instance direction)
                instance_node = v
                class_node = u
                class_to_instances[class_node].append(instance_node)
    
    # Also build instance -> class(es) mapping to handle subclass hierarchies
    instance_to_classes = defaultdict(set)
    for class_node, instances in class_to_instances.items():
        for inst in instances:
            # Add the direct class
            instance_to_classes[inst].add(class_node)
            # Add all ancestor classes (following inheritance edges)
            for ancestor in nx.ancestors(G, class_node):
                ancestor_type = G.nodes[ancestor].get('node_type', '')
                if ancestor_type in ['class', 'instance_class']:
                    instance_to_classes[inst].add(ancestor)
    
    print(f"Found {len(class_to_instances)} classes with instances")
    
    # Build extended mapping: class -> all instances (including from subclasses)
    # According to ins_weighting_rule.md, if a class has no direct instances but has subclasses with instances,
    # those subclass instances should also be considered
    class_to_all_instances = defaultdict(list)
    
    def get_all_subclasses(class_node, visited=None):
        """Recursively find all subclasses by traversing predecessor edges (inheritance)."""
        if visited is None:
            visited = set()
        if class_node in visited:
            return []
        visited.add(class_node)
        
        subclasses = []
        # Find all nodes that point to this class via inheritance edges
        for pred in G.predecessors(class_node):
            pred_type = G.nodes[pred].get('node_type', '')
            if pred_type in ['class', 'instance_class']:
                # Check if this is an inheritance edge
                edge_data = G[pred][class_node]
                if edge_data.get('edge_type') == 'inheritance':
                    subclasses.append(pred)
                    # Recursively get subclasses of this subclass
                    subclasses.extend(get_all_subclasses(pred, visited))
        return subclasses
    
    for class_node in G.nodes():
        if G.nodes[class_node].get('node_type') not in ['class', 'instance_class']:
            continue
        
        # Add direct instances
        if class_node in class_to_instances:
            class_to_all_instances[class_node].extend(class_to_instances[class_node])
        
        # Add instances from all descendant classes (subclasses)
        subclasses = get_all_subclasses(class_node)
        for subclass in subclasses:
            if subclass in class_to_instances:
                # These are instances of a subclass, so they should also be considered instances of the parent class
                class_to_all_instances[class_node].extend(class_to_instances[subclass])
    
    print(f"After including subclass instances: {len(class_to_all_instances)} classes have accessible instances")
    
    # Get all class-to-class edges (excluding instance_of, is_a, instance_connection)
    class_edges = []
    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get('node_type', '')
        v_type = G.nodes[v].get('node_type', '')
        edge_type = data.get('edge_type', '')
        
        # Check if both nodes are classes and edge is not an instance relationship
        if (u_type in ['class', 'instance_class'] and 
            v_type in ['class', 'instance_class'] and
            edge_type not in ['instance_of', 'is_a', 'instance_connection']):
            class_edges.append((u, v, data))
    
    print(f"Found {len(class_edges)} class-to-class edges")
    
    # Debug: Show some class-to-instance mappings (direct instances only)
    if len(class_to_instances) > 0:
        print(f"\nDebug: Sample DIRECT class-to-instance mappings:")
        for i, (cls, insts) in enumerate(list(class_to_instances.items())[:5]):
            cls_label = G.nodes[cls].get('label', cls)
            inst_texts = [G.nodes[inst].get('text', G.nodes[inst].get('label', inst)) for inst in insts[:3]]
            print(f"  {cls_label}: {len(insts)} direct instances (e.g., {inst_texts})")
    
    # Debug: Show ALL classes with instances from subclasses
    if len(class_to_all_instances) > 0:
        print(f"\nDebug: ALL classes with inherited instances from subclasses:")
        classes_with_inheritance = []
        for cls in G.nodes():
            if G.nodes[cls].get('node_type') not in ['class', 'instance_class']:
                continue
            direct = len(class_to_instances.get(cls, []))
            total = len(class_to_all_instances.get(cls, []))
            if direct != total:  # Classes where subclass instances matter
                cls_label = G.nodes[cls].get('label', cls)
                classes_with_inheritance.append((cls_label, direct, total))
        
        if classes_with_inheritance:
            print(f"  Found {len(classes_with_inheritance)} classes with inherited instances:")
            for cls_label, direct, total in classes_with_inheritance:
                print(f"    {cls_label}: {direct} direct + {total-direct} from subclasses = {total} total")
        else:
            print("  No classes found with inherited instances from subclasses")
    
    # Debug: Show sample class-to-class edges
    if len(class_edges) > 0:
        print(f"\nDebug: Sample class-to-class edges:")
        for i, (cls_a, cls_b, edge_data) in enumerate(class_edges[:5]):
            cls_a_label = G.nodes[cls_a].get('label', cls_a)
            cls_b_label = G.nodes[cls_b].get('label', cls_b)
            weight = edge_data.get('weight', 1.0)
            edge_type = edge_data.get('edge_type', 'unknown')
            print(f"  {cls_a_label} --[{edge_type}, w={weight:.2f}]--> {cls_b_label}")
            print(f"    Class A has {len(class_to_instances.get(cls_a, []))} instances")
            print(f"    Class B has {len(class_to_instances.get(cls_b, []))} instances")
            print(f"    Class A (including subclasses) has {len(class_to_all_instances.get(cls_a, []))} total instances")
            print(f"    Class B (including subclasses) has {len(class_to_all_instances.get(cls_b, []))} total instances")
    
    # Now propagate each class-to-class edge to instance-to-instance edges
    propagated_count = 0
    updated_count = 0
    skipped_no_instances = 0
    
    for class_A, class_B, edge_data in class_edges:
        # Get instances of class A (including from its subclasses)
        instances_A = class_to_all_instances.get(class_A, [])
        
        # Get instances of class B (including from its subclasses)
        instances_B = class_to_all_instances.get(class_B, [])
        
        # Debug for specific edge QualityRisk -> QualityAccident
        cls_a_label = G.nodes[class_A].get('label', '')
        cls_b_label = G.nodes[class_B].get('label', '')
        if 'qualityrisk' in cls_a_label.lower() and 'qualityaccident' in cls_b_label.lower():
            print(f"\n=== DEBUG: Found QualityRisk -> QualityAccident edge ===")
            print(f"  Class A: {cls_a_label} (node type: {G.nodes[class_A].get('node_type')})")
            print(f"  Class B: {cls_b_label} (node type: {G.nodes[class_B].get('node_type')})")
            print(f"  Edge weight: {edge_data.get('weight', 1.0):.2f}")
            print(f"  Edge q_value: {edge_data.get('q_value', 0.0):.4f}")
            print(f"  Instances of {cls_a_label}: {len(instances_A)}")
            if instances_A:
                for inst in instances_A[:3]:
                    inst_text = G.nodes[inst].get('text', G.nodes[inst].get('label', inst))
                    print(f"    - {inst_text}")
            print(f"  Instances of {cls_b_label}: {len(instances_B)}")
            if instances_B:
                for inst in instances_B[:3]:
                    inst_text = G.nodes[inst].get('text', G.nodes[inst].get('label', inst))
                    print(f"    - {inst_text}")
        
        # Track if we skip due to no instances
        if not instances_A or not instances_B:
            skipped_no_instances += 1
            continue
        
        # Create edges between all instances of A and all instances of B
        for inst_a in instances_A:
            for inst_b in instances_B:
                # Get weight and q_value from class edge
                weight = edge_data.get('weight', 1.0)
                q_value = edge_data.get('q_value', 0.0)
                edge_type = edge_data.get('edge_type', 'relation')
                label = edge_data.get('label', 'related_to')
                
                # Create new edge data with same attributes as class edge
                new_edge_data = {
                    'edge_type': edge_type,
                    'label': label,
                    'weight': weight,
                    'q_value': q_value,
                    'propagated_from': f"{class_A}->{class_B}",
                    'source_class': class_A,
                    'target_class': class_B
                }
                
                # Debug for QualityRisk -> QualityAccident instances
                if 'qualityrisk' in cls_a_label.lower() and 'qualityaccident' in cls_b_label.lower():
                    if inst_a == instances_A[0] and inst_b == instances_B[0]:  # Just show first pair
                        inst_a_text = G.nodes[inst_a].get('text', G.nodes[inst_a].get('label', inst_a))
                        inst_b_text = G.nodes[inst_b].get('text', G.nodes[inst_b].get('label', inst_b))
                        print(f"  Processing instance pair:")
                        print(f"    {inst_a_text} -> {inst_b_text}")
                        print(f"    Edge exists? {G.has_edge(inst_a, inst_b)}")
                        if G.has_edge(inst_a, inst_b):
                            existing = G[inst_a][inst_b]
                            print(f"    Existing edge_type: {existing.get('edge_type')}")
                            print(f"    Existing weight: {existing.get('weight')}")
                
                # Check if edge already exists
                if G.has_edge(inst_a, inst_b):
                    # Update existing edge if it's a generic instance_connection
                    existing_data = G[inst_a][inst_b]
                    if existing_data.get('edge_type') == 'instance_connection':
                        # Replace generic connection with specific class-based edge
                        G[inst_a][inst_b].update(new_edge_data)
                        updated_count += 1
                    # Otherwise keep the existing specific edge
                else:
                    # Create new edge
                    G.add_edge(inst_a, inst_b, **new_edge_data)
                    propagated_count += 1
    
    print(f"Propagated {propagated_count} new edges to instances")
    print(f"Updated {updated_count} existing instance_connection edges with class-based weights")
    if skipped_no_instances > 0:
        print(f"Skipped {skipped_no_instances} class edges because one or both classes had no instances")
    return G


def ensure_all_edges_have_weights(G: nx.DiGraph, default_weight: float = 1.0):
    """Ensure all edges have weight and q_value attributes."""
    print("\n==== Ensuring all edges have weights ====")
    
    added_weights = 0
    added_q_values = 0
    
    for u, v, data in G.edges(data=True):
        if 'weight' not in data or data['weight'] is None:
            data['weight'] = default_weight
            added_weights += 1
        
        if 'q_value' not in data or data['q_value'] is None:
            data['q_value'] = 0.0
            added_q_values += 1
    
    if added_weights > 0:
        print(f"Added default weight to {added_weights} edges")
    if added_q_values > 0:
        print(f"Added default q_value to {added_q_values} edges")
    
    print(f"All {G.number_of_edges()} edges now have weight and q_value attributes")
    return G


def init_q(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Initialize Q-table from existing q_values in graph."""
    Q = defaultdict(dict)
    for u, v, data in G.edges(data=True):
        Q[u][v] = data.get('q_value', 0.0)
    return Q


def available_actions(G: nx.DiGraph, state):
    return list(G.successors(state))


def epsilon_greedy(Q_row: Dict, actions: List, epsilon: float):
    if not actions:
        return None
    if random.random() < epsilon:
        return random.choice(actions)
    best_a = max(actions, key=lambda a: Q_row.get(a, 0.0))
    return best_a


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
    """Train Q-values for mixed class-instance paths."""
    Q = init_q(G)

    if isinstance(target, (list, set, tuple)):
        targets = set(target)
    else:
        targets = {target}

    # Precompute reachable nodes
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

    start_nodes = [n for n in G.nodes() if n in can_reach and n not in targets]
    if not start_nodes:
        raise ValueError("No eligible start nodes can reach the target.")

    # Training loop
    for ep in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))
        state = random.choice(start_nodes)

        for _ in range(max_steps):
            if state in targets:
                break
            actions = available_actions(G, state)
            actions = [a for a in actions if a in can_reach]
            if not actions:
                break

            action = epsilon_greedy(Q[state], actions, epsilon)
            if action is None:
                break

            reward = G[state][action].get("weight", 0.0)
            next_state = action

            next_actions = available_actions(G, next_state)
            next_actions = [a for a in next_actions if a in can_reach]
            max_next_Q = 0.0 if not next_actions else max(Q[next_state].get(a, 0.0) for a in next_actions)

            old_q = Q[state].get(action, 0.0)
            Q[state][action] = old_q + alpha * (reward + gamma * max_next_Q - old_q)

            state = next_state

    return Q


def enumerate_paths_to_target(
    G: nx.DiGraph,
    target,
    max_depth: int = 6,
    max_paths: int = 2000,
):
    """Enumerate paths that end at target."""
    R = G.reverse(copy=False)
    paths = []

    def dfs(curr, path):
        if len(paths) >= max_paths:
            return
        if len(path) - 1 > max_depth:
            return
        if curr not in R:
            return
        
        preds = list(R.successors(curr))
        if not preds:
            paths.append(list(reversed(path)))
            return
        
        for p in preds:
            if p in path:
                continue
            dfs(p, path + [p])

    dfs(target, [target])
    return paths


def path_scores(G: nx.DiGraph, Q: Dict, path: List) -> Tuple[float, float]:
    """Return (sum_Q, sum_weight) along the path."""
    q_sum = 0.0
    w_sum = 0.0
    for u, v in zip(path[:-1], path[1:]):
        q_sum += Q.get(u, {}).get(v, 0.0)
        w_sum += G[u][v].get("weight", 0.0)
    return q_sum, w_sum


def label_path(G: nx.DiGraph, path: List) -> List[str]:
    """Generate display labels for path nodes with type indicators."""
    labels = []
    for n in path:
        data = G.nodes[n]
        ntype = (data.get('node_type') or '').lower()
        
        if ntype == 'instance':
            text = data.get('text') or data.get('label') or str(n)
            labels.append(f"[I] {text}")
        elif ntype == 'class':
            label = data.get('label') or str(n)
            labels.append(f"[C] {label}")
        else:
            label = data.get('label') or data.get('text') or str(n)
            labels.append(label)
    
    return labels


def run_instance_q_learning(
    trained_graph_file: str,
    target_label: str = None,
    episodes: int = 3000,
    max_depth: int = 6,
):
    """
    Use trained graph as input and perform instance-level Q-learning.
    
    Args:
        trained_graph_file: Path to q_trained_graph JSON file
        target_label: Target node label to search for
        episodes: Number of Q-learning episodes
        max_depth: Maximum path depth
    """
    import os
    
    # Load trained graph
    if not os.path.exists(trained_graph_file):
        raise FileNotFoundError(f"Trained graph file not found: {trained_graph_file}")
    
    G = load_trained_graph(trained_graph_file)
    
    # Step 1: Connect all instances
    G = connect_all_instances(G, weight=1.0)
    
    # Step 2: Spread weights from classes to instances
    G = spread_class_weights_to_instances(G)
    
    # Step 3: Ensure all edges have weights
    G = ensure_all_edges_have_weights(G, default_weight=1.0)
    
    # Save enhanced graph
    base_name = os.path.splitext(os.path.basename(trained_graph_file))[0]
    enhanced_graph_file = f"{base_name}_instance_enhanced.json"
    graph_data = json_graph.node_link_data(G, edges="links")
    with open(enhanced_graph_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved enhanced graph to {enhanced_graph_file}")
    
    # Find target nodes
    if not target_label:
        raise ValueError("Target label must be specified")
    
    targets = []
    tl = target_label.lower()
    for n, data in G.nodes(data=True):
        lab = (data.get("label") or "").lower()
        text = (data.get("text") or "").lower()
        if tl in lab or tl in text:
            targets.append(n)
    
    if not targets:
        raise ValueError(f"Target label '{target_label}' not found in graph")
    
    # Expand targets
    tset = set(targets)
    for t in list(targets):
        for s in G.successors(t):
            tset.add(s)
    targets = list(tset)
    print(f"\nUsing {len(targets)} target nodes after expansion")
    
    # Train Q-values
    print(f"\nTraining Q-values for {episodes} episodes...")
    Q = q_learning_to_target(G, targets, episodes=episodes)
    
    # Enumerate paths
    print("\nEnumerating paths...")
    paths = []
    for t in targets:
        ps = enumerate_paths_to_target(G, t, max_depth=max_depth)
        paths.extend(ps)
    
    print(f"Found {len(paths)} paths")
    
    # Score paths
    scored = []
    for p in paths:
        q_sum, w_sum = path_scores(G, Q, p)
        
        # Count instances and classes
        instance_count = sum(1 for n in p if G.nodes[n].get('node_type', '').lower() == 'instance')
        class_count = sum(1 for n in p if G.nodes[n].get('node_type', '').lower() == 'class')
        
        scored.append({
            "path_nodes": p,
            "path_labels": label_path(G, p),
            "q_sum": q_sum,
            "weight_sum": w_sum,
            "length": len(p) - 1,
            "instance_count": instance_count,
            "class_count": class_count,
        })
    
    # Sort by q_sum and weight_sum
    scored.sort(key=lambda x: (x["q_sum"], x["weight_sum"]), reverse=True)
    
    # Print results
    print("\n==== Mixed class-instance path results ====")
    for i, s in enumerate(scored[:20], 1):
        print(f"#{i}: len={s['length']} classes={s['class_count']} instances={s['instance_count']} "
              f"q_sum={s['q_sum']:.3f} weight={s['weight_sum']:.2f}")
        print("   " + " -> ".join(s["path_labels"]))
    
    # Save results
    out = {
        "source_graph": trained_graph_file,
        "target_label": target_label,
        "episodes": episodes,
        "max_depth": max_depth,
        "paths": scored,
    }
    out_file = f"q_instance_paths_{target_label}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(scored)} paths to {out_file}")
    
    # Update graph with new Q-values and save final trained graph
    for u in Q:
        for v in Q[u]:
            if G.has_edge(u, v):
                G[u][v]['q_value'] = Q[u][v]
    
    final_trained_file = f"{base_name}_instance_trained.json"
    graph_data = json_graph.node_link_data(G, edges="links")
    with open(final_trained_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"Saved final trained graph to {final_trained_file}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Instance-level Q-learning using trained graph')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to q_trained_graph JSON file')
    parser.add_argument('--target', '-t', type=str, default=None,
                        help='Target label to search for')
    parser.add_argument('--episodes', '-e', type=int, default=4000,
                        help='Number of Q-learning episodes (default: 4000)')
    parser.add_argument('--depth', '-d', type=int, default=7,
                        help='Maximum path depth (default: 7)')
    args = parser.parse_args()
    
    # Get input file
    input_file = args.input
    if not input_file:
        # Look for most recent q_trained_graph file
        base = os.path.dirname(__file__)
        trained_files = [f for f in os.listdir(base) if f.startswith('q_trained_graph_') and f.endswith('.json')]
        if trained_files:
            # Use most recent
            trained_files.sort(key=lambda f: os.path.getmtime(os.path.join(base, f)), reverse=True)
            input_file = os.path.join(base, trained_files[0])
            print(f"Using most recent trained graph: {trained_files[0]}")
        else:
            raise FileNotFoundError("No q_trained_graph files found. Please specify --input")
    
    # Get target
    target_label = args.target
    if not target_label:
        target_label = input("Enter target label: ").strip()
        if not target_label:
            raise ValueError("Target label is required")
    
    run_instance_q_learning(
        trained_graph_file=input_file,
        target_label=target_label,
        episodes=args.episodes,
        max_depth=args.depth
    )
