import json
import networkx as nx
from networkx.readwrite import json_graph
import os
from typing import Dict


# parse weighting rules from weighting_rule.md
def parse_weighting_rule(path: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not os.path.exists(path):
        return mapping
    text = open(path, 'r', encoding='utf-8').read()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('```')]
    for line in lines:
        parts = [p.strip() for p in line.split() if p.strip()]
        if len(parts) < 2:
            continue
        weight_part = parts[0]
        edge_type = parts[1]
        label = parts[2] if len(parts) > 2 else ''
        # parse range like 0.45–0.60
        try:
            if '–' in weight_part:
                a, b = weight_part.split('–')
                w = (float(a) + float(b)) / 2.0
            elif '-' in weight_part and weight_part.count('-') == 1:
                a, b = weight_part.split('-')
                w = (float(a) + float(b)) / 2.0
            else:
                w = float(weight_part)
        except Exception:
            continue
        key = f"{edge_type}::{label}" if label and label != '—' else f"{edge_type}::"
        mapping[key] = w
    return mapping


def get_edge_weight(edge_data: dict, weight_map: Dict[str, float]) -> float:
    # prefer explicit mapping
    et = edge_data.get('edge_type', '')
    lab = edge_data.get('label', '')
    k1 = f"{et}::{lab}"
    k2 = f"{et}::"
    if k1 in weight_map:
        return float(weight_map[k1])
    if k2 in weight_map:
        return float(weight_map[k2])
    # fallback rules
    edge_info = f"{et} {lab}".lower()
    if 'isa' in edge_info or 'inheritance' in edge_info:
        return 5.0
    if 'correlateswith' in edge_info:
        return 3.0
    if 'directlyimpacts' in edge_info:
        return 5.0
    if 'directlycauses' in edge_info or 'directlycauses' in lab.lower():
        return 7.0
    return 1.0


def load_graph_for_qlearning(json_file='ontology_graph.json', weight_rule_file='OntoLable/weighting_rule.md',
                             reverse_edge_types=None, save_adjusted: str = None):
    """Load JSON graph, apply weights from weighting_rule.md and optionally reverse edges of given types.

    - json_file: path to the node-link JSON (merged_graph.json recommended)
    - weight_rule_file: path to weighting rule file
    - reverse_edge_types: set/list of edge_type strings to reverse direction for RL (e.g., ['instance_of'])
    - save_adjusted: if provided, path to save adjusted node-link JSON for downstream RL use
    """
    weight_map = parse_weighting_rule(weight_rule_file)

    with open(json_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # optionally modify link directions before constructing graph
    links = graph_data.get('links', [])
    if reverse_edge_types:
        new_links = []
        for l in links:
            et = l.get('edge_type', '')
            if et in reverse_edge_types:
                # swap source/target
                new = dict(l)
                new_source = l.get('target')
                new_target = l.get('source')
                new['source'] = new_source
                new['target'] = new_target
                new_links.append(new)
            else:
                new_links.append(l)
        graph_data['links'] = new_links

    # build directed nx graph
    G = json_graph.node_link_graph(graph_data, directed=True)

    # Debug: print weight mapping
    print(f"\n=== Weight Mapping from {weight_rule_file} ===")
    if weight_map:
        for key, val in weight_map.items():
            print(f"  {key} -> {val}")
    else:
        print("  No weight mappings found!")
    
    # assign weights
    weight_distribution = {}
    for u, v, data in G.edges(data=True):
        w = get_edge_weight(data, weight_map)
        # if the source node is labeled 'Instance' (generic), set weight 0
        u_label = G.nodes[u].get('label')
        if u_label == 'Instance':
            w = 0.0
        G[u][v]['weight'] = float(w)
        
        # Track weight distribution
        weight_distribution[w] = weight_distribution.get(w, 0) + 1
    
    print(f"\n=== Weight Distribution ===")
    for w in sorted(weight_distribution.keys()):
        print(f"  Weight {w}: {weight_distribution[w]} edges")

    # save adjusted graph json if requested
    if save_adjusted:
        data_out = json_graph.node_link_data(G)
        with open(save_adjusted, 'w', encoding='utf-8') as f:
            json.dump(data_out, f, ensure_ascii=False, indent=2)

    print(f"Loaded graph for Q-learning:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Print edge weight distribution
    weight_counts = {}
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        weight_counts[w] = weight_counts.get(w, 0) + 1

    print(f"\nEdge weight distribution:")
    for weight in sorted(weight_counts.keys()):
        print(f"  Weight {weight}: {weight_counts[weight]} edges")

    return G


def print_graph_info(G):
    print(f"\n{'='*60}")
    print(f"Graph Information")
    print(f"{'='*60}")

    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print(f"\nNode types:")
    for ntype, count in node_types.items():
        print(f"  {ntype}: {count}")

    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\nEdge types:")
    for etype, count in edge_types.items():
        print(f"  {etype}: {count}")

    print(f"\nSample nodes (first 5):")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        if i >= 5:
            break
        label = data.get('label', node)
        node_type = data.get('node_type', 'unknown')
        print(f"  {label} (type: {node_type})")

    print(f"\nAll edges with weights:")
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        u_label = G.nodes[u].get('label', u)
        v_label = G.nodes[v].get('label', v)
        weight = data.get('weight', 1)
        edge_label = data.get('label', data.get('edge_type', ''))
        print(f"  {u_label} -> {v_label} (weight: {weight}, label: {edge_label})")


if __name__ == "__main__":
    # by default use merged_graph.json (generated earlier) and weighting rules
    base = os.path.dirname(__file__)
    default_graph = os.path.join(base, 'merged_graph.json')
    default_weights = os.path.join(base, 'weighting_rule.md')
    # reverse instance_of edges so that instances -> classes for RL
    G = load_graph_for_qlearning(default_graph, default_weights, reverse_edge_types={'instance_of'},
                                 save_adjusted=os.path.join(base, 'rl_graph.json'))

    print_graph_info(G)

    print(f"\n{'='*60}")
    print("Graph is ready for Q-learning! Adjusted graph saved to OntoLable/rl_graph.json")
    print(f"{'='*60}")
