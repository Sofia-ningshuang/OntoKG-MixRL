import json
import os
from typing import List, Dict, Tuple
import networkx as nx
from networkx.readwrite import json_graph


def load_instance_paths(json_file: str) -> Dict:
    """Load q_instance_paths JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded instance paths from {json_file}")
    print(f"  Source graph: {data.get('source_graph', 'N/A')}")
    print(f"  Target: {data.get('target_label', 'N/A')}")
    print(f"  Total paths: {len(data.get('paths', []))}")
    
    return data


def load_graph_from_paths(paths_data: Dict) -> nx.DiGraph:
    """Load the corresponding trained graph from paths data."""
    source_graph = paths_data.get('source_graph', '')
    
    # Try to find the instance_trained graph
    if source_graph:
        base_name = os.path.splitext(os.path.basename(source_graph))[0]
        trained_file = f"{base_name}_instance_trained.json"
        
        if os.path.exists(trained_file):
            print(f"\nLoading trained graph from {trained_file}")
            with open(trained_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            G = json_graph.node_link_graph(graph_data, directed=True)
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            return G
    
    raise FileNotFoundError("Could not find trained graph file")


def find_evidence_nodes(G: nx.DiGraph, evidence_keywords: List[str]) -> List[str]:
    """Find nodes matching evidence keywords."""
    evidence_nodes = []
    
    for keyword in evidence_keywords:
        kw_lower = keyword.lower()
        for n, data in G.nodes(data=True):
            label = (data.get('label') or '').lower()
            text = (data.get('text') or '').lower()
            
            if kw_lower in label or kw_lower in text:
                evidence_nodes.append(n)
                node_type = data.get('node_type', 'unknown')
                display = data.get('text') or data.get('label') or str(n)
                print(f"  Found evidence node [{node_type}]: {display[:80]}")
                break
    
    return evidence_nodes


def adjust_evidence_weights(
    G: nx.DiGraph, 
    evidence_nodes: List[str], 
    weight_multiplier: float = 3.0
) -> nx.DiGraph:
    """Adjust weights for edges connected to evidence nodes."""
    print(f"\n==== Adjusting weights for evidence-related edges ====")
    print(f"Weight multiplier: {weight_multiplier}x")
    
    adjusted_edges = 0
    
    for node in evidence_nodes:
        # Adjust outgoing edges
        for u, v, data in G.out_edges(node, data=True):
            old_weight = data.get('weight', 1.0)
            new_weight = old_weight * weight_multiplier
            data['weight'] = new_weight
            data['evidence_adjusted'] = True
            adjusted_edges += 1
        
        # Adjust incoming edges
        for u, v, data in G.in_edges(node, data=True):
            old_weight = data.get('weight', 1.0)
            new_weight = old_weight * weight_multiplier
            data['weight'] = new_weight
            data['evidence_adjusted'] = True
            adjusted_edges += 1
    
    print(f"Adjusted {adjusted_edges} edges connected to evidence nodes")
    return G


def recalculate_path_scores(
    G: nx.DiGraph, 
    paths: List[Dict]
) -> List[Dict]:
    """Recalculate path scores with adjusted weights."""
    print("\n==== Recalculating path scores ====")
    
    for path_info in paths:
        path_nodes = path_info['path_nodes']
        
        # Recalculate weight sum
        weight_sum = 0.0
        evidence_edge_count = 0
        
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if G.has_edge(u, v):
                weight = G[u][v].get('weight', 1.0)
                weight_sum += weight
                
                if G[u][v].get('evidence_adjusted', False):
                    evidence_edge_count += 1
        
        path_info['adjusted_weight_sum'] = weight_sum
        path_info['evidence_edge_count'] = evidence_edge_count
        
        # Combined score: prioritize paths with evidence edges
        path_info['evidence_score'] = (
            evidence_edge_count * 100 +  # Heavily favor evidence paths
            path_info['q_sum'] * 10 +     # Q-learning score
            weight_sum                     # Adjusted weights
        )
    
    return paths


def evidence_based_optimization(
    paths_file: str,
    evidence_keywords: List[str],
    weight_multiplier: float = 3.0,
    top_n: int = 20
):
    """
    Perform evidence-based optimization on Q-learning paths.
    
    Args:
        paths_file: Path to q_instance_paths JSON file
        evidence_keywords: List of keywords to identify evidence nodes
        weight_multiplier: Multiplier for evidence-related edge weights
        top_n: Number of top paths to output
    """
    # Load paths and graph
    paths_data = load_instance_paths(paths_file)
    G = load_graph_from_paths(paths_data)
    
    # Find evidence nodes
    print("\n==== Finding evidence nodes ====")
    evidence_nodes = find_evidence_nodes(G, evidence_keywords)
    
    if not evidence_nodes:
        print("Warning: No evidence nodes found!")
        return
    
    print(f"\nFound {len(evidence_nodes)} evidence nodes")
    
    # Adjust weights
    G = adjust_evidence_weights(G, evidence_nodes, weight_multiplier)
    
    # Recalculate path scores
    paths = paths_data.get('paths', [])
    paths = recalculate_path_scores(G, paths)
    
    # Sort by evidence score
    paths.sort(key=lambda x: x['evidence_score'], reverse=True)
    
    # Output top N paths
    print(f"\n==== Top {top_n} evidence-optimized paths ====")
    for i, path_info in enumerate(paths[:top_n], 1):
        print(f"\n#{i}:")
        print(f"  Evidence edges: {path_info['evidence_edge_count']}")
        print(f"  Evidence score: {path_info['evidence_score']:.2f}")
        print(f"  Q-sum: {path_info['q_sum']:.3f}")
        print(f"  Original weight: {path_info['weight_sum']:.2f}")
        print(f"  Adjusted weight: {path_info['adjusted_weight_sum']:.2f}")
        print(f"  Length: {path_info['length']} (classes: {path_info['class_count']}, instances: {path_info['instance_count']})")
        print(f"  Path: {' -> '.join(path_info['path_labels'])}")
    
    # Save results
    output = {
        "source_file": paths_file,
        "evidence_keywords": evidence_keywords,
        "weight_multiplier": weight_multiplier,
        "evidence_nodes_count": len(evidence_nodes),
        "adjusted_edges_count": sum(1 for u, v, d in G.edges(data=True) if d.get('evidence_adjusted', False)),
        "top_paths": paths[:top_n],
    }
    
    base_name = os.path.splitext(os.path.basename(paths_file))[0]
    output_file = f"{base_name}_evidence_optimized.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n==== Saved evidence-optimized results to {output_file} ====")
    
    # Save adjusted graph
    graph_output_file = f"{base_name}_evidence_adjusted_graph.json"
    graph_data = json_graph.node_link_data(G)
    with open(graph_output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved adjusted graph to {graph_output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evidence-based path optimization')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to q_instance_paths JSON file')
    parser.add_argument('--multiplier', '-m', type=float, default=3.0,
                        help='Weight multiplier for evidence edges (default: 3.0)')
    parser.add_argument('--top', '-n', type=int, default=20,
                        help='Number of top paths to output (default: 20)')
    args = parser.parse_args()
    
    # Get input file
    input_file = args.input
    if not input_file:
        # Look for most recent q_instance_paths file
        base = os.path.dirname(__file__) or '.'
        instance_files = [f for f in os.listdir(base) 
                         if f.startswith('q_instance_paths_') and f.endswith('.json')]
        if instance_files:
            instance_files.sort(key=lambda f: os.path.getmtime(os.path.join(base, f)), reverse=True)
            input_file = os.path.join(base, instance_files[0])
            print(f"Using most recent instance paths: {instance_files[0]}")
        else:
            raise FileNotFoundError("No q_instance_paths files found. Please specify --input")
    
    # Define evidence keywords based on the case study
    # These match the evidence chain provided by the user
    evidence_keywords = [
        "TechnicalSchemeDefect",                       # Technical defect
        "Blue Ridge Design",                           # Design company
        "conflicting welding documents",               # Documentation issue
        "HazardCondition",                             # Hazard condition
        "Pre-Con",                                     # Contractor
        "failed to comply with welding requirement",   # Non-compliance
        "TriggerEvent",                                # Trigger event
        "Crane hook",                                  # Equipment
        "accidentally caught",                         # Accident mechanism
        "lifting insert",                              # Component
        "double tee #392",                            # Structural element
        "operator continued lifting",                  # Operator action
        "on-site warnings",                            # Warning ignored
        "QualityDamage",                               # Quality damage
        "Partial collapse",                            # Structural failure
        "double tees #392/#393",                      # Failed elements
        "fourth-floor support system",                 # Location
        "QualityAccident",                             # Accident type
        "4 workers killed",                            # Casualties
        "collapse of precast double tees",            # Failure mode
    ]
    
    print("="*80)
    print("Evidence-Based Path Optimization")
    print("="*80)
    print("\nEvidence chain:")
    print("  1. TechnicalSchemeDefect (Blue Ridge Design provided conflicting welding documents)")
    print("  2. HazardCondition (Pre-Con failed to comply with welding requirement)")
    print("  3. TriggerEvent (Crane hook accidentally caught the lifting insert of double tee #392; operator continued lifting despite on-site warnings)")
    print("  4. QualityDamage (Partial collapse of double tees #392/#393 from the fourth-floor support system)")
    print("  5. QualityAccident (4 workers killed in the collapse of precast double tees)")
    print("="*80 + "\n")
    
    evidence_based_optimization(
        paths_file=input_file,
        evidence_keywords=evidence_keywords,
        weight_multiplier=args.multiplier,
        top_n=args.top
    )
