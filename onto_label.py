import rdflib
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import os

# Load TTL file 加载本体文件.ttl
g = rdflib.Graph()
# Resolve paths relative to this script so outputs land in the OntoLable folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TTL file may live in a `src` subfolder; try several candidate paths
ttl_candidates = [
    os.path.join(BASE_DIR, "CQAR_Ontology_2.0.ttl"),
    os.path.join(BASE_DIR, "src", "CQAR_Ontology_2.0.ttl"),
    os.path.join(BASE_DIR, "..", "src", "CQAR_Ontology_2.0.ttl"),
]
ttl_path = None
for p in ttl_candidates:
    if os.path.exists(p):
        ttl_path = p
        break
if ttl_path is None:
    # last resort: try the filename alone (cwd)
    if os.path.exists("CQAR_Ontology_2.0.ttl"):
        ttl_path = "CQAR_Ontology_2.0.ttl"
if ttl_path is None:
    raise FileNotFoundError("Could not find CQAR_Ontology_2.0.ttl; looked in: %s" % (ttl_candidates,))

print(f"Loading TTL from: {ttl_path}")
g.parse(ttl_path, format="turtle")

# Namespaces 命名空间
OWL = rdflib.OWL
RDFS = rdflib.RDFS
rdf_type = rdflib.RDF.type

# Helper function to extract name from URI 提取本体中的URI以保证元素唯一性
def get_name_from_uri(uri):
    uri_str = str(uri)
    # Extract the last part after # or /
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    elif '/' in uri_str:
        return uri_str.split('/')[-1]
    return uri_str

# Prepare graph 初始化图
G = nx.DiGraph()

# Helper function to check if a class is a subclass of "Instance" 检查是否有"实例"的子类
def is_subclass_of_instance(cls, g, classes):
    """Recursively check if a class is a subclass of 'Instance'"""
    # Check if this class name is "Instance" 检查是否有"实例"类
    if get_name_from_uri(cls) == "Instance":
        return True
    
    # Check all parent classes 对所有父类进行检查，存在"实例"的子类为真
    for parent in g.objects(cls, RDFS.subClassOf):
        if (parent, rdf_type, OWL.Class) in g:
            if is_subclass_of_instance(parent, g, classes):
                return True
    return False

# 1) Add Classes as nodes 将本体中的"类"增加为节点
classes = set(g.subjects(rdf_type, OWL.Class))
for cls in classes:
    # Check if this class is a subclass of "Instance" 检查是否是"实例"的子类，如果是存为实例类，如果不是存为类
    if is_subclass_of_instance(cls, g, classes):
        G.add_node(str(cls), node_type='instance_class', label=get_name_from_uri(cls))
    else:
        G.add_node(str(cls), node_type='class', label=get_name_from_uri(cls))

# 1.5) Add Instances as nodes 将本体中的"实例"增加为节点
instances = set()
for cls in classes:
    for instance in g.subjects(rdf_type, cls):
        # Check if it's an instance (not a class itself) 检查是否是"实例"，而不是"类"
        if (instance, rdf_type, OWL.Class) not in g:
            instances.add(instance)
            G.add_node(str(instance), node_type='instance', label=get_name_from_uri(instance))
            # Add edge from instance to its class 建立实例所属边
            G.add_edge(str(instance), str(cls), edge_type='instance_of')

# 2) Add Class inheritances as edges (reversed: child -> parent) 将类的继承建立为边
for cls in classes:
    for parent in g.objects(cls, RDFS.subClassOf):
        if (parent, rdf_type, OWL.Class) in g:
            # Reverse direction: subclass (child) -> superclass (parent) 逆序继承边，为了与多层知识图谱的语义关系保持一致（见OntoKG.png）
            G.add_edge(str(cls), str(parent), edge_type='inheritance')

# 3) Add Object Properties as edges 将本体中的"对象属性"（即三元组的谓语）建立为边
object_properties = set(g.subjects(rdf_type, OWL.ObjectProperty))
for prop in object_properties:
    for domain in g.objects(prop, RDFS.domain):
        for range_ in g.objects(prop, RDFS.range):
            if (domain, rdf_type, OWL.Class) in g and (range_, rdf_type, OWL.Class) in g:
                label = get_name_from_uri(prop)
                # Reverse direction for properties containing 'IsA' 逆序IsA边，为了与多层知识图谱的语义关系保持一致（见OntoKG.png）
                if 'isa' in label.lower():
                    G.add_edge(str(range_), str(domain), edge_type='object_property', label=label)
                else:
                    G.add_edge(str(domain), str(range_), edge_type='object_property', label=label)

# 4) Define colors for different node types 定义不同类型的节点显示颜色
color_map = []
for node in G.nodes(data=True):
    if node[1]['node_type'] == 'instance_class':
        color_map.append('red')
    elif node[1]['node_type'] == 'class':
        color_map.append('skyblue')
    elif node[1]['node_type'] == 'instance':
        color_map.append('orange')
    else:
        color_map.append('gray')

# 5) Print the graph (try to avoid overlaps) 输出图（避免覆盖），此项调整图的代码过长，实际应用可简化，阅读代码逻辑可直接跳至270行
plt.figure(figsize=(18, 14))

# Try Graphviz 'neato' layout for better non-overlapping positions, fall back to kamada_kawai or spring 调用Graphviz进行节点排布
pos = None
try:
    from networkx.drawing.nx_agraph import graphviz_layout

    pos = graphviz_layout(G, prog='neato')
except Exception:
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, k=0.7, iterations=200)

edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
# Create labels dict with node names from the label attribute
labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}

# node sizes by type
node_sizes = []
for n, d in G.nodes(data=True):
    if d.get('node_type') == 'instance_class':
        node_sizes.append(900)
    elif d.get('node_type') == 'class':
        # slightly reduce class node size to reduce visual overlap
        node_sizes.append(580)
    elif d.get('node_type') == 'instance':
        node_sizes.append(400)
    else:
        node_sizes.append(300)

# We'll draw nodes and edges after refining positions so the final plot
# uses the redistributed coordinates.

# Improve separation for red nodes (instance_class) to reduce overlaps
def separate_special_nodes(pos, G, special_type='instance_class', iterations=50, scale=0.12):
    """Adjust positions in-place in `pos` to push nodes of special_type away from nearby nodes.

    - pos: dict node -> (x,y)
    - G: graph with node attribute 'node_type'
    - special_type: the node_type to separate (red nodes)
    - iterations: number of relaxation steps
    - scale: step size multiplier for displacement
    """
    try:
        import math
        import random
    except Exception:
        math = None
        random = None

    nodes = list(G.nodes())
    N = len(nodes)
    coords = {n: list(pos[n]) for n in nodes}

    # compute typical nearest-neighbor distance to set a min separation
    dists = []
    for i, a in enumerate(nodes):
        ax, ay = coords[a]
        min_d = None
        for j, b in enumerate(nodes):
            if i == j:
                continue
            bx, by = coords[b]
            dx = ax - bx
            dy = ay - by
            dist = (dx * dx + dy * dy) ** 0.5
            if min_d is None or dist < min_d:
                min_d = dist
        if min_d is not None:
            dists.append(min_d)
    if dists:
        avg_nn = sum(dists) / len(dists)
    else:
        avg_nn = 0.1
    min_sep = avg_nn * 0.9

    special_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == special_type]
    if not special_nodes:
        return

    for _ in range(iterations):
        moved = False
        for s in special_nodes:
            sx, sy = coords[s]
            dx_total = 0.0
            dy_total = 0.0
            for o in nodes:
                if o == s:
                    continue
                ox, oy = coords[o]
                dx = sx - ox
                dy = sy - oy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-6:
                    # jitter
                    jitter_x = (random.random() - 0.5) * 1e-2 if random else 1e-3
                    jitter_y = (random.random() - 0.5) * 1e-2 if random else 1e-3
                    dx_total += jitter_x
                    dy_total += jitter_y
                    moved = True
                elif dist < min_sep:
                    # push away proportionally to shortfall
                    push = (min_sep - dist) / (dist + 1e-6)
                    dx_total += (dx / (dist + 1e-9)) * push
                    dy_total += (dy / (dist + 1e-9)) * push
                    moved = True
            if moved:
                coords[s][0] = sx + dx_total * scale
                coords[s][1] = sy + dy_total * scale

    # write back
    for n in nodes:
        pos[n] = tuple(coords[n])


# run separation to reduce overlapping red nodes
# First, run an additional distribution/refinement step so all nodes fill the space more evenly.
def distribute_nodes(pos, G, scale_factor=1.8, spring_iters=250, seed=42):
    """Refine and expand node positions so nodes are more evenly distributed.

    Uses the current `pos` as an initial layout and runs a spring_layout
    with a tuned `k` based on node count, then normalizes coordinates to
    better fill the drawing area.
    """
    import math

    n = max(1, G.number_of_nodes())
    # heuristic for k: sqrt(1/n) is default-ish; scale up to spread more
    k = math.sqrt(1.0 / n) * (scale_factor)
    try:
        pos = nx.spring_layout(G, k=k, pos=pos, iterations=spring_iters, seed=seed)
    except Exception:
        # fallback: keep original pos
        pass

    # normalize to [-1,1] box with margins so plotting uses full area
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = maxx - minx if maxx > minx else 1.0
    dy = maxy - miny if maxy > miny else 1.0
    for nnode in pos:
        x, y = pos[nnode]
        nxn = (x - minx) / dx
        nyn = (y - miny) / dy
        # remap to -1..1 range (centered)
        pos[nnode] = ((nxn - 0.5) * 2.0, (nyn - 0.5) * 2.0)
    return pos


# distribute nodes globally to fill space
pos = distribute_nodes(pos, G, scale_factor=1.8, spring_iters=250)

# run separation to reduce overlapping red nodes
separate_special_nodes(pos, G, special_type='instance_class', iterations=80, scale=0.14)

# Also separate class (blue) nodes to reduce their overlap
# Use slightly different parameters so blue nodes spread modestly
separate_special_nodes(pos, G, special_type='class', iterations=120, scale=0.10)

# Draw nodes and edges after repositioning
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_sizes)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=12)

# Draw labels with white bbox to reduce overlaps
text_items = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
for t in text_items.values():
    t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.25))

# Draw edge labels separately
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

# 6) Export graph to JSON file 输出图并生成JSON文件
graph_data = json_graph.node_link_data(G, edges="links")
output_file = os.path.join(BASE_DIR, 'onto_graph.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2, ensure_ascii=False)
print(f"Graph exported to {output_file}")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

plt.axis('off')
plt.tight_layout()
plt.show()