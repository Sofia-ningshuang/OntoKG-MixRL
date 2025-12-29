import json
import os
import re
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx


def configure_chinese_font():
	"""Configure matplotlib to support Chinese characters."""
	try:
		# List of common Chinese fonts on Windows and other platforms
		chinese_fonts = [
			'SimHei',           # Windows
			'Microsoft YaHei',  # Windows
			'STHeiti',          # macOS
			'PingFang SC',      # macOS
			'WenQuanYi Micro Hei',  # Linux
			'Noto Sans CJK SC', # Linux
		]
		
		# Try to find an available Chinese font
		available_fonts = [f.name for f in fm.fontManager.ttflist]
		
		for font in chinese_fonts:
			if font in available_fonts:
				plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
				plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
				print(f"Using Chinese font: {font}")
				return font
		
		# If no specific font found, try generic approach
		plt.rcParams['axes.unicode_minus'] = False
		print("Warning: No specific Chinese font found, using default")
		return None
		
	except Exception as e:
		print(f"Error configuring Chinese font: {e}")
		return None


def contains_chinese(text: str) -> bool:
	"""Check if text contains Chinese characters."""
	if not text:
		return False
	return bool(re.search(r'[\u4e00-\u9fff]', text))


def load_graph_from_json(path: str) -> Tuple[nx.Graph, Dict[str, Any]]:
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)

	G = nx.DiGraph()
	nodes = data.get('nodes', [])
	links = data.get('links', [])

	for n in nodes:
		nid = n.get('id')
		G.add_node(nid, **n)

	for l in links:
		src = l.get('source')
		tgt = l.get('target')
		if src and tgt:
			# keep edge attributes
			G.add_edge(src, tgt, **l)

	return G, data


def color_map_for_nodes(G: nx.Graph):
	color_map = []
	for n, d in G.nodes(data=True):
		t = d.get('node_type', '').lower()
		if t == 'class':
			color_map.append('#4C78A8')
		elif t == 'instance':
			color_map.append('#F58518')
		else:
			color_map.append('#54A24B')
	return color_map


def color_map_for_edges(G: nx.Graph):
	colors = []
	widths = []
	for u, v, d in G.edges(data=True):
		et = d.get('edge_type', '').lower()
		if et == 'instance_of':
			colors.append('#d62728')
			widths.append(1.5)
		elif et == 'inheritance':
			colors.append('#7f7f7f')
			widths.append(1.0)
		elif et == 'object_property':
			colors.append('#2ca02c')
			widths.append(1.2)
		else:
			colors.append('#000000')
			widths.append(0.8)
	return colors, widths


def compute_positions(G: nx.Graph, scale: float = 1.0):
	# Separate nodes into classes and instances
	class_nodes = []
	instance_nodes = []
	other_nodes = []
	
	for n in G.nodes():
		node_type = G.nodes[n].get('node_type', '').lower()
		if node_type == 'class':
			class_nodes.append(n)
		elif node_type == 'instance':
			instance_nodes.append(n)
		else:
			other_nodes.append(n)
	
	print(f"Layout: {len(class_nodes)} classes (top layer), {len(instance_nodes)} instances (bottom layer)")
	
	pos = {}
	
	# Create subgraphs for layout
	if class_nodes:
		G_classes = G.subgraph(class_nodes)
		try:
			pos_classes = nx.spring_layout(G_classes, k=1.0, iterations=300, scale=scale)
		except:
			pos_classes = {n: (i * scale / max(len(class_nodes), 1), 0) for i, n in enumerate(class_nodes)}
		
		# Place classes in upper layer (positive y)
		for n, (x, y) in pos_classes.items():
			pos[n] = (x, y + 2.0 * scale)  # Shift up
	
	if instance_nodes:
		G_instances = G.subgraph(instance_nodes)
		try:
			pos_instances = nx.spring_layout(G_instances, k=1.0, iterations=300, scale=scale)
		except:
			pos_instances = {n: (i * scale / max(len(instance_nodes), 1), 0) for i, n in enumerate(instance_nodes)}
		
		# Place instances in lower layer (negative y)
		for n, (x, y) in pos_instances.items():
			pos[n] = (x, y - 2.0 * scale)  # Shift down
	
	# Place other nodes in the middle
	if other_nodes:
		for i, n in enumerate(other_nodes):
			pos[n] = (i * scale / max(len(other_nodes), 1), 0)
	
	return pos


def visualize_graph(json_path: str, out_png: str = None, figsize=(16, 12), show: bool = True):
	G, raw = load_graph_from_json(json_path)
	
	# Check if graph contains Chinese text and configure font if needed
	has_chinese = False
	for n, d in G.nodes(data=True):
		label = d.get('label', '')
		text = d.get('text', '')
		if contains_chinese(label) or contains_chinese(text):
			has_chinese = True
			break
	
	if has_chinese:
		configure_chinese_font()

	pos = compute_positions(G, scale=3.0)

	node_colors = color_map_for_nodes(G)
	edge_colors, edge_widths = color_map_for_edges(G)

	plt.figure(figsize=figsize)
	ax = plt.gca()
	ax.set_title(os.path.basename(json_path))
	ax.set_axis_off()

	# draw nodes
	nx.draw_networkx_nodes(G, pos,
						   node_color=node_colors,
						   node_size=300,
						   alpha=0.9)

	# draw edges
	nx.draw_networkx_edges(G, pos,
						   edge_color=edge_colors,
						   width=edge_widths,
						   arrowsize=12,
						   alpha=0.8)

	# draw small labels for instance/class type to avoid crowding
	def short(s: str, length: int = 80) -> str:
		s = s.replace('\n', ' ').strip()
		return (s[:length] + '...') if len(s) > length else s

	labels = {}
	for n, d in G.nodes(data=True):
		node_label = d.get('label')
		if d.get('node_type') == 'instance':
			# prefer the 'text' field for instance nodes when available
			text = d.get('text')
			if text:
				labels[n] = short(text, 100)
			else:
				labels[n] = node_label
		elif d.get('node_type') == 'class':
			labels[n] = node_label

	nx.draw_networkx_labels(G, pos, labels, font_size=8)

	plt.tight_layout()
	if out_png:
		plt.savefig(out_png, dpi=200)
	if show:
		plt.show()
	plt.close()


if __name__ == '__main__':
	import argparse

	p = argparse.ArgumentParser(description='Visualize graph JSON (nodes/links)')
	p.add_argument('--json', help='path to graph json', default=None)
	p.add_argument('--out', help='output png path', default=None)
	p.add_argument('--noshow', action='store_true', help='do not show interactive window')
	args = p.parse_args()

	base = os.path.dirname(__file__)
	json_path = args.json or os.path.join(base, 'merged_graph.json')
	out_png = args.out or os.path.join(base, 'merged_graph.png')

	visualize_graph(json_path, out_png=out_png, show=(not args.noshow))

