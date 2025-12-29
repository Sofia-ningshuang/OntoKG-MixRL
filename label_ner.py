import json
import os
import uuid
from typing import Dict, Any


def load_json(path: str):
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def save_json(obj: Any, path: str):
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def generate_instance_id(class_id: str, label: str) -> str:
	base = class_id.rsplit('/', 1)[0]
	short = uuid.uuid4().hex[:10]
	# follow pattern TE.<Label>.<short>
	return f"{base}/TE.{label}.{short}"


def map_labels_to_onto(case_path: str, onto_path: str, out_path: str) -> Dict[str, int]:
	case_data = load_json(case_path)
	onto = load_json(onto_path)

	nodes = onto.setdefault('nodes', [])
	links = onto.setdefault('links', [])

	# build class label -> class id map
	class_map = {n['label']: n['id'] for n in nodes if n.get('node_type') == 'class' and 'label' in n}

	added_nodes = 0
	added_links = 0

	for task in case_data:
		for ann in task.get('annotations', []) or []:
			for res in ann.get('result', []) or []:
				if res.get('type') != 'labels':
					continue
				value = res.get('value', {})
				labels = value.get('labels', [])
				text = value.get('text')
				for label in labels:
					class_id = class_map.get(label)
					if not class_id:
						continue
					instance_id = generate_instance_id(class_id, label)
					node = {
						'node_type': 'instance',
						'label': label,
						'id': instance_id,
					}
					if text:
						node['text'] = text
					nodes.append(node)
					# instance_of is an is_a relationship: instance -> class
					link = {
						'edge_type': 'instance_of',
						'source': instance_id,
						'target': class_id,
					}
					links.append(link)
					added_nodes += 1
					added_links += 1

	save_json(onto, out_path)
	return {'nodes': added_nodes, 'links': added_links}


def default_paths():
	base = os.path.dirname(__file__)

	def find_file(name):
		candidates = [
			os.path.join(base, name),
			os.path.join(base, 'src', name),
			os.path.join(base, '..', 'src', name),
			name,  # cwd
		]
		for p in candidates:
			if os.path.exists(p):
				return p
		# fallback to first candidate
		return candidates[0]

	return (
		find_file('case01.json'),
		find_file('onto_graph.json'),
		os.path.join(base, 'merged_graph.json'),
	)


if __name__ == '__main__':
	import argparse

	p = argparse.ArgumentParser(description='Map labeled data to ontology graph and create instances.')
	p.add_argument('--case', help='path to case json', default=None)
	p.add_argument('--onto', help='path to ontology graph json', default=None)
	p.add_argument('--out', help='output merged graph json', default=None)
	args = p.parse_args()

	case_path, onto_path, out_path = default_paths()
	if args.case:
		case_path = args.case
	if args.onto:
		onto_path = args.onto
	if args.out:
		out_path = args.out

	stats = map_labels_to_onto(case_path, onto_path, out_path)
	print(f"Added {stats['nodes']} instance nodes and {stats['links']} instance_of links to {out_path}")
