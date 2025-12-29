import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

# Base directory for this script (OntoLable folder)
BASE_DIR = Path(__file__).resolve().parent


def stable_color(name: str) -> str:
    """Stable color per-name: returns #RRGGBB.
    """
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))


def load_ontology(json_path: Path):
    """Load a simple JSON list of class dicts (uri/local_name/superclasses).

    This is the same small helper used previously for taxonomy generation.
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    uri_to_name = {}
    parents = defaultdict(set)    # child_uri -> set(parent_uri)
    children = defaultdict(set)   # parent_uri -> set(child_uri)

    # Support two formats:
    # 1) a list of class dicts: [{"uri":..., "local_name":..., "superclasses":[...]}]
    # 2) a node-link graph dict with 'nodes' and 'links' as in ontology_graph.json
    if isinstance(raw, dict) and "nodes" in raw:
        nodes = raw.get("nodes", [])
        links = raw.get("links", [])
        # collect class nodes
        for n in nodes:
            if n.get("node_type") != "class":
                continue
            uri = n.get("id")
            name = n.get("label") or uri
            uri_to_name[uri] = name
        all_uris = set(uri_to_name.keys())
        for e in links:
            if e.get("edge_type") != "inheritance":
                continue
            src = e.get("source")
            tgt = e.get("target")
            # src is child, tgt is parent
            if src in all_uris and tgt in all_uris:
                parents[src].add(tgt)
                children[tgt].add(src)
    else:
        data = raw
        for item in data:
            # item could be a dict with 'uri' or a simple string; guard accordingly
            if not isinstance(item, dict):
                continue
            uri = item.get("uri") or item.get("id")
            if not uri:
                continue
            name = item.get("local_name") or item.get("label") or uri.split("/")[-1]
            uri_to_name[uri] = name

        all_uris = set(uri_to_name.keys())
        for item in data:
            if not isinstance(item, dict):
                continue
            u = item.get("uri") or item.get("id")
            for p in item.get("superclasses") or []:
                if p in all_uris:
                    parents[u].add(p)
                    children[p].add(u)

    return uri_to_name, parents, children


def find_roots(uri_to_name, parents):
    all_uris = set(uri_to_name.keys())
    roots = [u for u in all_uris if len(parents.get(u, set())) == 0]
    roots.sort(key=lambda x: uri_to_name[x].lower())
    return roots


def build_tree(children, root_uri):
    def rec(u):
        sub = {}
        kids = sorted(children.get(u, set()), key=lambda x: x)
        for k in kids:
            sub[k] = rec(k)
        return sub
    return {root_uri: rec(root_uri)}


def emit_taxonomy_choice(uri_to_name, tree, indent="    "):
    lines = []
    for uri, sub in tree.items():
        name = xml_escape(uri_to_name[uri])
        if sub:
            lines.append(f'{indent}<Choice value="{name}">')
            lines.extend(emit_taxonomy_choice(uri_to_name, sub, indent + "  "))
            lines.append(f"{indent}</Choice>")
        else:
            lines.append(f'{indent}<Choice value="{name}"/>')
    return lines


def generate_xml(uri_to_name, parents, children, out_path: Path):
    roots = find_roots(uri_to_name, parents)

    lines = []
    lines.append("<View>")

    # Coarse labels (top-level roots) - two-level taxonomy's coarse layer
    lines.append('  <Labels name="coarse" toName="text">')
    for r in roots:
        name = xml_escape(uri_to_name[r])
        color = stable_color(uri_to_name[r])
        lines.append(f'    <Label value="{name}" background="{color}"/>')
    lines.append('  </Labels>')
    lines.append("")

    # Fine-grained taxonomy
    lines.append('  <Taxonomy name="fine" toName="text" perRegion="true">')
    for r in roots:
        tree = build_tree(children, r)
        lines.extend(emit_taxonomy_choice(uri_to_name, tree, indent="    "))
    lines.append('  </Taxonomy>')
    lines.append("")

    # One-level general labels (all classes) will be inserted by caller when requested
    lines.append('  <Text name="text" value="$text"/>')
    lines.append("</View>")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(roots)


def generate_entity_relation_xml(graph_json: Path, out_path: Path, only_classes: bool = True):
    """Generate a Label-Studio config for entity and relation labeling from `ontology_graph.json`.

    Entities are taken from nodes where `node_type` == 'class' (if `only_classes`).
    Relations are taken from unique `edge_type` values found in `links`.
    """
    j = json.loads(graph_json.read_text(encoding="utf-8"))

    nodes = j.get("nodes", [])
    links = j.get("links", [])

    labels = []
    seen = set()
    for n in nodes:
        if only_classes and n.get("node_type") != "class":
            continue
        lab = n.get("label") or n.get("id")
        if not lab:
            continue
        if lab in seen:
            continue
        seen.add(lab)
        labels.append(lab)
    labels.sort(key=lambda x: x.lower())

    rel_types = []
    rseen = set()
    for e in links:
        et = e.get("edge_type") or e.get("label")
        if not et:
            continue
        if et in rseen:
            continue
        rseen.add(et)
        rel_types.append(et)
    rel_types.sort()

    lines = []
    lines.append("<View>")
    lines.append('  <Labels name="label" toName="text">')
    for lab in labels:
        color = stable_color(lab)
        lines.append(f'    <Label value="{xml_escape(lab)}" background="{color}"/>')
    lines.append('  </Labels>')
    lines.append("")

    if rel_types:
        lines.append('  <Relations name="relation" toName="label">')
        for rt in rel_types:
            lines.append(f'    <Relation value="{xml_escape(rt)}"/>')
        lines.append('  </Relations>')
        lines.append("")

    lines.append('  <Text name="text" value="$text"/>')
    lines.append("</View>")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(labels), len(rel_types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(BASE_DIR / "onto_graph.json"), help="path to cqtr_classes_full.json or ontology_graph.json")
    ap.add_argument("--out", default=str(BASE_DIR / "label_studio_two_level.xml"), help="output xml path for two-level taxonomy")
    ap.add_argument("--no-level-out", default=str(BASE_DIR / "label_studio_no_level.xml"),
                    help="output xml path for one-level labels (no-level)")
    ap.add_argument("--add-general", action="store_true", help="also include a one-level general Labels block containing all classes")
    args = ap.parse_args()

    # 1) Always generate two-level taxonomy XML
    uri_to_name, parents, children = load_ontology(Path(args.json))
    n_roots = generate_xml(uri_to_name, parents, children, Path(args.out))
    # optionally add one-level general labels (all classes)
    if args.add_general:
        txt = Path(args.out).read_text(encoding="utf-8")
        parts = txt.split('\n')
        idx = None
        for i, l in enumerate(parts):
            if l.strip().startswith('<Taxonomy'):
                idx = i
                break
        if idx is None:
            for i, l in enumerate(parts):
                if l.strip().startswith('<Text'):
                    idx = i
                    break
        general_block = []
        general_block.append('  <Labels name="label" toName="text">')
        all_labels = sorted(uri_to_name.values(), key=lambda x: x.lower())
        for lab in all_labels:
            color = stable_color(lab)
            general_block.append(f'    <Label value="{xml_escape(lab)}" background="{color}"/>')
        general_block.append('  </Labels>')
        general_block.append('')
        if idx is None:
            parts = parts + general_block
        else:
            parts = parts[:idx] + general_block + parts[idx:]
        Path(args.out).write_text('\n'.join(parts), encoding='utf-8')
    print(f"[OK] taxonomy roots={n_roots}, wrote: {args.out}")

    # 2) Also generate a one-level labels-only XML (entities + relations)
    labels_count, rel_count = generate_entity_relation_xml(Path(args.json), Path(args.no_level_out))
    print(f"[OK] entities={labels_count}, relations={rel_count}, wrote: {args.no_level_out}")


if __name__ == "__main__":
    main()
