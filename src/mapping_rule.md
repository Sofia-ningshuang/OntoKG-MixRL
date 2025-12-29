
first
in case01.json
search each "labels" 
for example, "labels": ["StructuralSafetyAccident"]

then
search and match "label" in onto_graph.json
for example,
"node_type": "class",
"label": "StructuralSafetyAccident",
"id": "http://www.semanticweb.org/sofia/ontologies/2025/9/cqtr-ongtology-1.1/StructuralSafetyAccident"

when the content of "labels" in case01.json matches the content of "label" in onto_graph.json
add one node in the onto_graph.json with type of
"node_type": "instance"
"label": for example, "StructuralSafetyAccident"
"id": for example, generated id according to the class
"text": for example, "the parking garage under construction at the Miami-Dade College (MDC) in Doral, Florida, partially collapsed killing four workers"

meanwhile, add related edges in the onto_graph.json with type of
"edge_type": "instance_of"
"source": the class id
"target": the new generated instance id