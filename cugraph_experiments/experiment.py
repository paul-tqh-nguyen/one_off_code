
# python3 ./experiment.py | grep -v "forward_output"

import networkx as nx
import metagraph as mg
import cugraph

dpr = mg.resolver

print(
'''
0 <--2-- 1 
|      ^ | 
|     /  | 
1    7   3 
|   /    | 
v        v 
3 --8--> 4 
'''
)

ebunch = [
    (0, 3, 1),
    (1, 0, 2),
    (1, 4, 3),
    (3, 1, 7),
    (3, 4, 8),
]
nx_graph = nx.DiGraph()
nx_graph.add_weighted_edges_from(ebunch)
graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph, weight_label="weight")
k = 4
enable_normalization = False
include_endpoints = False

graph_cugraph_wrapped = dpr.translate(graph, dpr.types.Graph.CuGraphType)
cugraph_graph = graph_cugraph_wrapped.value

mg_nx_result = dpr.algo.vertex_ranking.betweenness_centrality(graph, k, enable_normalization, include_endpoints)
mg_cugraph_result = dpr.algo.vertex_ranking.betweenness_centrality(graph_cugraph_wrapped, k, enable_normalization, include_endpoints)

nx_result = nx.betweenness_centrality(nx_graph, k, enable_normalization, include_endpoints)
cugraph_result = cugraph.betweenness_centrality(cugraph_graph, normalized=enable_normalization, endpoints=include_endpoints)

print(f"cugraph_graph {repr(cugraph_graph)}")
print('\n'*3)
print(f"type(cugraph_graph) {repr(type(cugraph_graph))}")
print(f"nx_result               {repr(nx_result)}")
print(f"mg_nx_result.value      {repr(mg_nx_result.value)}")
print(f"cugraph_result          \n{repr(cugraph_result)}")
print(f"mg_cugraph_result.value \n{repr(mg_cugraph_result.value)}")

for _ in range(4):
    cugraph_result = cugraph.betweenness_centrality(cugraph_graph, normalized=enable_normalization, endpoints=include_endpoints)
    print(f"cugraph_result {repr(cugraph_result)}")
