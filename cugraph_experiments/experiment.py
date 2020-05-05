
import networkx as nx
import metagraph as mg
import cugraph

r = mg.resolver

if 4:
    ebunch = [
(0, 3, 1),
(3, 1, 7),
(3, 4, 8),
(1, 0, 2),
(1, 4, 3),
(4, 5, 9),
(2, 4, 4),
(2, 5, 5),
(2, 7, 6),
(5, 6, 10),
(6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = r.wrapper.Graph.NetworkXGraph(nx_graph, weight_label='weight')
    k = 8
    enable_normalization = False
    include_endpoints = False


r.plan.translate(graph, r.types.Graph.CuGraphType)

graph_cugraph = r.translate(graph, r.types.Graph.CuGraphType)

print(f'nx.betweenness_centrality(graph.value, k, enable_normalization, include_endpoints) \n{repr(nx.betweenness_centrality(graph.value, k, enable_normalization, include_endpoints))}')
print(f'cugraph.betweenness_centrality(graph_cugraph.value, normalized=enable_normalization, endpoints=include_endpoints) \n{repr(cugraph.betweenness_centrality(graph_cugraph.value, normalized=enable_normalization, endpoints=include_endpoints))}')



