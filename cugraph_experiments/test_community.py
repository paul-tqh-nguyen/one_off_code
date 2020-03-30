
import cugraph
import cudf
import random

random.seed(1234)

# Helpers

from functools import reduce

def n_choose_k(n, k):
    k = min(k, n-k)
    numerator = reduce(int.__mul__, range(n, n-k, -1), 1)
    denominator = reduce(int.__mul__, range(1, k+1), 1)
    return numerator // denominator

# Louvain Community Detection

# @todo get this working
#Not yet working
# def test_louvain_case_1():
#     '''Two cliques. Weight uninitialized (implicitly all equal weight). Various number of connections between two cliques.'''
    
#     clique_a_sources = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8]
#     clique_a_destinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]
    
#     clique_b_node_id_delta = 100
#     clique_b_sources = [node + clique_b_node_id_delta for node in clique_a_sources]
#     clique_b_destinations = [node + clique_b_node_id_delta for node in clique_a_destinations]

#     clique_a_nodes = list(set(clique_a_sources+clique_a_destinations))
#     clique_b_nodes = list(set(clique_b_sources+clique_b_destinations))
    
#     sources = list(clique_a_sources + clique_b_sources)
#     destinations = list(clique_a_destinations + clique_b_destinations)

#     inter_clique_edges = set()

#     num_iterations = len(clique_a_nodes)
#     for index in range(num_iterations):
#         random_a_node = random.choice(clique_a_nodes)
#         random_b_node = random.choice(clique_b_nodes)
#         inter_clique_edge = (random_a_node, random_b_node)
#         inter_clique_edges.add(inter_clique_edge)
#         sources.append(random_a_node)
#         destinations.append(random_b_node)
#         gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
#         g = cugraph.Graph()
#         g.from_cudf_edgelist(gdf, source="source", destination="destination")
#         node_to_label_map, modularity_score = cugraph.louvain(g)
#         print('\n\n\n')
#         print(f'Iteration: {index}')
#         print(f'node_to_label_map {node_to_label_map}')
#         print(f'modularity_score {modularity_score}')
#         vertices = list(node_to_label_map['vertex'])
#         labels = list(node_to_label_map['partition'])
#         node_to_label_map_as_dict = {vertex: label for vertex, label in zip(vertices, labels)}
#         edges = list(zip(sources, destinations))
#         for a_node in clique_a_nodes:
#             if node_to_label_map_as_dict[a_node] != 0:
#                 print(f'{a_node}->{node_to_label_map_as_dict[a_node]} {[e for e in edges if a_node in e]}')

# Spectral Clustering

# @todo get this working
# cugraph.spectralBalancedCutClustering
# cugraph.spectralModularityMaximizationClustering
# cugraph.spectralBalancedCutClustering + cugraph.analyzeClustering_edge_cut
# cugraph.spectralBalancedCutClustering + cugraph.analyzeClustering_modularity
# cugraph.spectralBalancedCutClustering + cugraph.analyzeClustering_ratio_cut

# Subgraph Extraction

# @todo get this working
def test_subgraph_extraction():
    sources = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8]
    destinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]
    assert len(sources) == len(destinations)
    assert len(sources) == 45
    assert len(destinations) == 45
    nodes = set(sources+destinations)
    number_of_nodes = len(nodes)
    assert number_of_nodes == 10
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source="source", destination="destination")
    
    assert g.number_of_edges() == number_of_nodes * (number_of_nodes-1) # total number of edges x 2 for both directions
    assert g.number_of_edges() == 2 * len(sources)
    assert g.number_of_edges() == 2 * len(destinations)
    assert g.number_of_edges() == 90
    assert g.number_of_vertices() == number_of_nodes
    assert g.number_of_nodes() == number_of_nodes
    
    for number_of_subgraph_nodes in range(3,len(nodes)):
        subgraph_nodes = sorted(random.sample(nodes, number_of_subgraph_nodes)) # @todo why do these have to be sorted?
        subgraph_node_series = cudf.Series(subgraph_nodes)
        subgraph = cugraph.subgraph(g, subgraph_node_series)
        assert subgraph.number_of_edges() == number_of_subgraph_nodes * (number_of_subgraph_nodes-1)
        assert subgraph.number_of_vertices() == number_of_subgraph_nodes
        assert subgraph.number_of_nodes() == number_of_subgraph_nodes

# Triangle Count

def test_triangle_count_trivial():
    sources =      [0,1,2]
    destinations = [1,2,0]
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source="source", destination="destination")
    assert cugraph.triangles(g) // 3 == 1

def test_triangle_count_fully_connected_graph():
    sources =      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8]
    destinations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source="source", destination="destination")
    assert cugraph.triangles(g) // 3 == 120 # 10 choose 3

