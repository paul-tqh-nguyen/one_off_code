#!/usr/bin/python3

###########
# Imports #
###########

import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from functools import reduce
from typing import Union, Tuple, List

from misc_utilities import timer, histogram, temp_plt_figure, timeout, debug_on_error, trace, parallel_map
from preprocess import K_CORE_CHOICES_FOR_K
from preprocess import K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE

from preprocess import ACTORS_LABEL_PROP_CSV_TEMPLATE, DIRECTORS_LABEL_PROP_CSV_TEMPLATE
from preprocess import ACTORS_LOUVAIN_CSV_TEMPLATE, DIRECTORS_LOUVAIN_CSV_TEMPLATE

from preprocess import ACTORS_HITS_AUTHORITY_CSV_TEMPLATE, DIRECTORS_HITS_AUTHORITY_CSV_TEMPLATE
from preprocess import ACTORS_HITS_HUB_CSV_TEMPLATE, DIRECTORS_HITS_HUB_CSV_TEMPLATE
from preprocess import ACTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE, DIRECTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE
from preprocess import ACTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE, DIRECTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE
from preprocess import ACTORS_PAGERANK_CSV_TEMPLATE, DIRECTORS_PAGERANK_CSV_TEMPLATE
from preprocess import ACTORS_CLOSENESS_CSV_TEMPLATE, DIRECTORS_CLOSENESS_CSV_TEMPLATE
from preprocess import ACTORS_BETWEENNESS_CSV_TEMPLATE, DIRECTORS_BETWEENNESS_CSV_TEMPLATE
from preprocess import ACTORS_EIGENVECTOR_CSV_TEMPLATE, DIRECTORS_EIGENVECTOR_CSV_TEMPLATE
from preprocess import ACTORS_DEGREE_CSV_TEMPLATE, DIRECTORS_DEGREE_CSV_TEMPLATE
from preprocess import ACTORS_KATZ_CSV_TEMPLATE, DIRECTORS_KATZ_CSV_TEMPLATE

####################################
# Globals & Global Initializations #
####################################

matplotlib.use("Agg")

#############
# Utilities #
#############

def node_label_csv_file_into_label_to_nodes_map(node_label_csv_file: str) -> dict:
    node_label_df = pd.read_csv(node_label_csv_file)
    node_label_pairs = ((row.node, row.label) for row in node_label_df.itertuples())
    label_to_nodes_map = dict()
    for node, label in node_label_pairs:
        if label in label_to_nodes_map:
            label_to_nodes_map[label].add(node)
        else:
            label_to_nodes_map[label] = {node}
    return label_to_nodes_map

###########
# Drawing #
###########

def random_hex_color() -> str:
    return '#'+''.join(("%02X" % random.randint(0,255) for _ in range(3)))

EDGE_COLOR = "#cccccc"
DEFAULT_NODE_COLOR = 'darkblue'
NUMBER_OF_SPRING_LAYOUT_ITERATIONS = 10

DEFAULT_NODE_SIZE = 10
MINIMUM_NODE_SIZE = 1
NODE_SIZE_GROWTH_POTENTIAL = 300 - MINIMUM_NODE_SIZE

def draw_graph_to_file(output_location: str, graph: nx.Graph, **kwargs) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=NUMBER_OF_SPRING_LAYOUT_ITERATIONS)
        nx.draw_networkx_edges(graph,
                               layout, 
                               width=1,
                               alpha=0.1,
                               edge_color=EDGE_COLOR)
        assert len(kwargs) <= 1
        if 'label_to_nodes_map' in kwargs:
            label_to_nodes_map = kwargs['label_to_nodes_map']
            for label, nodes in label_to_nodes_map.items():
                node_color = random_hex_color()
                nx.draw_networkx_nodes(graph,
                                       layout,
                                       nodelist=nodes,
                                       node_color=node_color,
                                       node_size=DEFAULT_NODE_SIZE,
                                       ax=plot)
            plot.set_title(f'{len(label_to_nodes_map)} communities')
        elif 'node_to_value_map' in kwargs:
            node_to_value_map = kwargs['node_to_value_map']
            if len(node_to_value_map) > 0:
                max_value = max(node_to_value_map.values())
                for node, value in node_to_value_map.items():
                    normalized_value = value/max_value
                    nx.draw_networkx_nodes(graph,
                                           layout,
                                           nodelist=[node],
                                           node_color=DEFAULT_NODE_COLOR,
                                           node_size=MINIMUM_NODE_SIZE+NODE_SIZE_GROWTH_POTENTIAL*normalized_value,
                                           ax=plot)
        else:
            assert len(kwargs)==0
            nx.draw_networkx_nodes(graph,
                                   layout,
                                   nodelist=graph.nodes,
                                   node_color=DEFAULT_NODE_COLOR,
                                   node_size=DEFAULT_NODE_SIZE,
                                   ax=plot)
        figure.savefig(output_location)
    return

##########
# K-Core #
##########

def visualize_k_core_csv(csv_file: str) -> None:
    df = pd.read_csv(csv_file)
    graph = nx.from_pandas_edgelist(df, 'source', 'target')
    png_file = '.'.join(csv_file.split('.')[:-1])+'.png'
    assert csv_file[:-4] == png_file[:-4]
    draw_graph_to_file(png_file, graph)
    return 

###############
# Communities #
###############

def visualize_community_csv(edge_list_csv_label_csv_pair: Tuple[str,str]) -> None:
    edge_list_csv_file, node_label_csv_file = edge_list_csv_label_csv_pair
    edge_list_df = pd.read_csv(edge_list_csv_file)
    graph = nx.from_pandas_edgelist(edge_list_df, 'source', 'target')
    png_file = '.'.join(node_label_csv_file.split('.')[:-1])+'.png'
    assert node_label_csv_file[:-4] == png_file[:-4]
    label_to_nodes_map = node_label_csv_file_into_label_to_nodes_map(node_label_csv_file)
    draw_graph_to_file(png_file, graph, label_to_nodes_map=label_to_nodes_map)
    return 

def visualize_community_csvs(k_core_csv_files: List[str]) -> None:
    label_propagation_csv_files = reduce(list.__add__, ([ACTORS_LABEL_PROP_CSV_TEMPLATE % k, DIRECTORS_LABEL_PROP_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Label Propagation Data.')
    parallel_map(visualize_community_csv, zip(k_core_csv_files, label_propagation_csv_files))
    louvain_csv_files = reduce(list.__add__, ([ACTORS_LOUVAIN_CSV_TEMPLATE % k, DIRECTORS_LOUVAIN_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Louvain Data.')
    parallel_map(visualize_community_csv, zip(k_core_csv_files, louvain_csv_files))
    return

##################
# Vertex Ranking #
##################

def visualize_vertex_ranking_csv(edge_list_csv_value_csv_pair: Tuple[str,str]) -> None:
    edge_list_csv_file, node_value_csv_file = edge_list_csv_value_csv_pair
    edge_list_df = pd.read_csv(edge_list_csv_file)
    graph = nx.from_pandas_edgelist(edge_list_df, 'source', 'target')
    png_file = '.'.join(node_value_csv_file.split('.')[:-1])+'.png'
    node_value_csv_file
    node_value_df = pd.read_csv(node_value_csv_file)
    node_to_value_map = {row.node: row.value for row in node_value_df.itertuples()}
    draw_graph_to_file(png_file, graph, node_to_value_map=node_to_value_map)
    return 

def visualize_vertex_ranking_csvs(k_core_csv_files: List[str]) -> None:
    hits_authority_csv_files = reduce(list.__add__, ([ACTORS_HITS_AUTHORITY_CSV_TEMPLATE % k, DIRECTORS_HITS_AUTHORITY_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Hits Authority Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, hits_authority_csv_files))
    
    hits_hub_csv_files = reduce(list.__add__, ([ACTORS_HITS_HUB_CSV_TEMPLATE % k, DIRECTORS_HITS_HUB_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Hits Hub Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, hits_hub_csv_files))
    
    square_clustering_coefficient_csv_files = reduce(list.__add__, ([ACTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE % k, DIRECTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Square Clustering Coefficient Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, square_clustering_coefficient_csv_files))
    
    clustering_coefficient_csv_files = reduce(list.__add__, ([ACTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE % k, DIRECTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Clustering Coefficient Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, clustering_coefficient_csv_files))
    
    pagerank_csv_files = reduce(list.__add__, ([ACTORS_PAGERANK_CSV_TEMPLATE % k, DIRECTORS_PAGERANK_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Pagerank Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, pagerank_csv_files))
    
    closeness_csv_files = reduce(list.__add__, ([ACTORS_CLOSENESS_CSV_TEMPLATE % k, DIRECTORS_CLOSENESS_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Closeness Centrality Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, closeness_csv_files))
    
    betweenness_csv_files = reduce(list.__add__, ([ACTORS_BETWEENNESS_CSV_TEMPLATE % k, DIRECTORS_BETWEENNESS_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Betweenness Centrality Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, betweenness_csv_files))
    
    eigenvector_csv_files = reduce(list.__add__, ([ACTORS_EIGENVECTOR_CSV_TEMPLATE % k, DIRECTORS_EIGENVECTOR_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Eigenvector Centrality Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, eigenvector_csv_files))
    
    degree_csv_files = reduce(list.__add__, ([ACTORS_DEGREE_CSV_TEMPLATE % k, DIRECTORS_DEGREE_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Degree Centrality Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, degree_csv_files))
    
    katz_csv_files = reduce(list.__add__, ([ACTORS_KATZ_CSV_TEMPLATE % k, DIRECTORS_KATZ_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing Katz Centrality Data.')
    parallel_map(visualize_vertex_ranking_csv, zip(k_core_csv_files, katz_csv_files))
    return

########
# Main #
########

def visualize_k_core_graphs() -> None:
    k_core_csv_files = reduce(list.__add__, ([K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE % k, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    print('Visualizing K-Core Data.')
    parallel_map(visualize_k_core_csv, k_core_csv_files)
    visualize_community_csvs(k_core_csv_files)
    visualize_vertex_ranking_csvs(k_core_csv_files)
    return

@debug_on_error
def visualize() -> None:
    visualize_k_core_graphs()
    return

if __name__ == '__main__':
    visualize()
