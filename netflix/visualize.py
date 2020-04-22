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
from typing import Union, Tuple

from misc_utilities import timer, histogram, temp_plt_figure, timeout, debug_on_error, parallel_map
from preprocess import K_CORE_CHOICES_FOR_K
from preprocess import K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE
from preprocess import ACTORS_LABEL_PROP_CSV_TEMPLATE, DIRECTORS_LABEL_PROP_CSV_TEMPLATE
from preprocess import ACTORS_LOUVAIN_CSV_TEMPLATE, DIRECTORS_LOUVAIN_CSV_TEMPLATE

####################################
# Globals & Global Initializations #
####################################

matplotlib.use("Agg")
NUMBER_OF_SPRING_LAYOUT_ITERATIONS = 40

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
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return '#%02X%02X%02X' % (r,g,b)

EDGE_COLOR = "#cccccc"
DEFAULT_NODE_COLOR = 'darkblue'

def draw_graph_to_file(output_location: str, graph: nx.Graph, label_to_nodes_map: dict) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=NUMBER_OF_SPRING_LAYOUT_ITERATIONS)
        nx.draw_networkx_edges(graph,
                               layout, 
                               width=1,
                               alpha=0.1,
                               edge_color=EDGE_COLOR)
        if label_to_nodes_map:
            for label, nodes in label_to_nodes_map.items():
                node_color = random_hex_color()
                nx.draw_networkx_nodes(graph,
                                       layout,
                                       nodelist=nodes,
                                       node_color=node_color,
                                       node_size=10,
                                       ax=plot)
        else:
            nx.draw_networkx_nodes(graph,
                                   layout,
                                   nodelist=graph.nodes,
                                   node_color=DEFAULT_NODE_COLOR,
                                   node_size=10,
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
    draw_graph_to_file(png_file, graph, None)
    return 

###############
# Communities #
###############

def visualize_community_csv(edge_list_csv_label_csv_pair: Tuple[str,str]) -> None:
    edge_list_csv_file, node_label_csv_file = edge_list_csv_label_csv_pair
    print(f"edge_list_csv_file {repr(edge_list_csv_file)}")
    print(f"node_label_csv_file {repr(node_label_csv_file)}")
    edge_list_df = pd.read_csv(edge_list_csv_file)
    graph = nx.from_pandas_edgelist(edge_list_df, 'source', 'target')
    png_file = '.'.join(node_label_csv_file.split('.')[:-1])+'.png'
    assert node_label_csv_file[:-4] == png_file[:-4]
    label_to_nodes_map = node_label_csv_file_into_label_to_nodes_map(node_label_csv_file)
    draw_graph_to_file(png_file, graph, label_to_nodes_map)
    return 

########
# Main #
########

def visualize_k_core_graphs() -> None:
    k_core_csv_files = reduce(list.__add__, ([K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE % k, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    parallel_map(visualize_k_core_csv, k_core_csv_files)
    label_propagation_csv_files = reduce(list.__add__, ([ACTORS_LABEL_PROP_CSV_TEMPLATE % k, DIRECTORS_LABEL_PROP_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    parallel_map(visualize_community_csv, zip(k_core_csv_files, label_propagation_csv_files))
    louvain_csv_files = reduce(list.__add__, ([ACTORS_LOUVAIN_CSV_TEMPLATE % k, DIRECTORS_LOUVAIN_CSV_TEMPLATE % k] for k in K_CORE_CHOICES_FOR_K))
    parallel_map(visualize_community_csv, zip(k_core_csv_files, louvain_csv_files))
    return

@debug_on_error
def visualize() -> None:
    visualize_k_core_graphs()
    return

if __name__ == '__main__':
    visualize()
