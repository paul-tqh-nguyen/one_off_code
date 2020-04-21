#!/usr/bin/python3

###########
# Imports #
###########

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from functools import reduce

from misc_utilities import timer, histogram, temp_plt_figure, timeout, debug_on_error, parallel_map
from preprocess import K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE, K_CORE_CHOICES_FOR_K

####################################
# Globals & Global Initializations #
####################################

matplotlib.use("Agg")
NUMBER_OF_SPRING_LAYOUT_ITERATIONS = 40

###########
# Drawing #
###########

def draw_graph_to_file(graph: nx.Graph, output_location: str) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=NUMBER_OF_SPRING_LAYOUT_ITERATIONS)
        nx.draw_networkx_edges(graph,
                               layout, 
                               width=1,
                               alpha=0.1,
                               edge_color="#cccccc")
        nx.draw_networkx_nodes(graph,
                               layout,
                               nodelist=graph.nodes,
                               node_color='darkblue',
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
    draw_graph_to_file(graph, png_file)
    return 

def visualize_k_core_graphs() -> None:
    csv_files = reduce(list.__add__, ([K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE%k,K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE%k] for k in K_CORE_CHOICES_FOR_K))
    parallel_map(visualize_k_core_csv, csv_files)
    return

@debug_on_error
def visualize() -> None:
    visualize_k_core_graphs()
    return

if __name__ == '__main__':
    visualize()
