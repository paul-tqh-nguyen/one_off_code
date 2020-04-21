#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from misc_utilities import timer, histogram, temp_plt_figure, timeout
from preprocess import K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE, K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE, K_CORE_CHOICES_FOR_K

matplotlib.use("Agg")
NUMBER_OF_SPRING_LAYOUT_ITERATIONS = 100

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

def visualize_k_core_graphs() -> None:
    for k in K_CORE_CHOICES_FOR_K:
        actor_csv = K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE % k
        actor_df = pd.read_csv(actor_csv)
        actor_graph = nx.from_pandas_edgelist(actor_df, 'source', 'target')
        actor_png = '.'.join(actor_csv.split('.')[:-1])+'.png'
        draw_graph_to_file(actor_graph, actor_png)
        
        director_csv = K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE % k
        director_df = pd.read_csv(director_csv)
        director_graph = nx.from_pandas_edgelist(director_df, 'source', 'target')
        director_png = '.'.join(director_csv.split('.')[:-1])+'.png'
        draw_graph_to_file(director_graph, director_png)
    return

@debug_on_error
def visualize() -> None:
    visualize_k_core_graphs()
    return

if __name__ == '__main__':
    visualize()
