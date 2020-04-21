#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from misc_utilities import timer, histogram, temp_plt_figure, timeout

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

@debug_on_error
def main() -> None:
    return

if __name__ == '__main__':
    main()
