#!/usr/bin/python3

import math
import multiprocessing
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from typing import List

from misc_utilities import timer, histogram, temp_plt_figure, debug_on_error, timeout

matplotlib.use("Agg")

RAW_DATA_CSV = './netflix_titles.csv'

DIRECTOR_ACTOR_EDGE_LIST_CSV = './director_actor_edge_list.csv'
PROJECTED_ACTORS_CSV = './projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './projected_directors.csv'
K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE = './projected_actors_k_core_%d.csv'
K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE = './projected_directors_k_core_%d.csv'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

K_CORE_CHOICE_FOR_K = [10, 20, 30, 45, 60, 75, 100]

def draw_graph_to_file(graph: nx.Graph, output_location: str, number_of_spring_layout_iterations: int = 15) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=number_of_spring_layout_iterations)
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

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def reduce_graph_size_via_increasing_k_core(graph: nx.Graph, node_count_upperbound: int) -> nx.Graph:
    if len(graph)<node_count_upperbound:
        reduced_graph = graph
    else:
        degrees = dict(graph.degree).values()
        degree_histogram = histogram(degrees)
        sorted_degree_histogram = sorted(degree_histogram.items(), key=lambda x: -x[0])
        current_sum = 0
        for (degree, number_of_nodes_with_degree) in sorted_degree_histogram:
            current_sum += number_of_nodes_with_degree
            if current_sum >= node_count_upperbound:
                k_core_guess = degree/2
                break
        k_lower_bound = 0
        k_upper_bound = sorted_degree_histogram[0][0]
        bigger_graph = graph
        while True:
            reduced_graph = nx.k_core(bigger_graph, k_core_guess)
            if len(reduced_graph.nodes) < node_count_upperbound:
                k_upper_bound = k_core_guess
                new_k_core_guess = (k_core_guess + k_lower_bound) // 2
                if new_k_core_guess == k_lower_bound:
                    break
                k_core_guess = new_k_core_guess
            elif len(reduced_graph.nodes) > node_count_upperbound:
                bigger_graph = reduced_graph
                k_lower_bound = k_core_guess
                new_k_core_guess = (k_core_guess + k_upper_bound) // 2
                if k_core_guess == new_k_core_guess:
                    k_core_guess += 1
                    reduced_graph = nx.k_core(bigger_graph, k_core_guess)
                    break
                k_core_guess = new_k_core_guess
            else:
                break
    return reduced_graph

def _generate_png(input_tuple: tuple) -> None:
    with timeout(1800):
        (k, graph, graph_node_type) = input_tuple
        reduced_graph = nx.k_core(graph, k)
        number_of_spring_layout_iterations_choices = [5, 10, 15, 25, 40, 60, 90, 150, 200, 500]
        for number_of_spring_layout_iterations in number_of_spring_layout_iterations_choices:
            draw_graph_to_file(reduced_graph, f'./projected_{graph_node_type}_size_{len(graph.nodes)}_k_core_{k}_layout_iterations_{number_of_spring_layout_iterations}.png')
    return

def preprocess_data() -> None:
    with timer(section_name="Initial raw data loading"):
        all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
        movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type']).dropna()
        #all_df = all_df[all_df.country.str.find(r'United States')>=0]
        for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
            movies_df = expand_dataframe_list_values_for_column(movies_df, column)
        movies_df = movies_df.rename(columns={'cast': 'actor'})
        movies_df = movies_df[movies_df.director != movies_df.actor]
        print(f'Original Number of Directors: {len(movies_df.director.unique())}')
        print(f'Original Number of Actors: {len(movies_df.actor.unique())}')
        movies_df.to_csv(DIRECTOR_ACTOR_EDGE_LIST_CSV, index=False)
        movies_graph = nx.from_pandas_edgelist(movies_df, 'actor', 'director')
    with timer(section_name="Actor graph projection"):
        full_projected_actors_graph = nx.projected_graph(movies_graph, movies_df['actor'])
        projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_actors_graph)
        projected_actors_edgelist = projected_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
        projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
        print(f"Number of Actors: {len(full_projected_actors_graph.nodes)}")
        assert set(projected_actors_edgelist.stack().unique()).issubset(set(full_projected_actors_graph.nodes))
        #assert set(projected_actors_edgelist.stack().unique()).issubset(set(movies_df.actor.unique()))
    with timer(section_name="Director graph projection"):
        full_projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
        projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_directors_graph)
        projected_directors_edgelist = projected_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
        projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
        print(f"Number of Directors: {len(full_projected_directors_graph.nodes)}")
        assert set(projected_directors_edgelist.stack().unique()).issubset(set(full_projected_directors_graph.nodes))
        #assert set(projected_directors_edgelist.stack().unique()).issubset(set(movies_df.director.unique()))
    with timer(section_name="Director graph K-core computation"):
        for k in K_CORE_CHOICE_FOR_K:
            k_core_directors_graph = nx.k_core(full_projected_directors_graph, k)
            k_core_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_directors_graph)
            k_core_directors_edgelist = k_core_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
            k_core_directors_edgelist.to_csv(K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE%k, index=False)
    with timer(section_name="Actor/Director graph projection visualization"):
        p = multiprocessing.Pool()
        k_choices: List[int] = []
        graphs: List[nx.Graphs] = []
        graph_node_types: List[str] = []
        for k_core_value in K_CORE_CHOICE_FOR_K:
            k_choices.append(k_core_value)
            graphs.append(full_projected_actors_graph)
            graph_node_types.append('actor')
            k_choices.append(k_core_value)
            graphs.append(full_projected_directors_graph)
            graph_node_types.append('director')
        p.map(_generate_png, zip(k_choices, graphs, graph_node_types))
        p.close()
        p.join()
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

