#!/usr/bin/python3

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from misc_utilities import timer, histogram, temp_plt_figure, debug_on_error

matplotlib.use("Agg")

RAW_DATA_CSV = './netflix_titles.csv'

DIRECTOR_ACTOR_EDGE_LIST_CSV = './director_actor_edge_list.csv'
PROJECTED_ACTORS_CSV = './projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './projected_directors.csv'
PROJECTED_ACTORS_PNG = './projected_actors.png'
PROJECTED_DIRECTORS_PNG = './projected_directors.png'

MAXIMUM_NUMBER_OF_ACTORS = 500 # 5000
MAXIMUM_NUMBER_OF_DIRECTORS = 500 # 5000

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

def draw_graph_to_file(graph: nx.Graph, output_location: str) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=15)
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

def preprocess_data() -> None:
    with timer(section_name="Initial raw data loading"):
        all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
        movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type'])
        movies_df = movies_df.dropna()
        for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
            movies_df = expand_dataframe_list_values_for_column(movies_df, column)
        print(f'Original Number of Directors: {len(movies_df.director.unique())}')
        print(f'Original Number of Actors: {len(movies_df.cast.unique())}')
        movies_df = movies_df[movies_df.director != movies_df.cast]
        movies_df.to_csv(DIRECTOR_ACTOR_EDGE_LIST_CSV, index=False)
        movies_graph = nx.from_pandas_edgelist(movies_df, 'cast', 'director', True)
    with timer(section_name="Actor graph projection"):
        projected_actors_graph = nx.projected_graph(movies_graph, movies_df['cast'])
        projected_actors_graph = reduce_graph_size_via_increasing_k_core(projected_actors_graph, MAXIMUM_NUMBER_OF_ACTORS)
        projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(projected_actors_graph)
        projected_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
        projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
        print(f"Number of Actors: {len(projected_actors_graph.nodes)}")
    with timer(section_name="Actor graph projection visualization"):
        draw_graph_to_file(projected_actors_graph, PROJECTED_ACTORS_PNG)
    with timer(section_name="Director graph projection"):
        projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
        projected_directors_graph = reduce_graph_size_via_increasing_k_core(projected_directors_graph, MAXIMUM_NUMBER_OF_DIRECTORS)
        projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(projected_directors_graph)
        projected_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
        projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
        print(f"Number of Directors: {len(projected_directors_graph.nodes)}")
    with timer(section_name="Director graph projection visualization"):
        draw_graph_to_file(projected_directors_graph, PROJECTED_DIRECTORS_PNG)
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

