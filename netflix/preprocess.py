#!/usr/bin/python3

###########
# Imports #
###########

import math
import pandas as pd
import networkx as nx
import multiprocessing as mp
import community as community_louvain
from typing import List, Tuple, Union

from misc_utilities import timer, debug_on_error, redirected_output, tqdm_with_message, trace

def tqdm_with_message(*args, **kwargs): # @todo remove this
    return args[0]

###########
# Globals #
###########

RAW_DATA_CSV = './netflix_titles.csv'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

DIRECTOR_ACTOR_EDGE_LIST_CSV = './director_actor_edge_list.csv'
PROJECTED_ACTORS_CSV = './projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './projected_directors.csv'

K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE = './projected_actors_k_core_%d.csv'
K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE = './projected_directors_k_core_%d.csv'
#K_CORE_CHOICES_FOR_K = [10, 20, 30, 45, 60, 75, 100]
K_CORE_CHOICES_FOR_K = [45]

ACTORS_LABEL_PROP_CSV_TEMPLATE = './projected_actors_k_core_%d_label_propagation.csv'
DIRECTORS_LABEL_PROP_CSV_TEMPLATE = './projected_directors_k_core_%d_label_propagation.csv'

ACTORS_LOUVAIN_CSV_TEMPLATE = './projected_actors_k_core_%d_louvain.csv'
DIRECTORS_LOUVAIN_CSV_TEMPLATE = './projected_directors_k_core_%d_louvain.csv'

#################
# Load Raw Data #
#################

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: Union[str, int]) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def load_raw_data() -> Tuple[nx.Graph, pd.DataFrame]:
    with timer(section_name='Initial raw data loading'):
        all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
        movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type']).dropna()
        for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
            movies_df = expand_dataframe_list_values_for_column(movies_df, column)
        movies_df = movies_df.rename(columns={'cast': 'actor'})
        movies_df = movies_df[~movies_df.actor.isin(movies_df.director)]
        print(f'Original Number of Directors: {len(movies_df.director.unique())}')
        print(f'Original Number of Actors: {len(movies_df.actor.unique())}')
        movies_df.to_csv(DIRECTOR_ACTOR_EDGE_LIST_CSV, index=False)
        movies_graph = nx.from_pandas_edgelist(movies_df, 'actor', 'director')
        assert nx.is_bipartite(movies_graph)
        assert len(set(movies_df.actor) & set (movies_df.director)) == 0
    return movies_graph, movies_df

##################
# Project Graphs #
##################

def project_actor_graph_serial(movies_graph: nx.Graph, movies_df: pd.DataFrame) -> nx.Graph:
    full_projected_actors_graph = nx.projected_graph(movies_graph, movies_df['actor'])
    projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_actors_graph)
    projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
    print(f'Number of Actors: {len(full_projected_actors_graph.nodes)}')
    assert set(projected_actors_edgelist.stack().unique()).issubset(set(full_projected_actors_graph.nodes))
    assert set(projected_actors_edgelist.stack().unique()).issubset(set(movies_df.actor.unique()))
    return full_projected_actors_graph

def project_director_graph_serial(movies_graph: nx.Graph, movies_df: pd.DataFrame) -> nx.Graph:
    full_projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
    projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_directors_graph)
    projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
    print(f'Number of Directors: {len(full_projected_directors_graph.nodes)}')
    assert set(projected_directors_edgelist.stack().unique()).issubset(set(full_projected_directors_graph.nodes))
    assert set(projected_directors_edgelist.stack().unique()).issubset(set(movies_df.director.unique()))
    return full_projected_directors_graph

def project_actor_graph(movies_graph: nx.Graph, movies_df: pd.DataFrame, output_queue: mp.SimpleQueue) -> None:
    output_dict = dict()
    def note_output_string(new_output_string_value: str) -> None:
        output_dict['output_string'] = new_output_string_value
        return
    with redirected_output(note_output_string):
        with timer(section_name='Actor graph projection'):
            full_projected_actors_graph = project_actor_graph_serial(movies_graph, movies_df)
    output_queue.put((full_projected_actors_graph, output_dict['output_string']))
    assert len(output_dict)==1
    return

def project_director_graph(movies_graph: nx.Graph, movies_df: pd.DataFrame, output_queue: mp.SimpleQueue) -> None:
    output_dict = dict()
    def note_output_string(new_output_string_value: str) -> None:
        output_dict['output_string'] = new_output_string_value
        return
    with redirected_output(note_output_string):
        with timer(section_name='Director graph projection'):
            full_projected_directors_graph = project_director_graph_serial(movies_graph, movies_df)
    output_queue.put((full_projected_directors_graph, output_dict['output_string']))
    assert len(output_dict)==1
    return

def project_graphs(movies_graph: nx.Graph, movies_df: pd.DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    print()

    actor_output_queue = mp.SimpleQueue()
    actor_process = mp.Process(target=project_actor_graph, args=(movies_graph, movies_df, actor_output_queue))
    actor_process.start()
    print('Started projection of actors.')
    
    director_output_queue = mp.SimpleQueue()
    director_process = mp.Process(target=project_director_graph, args=(movies_graph, movies_df, director_output_queue))
    director_process.start()
    print('Started projection of directors.')
    
    full_projected_actors_graph, actor_printouts = actor_output_queue.get()
    full_projected_directors_graph, director_printouts = director_output_queue.get()
    actor_process.join()
    actor_process.close()
    director_process.join()
    director_process.close()

    print()
    print(actor_printouts)
    print(director_printouts)
    assert actor_output_queue.empty()
    assert director_output_queue.empty()
    
    return full_projected_actors_graph, full_projected_directors_graph

##########################
# Generate K-Core Graphs #
##########################

@trace
def generate_k_core_graph(full_projected_graph: nx.Graph, k: int, graph_node_type: str, output_queue: mp.SimpleQueue) -> None:
    k_core_graph = nx.k_core(full_projected_graph, k)
    k_core_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_graph)
    template = K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE if graph_node_type == 'actor' else K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE
    k_core_edgelist.to_csv(template%k, index=False)
    import time, random; time.sleep(random.randint(1,3))
    print(f"graph_node_type {repr(graph_node_type)}")
    print(f"k_core_graph {repr(k_core_graph)}")
    print(f"len(k_core_graph) {repr(len(k_core_graph))}")
    print(f"len(k_core_edgelist) {repr(len(k_core_edgelist))}")
    print(f"template%k {repr(template%k)}")
    output_queue.put((k_core_graph, graph_node_type, k))
    return

@trace
def generate_k_core_graphs(full_projected_actors_graph: nx.Graph, full_projected_directors_graph: nx.Graph) -> Tuple[dict,dict]:
    k_to_actor_k_core_graph_map = dict()
    k_to_director_k_core_graph_map = dict()
    with timer(section_name='K-core computation'):
        processes: List[mp.Process] = []
        output_queue = mp.SimpleQueue()
        sorted_k_core_choices_for_k = sorted(K_CORE_CHOICES_FOR_K)
        for k in sorted_k_core_choices_for_k:
            actor_process = mp.Process(target=generate_k_core_graph, args=(full_projected_actors_graph, k, 'actor', output_queue))
            actor_process.start()
            processes.append(actor_process)
            
            director_process = mp.Process(target=generate_k_core_graph, args=(full_projected_directors_graph, k, 'director', output_queue))
            director_process.start()
            processes.append(director_process)
        
        for _ in tqdm_with_message(range(len(processes)), post_yield_message_func = lambda index: f'Gathering Results from K-Core Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
            k_core_graph, graph_node_type, k = output_queue.get()
            print('===============================================')
            print('generate_k_core_graphs')
            print(f"graph_node_type {repr(graph_node_type)}")
            print(f"k {k}")
            print(f"k_core_graph {repr(k_core_graph)}")
            print(f"len(k_core_graph) {repr(len(k_core_graph))}")
            print('===============================================')
            if graph_node_type == 'actor':
                k_to_actor_k_core_graph_map[k] = k_core_graph
            else:
                assert graph_node_type == 'director'
                k_to_director_k_core_graph_map[k] = k_core_graph
        for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join K-Core Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
            process.join()
    return k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map

#####################
# Generate Clusters #
#####################

def write_node_to_label_map_to_csv(node_to_label_map: dict, csv_file: str) -> None:
    if len(node_to_label_map) == 0:
        with open(csv_file, 'w') as f:
            f.write('node,label')
    else:
        label_df = pd.DataFrame.from_dict(node_to_label_map, orient='index')
        label_df.to_csv(csv_file, index_label='node', header=['label'])
    return

# Label Propagation

@trace
def generate_label_propagation_csv(graph: nx.Graph, csv_file: str) -> None:
    print(f"csv_file {repr(csv_file)}")
    print(f"len(graph.nodes) {len(graph.nodes)}")
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(graph)
    node_to_label_map = dict()
    for label, nodes in enumerate(communities):
        for node in nodes:
            assert node not in node_to_label_map
            node_to_label_map[node] = label
    print(f"node_to_label_map {node_to_label_map}")
    write_node_to_label_map_to_csv(node_to_label_map, csv_file)
    return

@trace
def generate_label_propagation_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_LABEL_PROP_CSV_TEMPLATE%k
        print(f"csv_file {repr(csv_file)}")
        print(f"k {k}")
        print(f"graph {repr(graph)}")
        print(f"len(graph.nodes) {len(graph.nodes)}")
        process = mp.Process(target=generate_label_propagation_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_LABEL_PROP_CSV_TEMPLATE%k
        print(f"csv_file {repr(csv_file)}")
        print(f"k {k}")
        print(f"graph {repr(graph)}")
        print(f"len(graph.nodes) {len(graph.nodes)}")
        process = mp.Process(target=generate_label_propagation_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Louvain

@trace
def generate_louvain_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_label_map = community_louvain.best_partition(graph)
    print(f"node_to_label_map {node_to_label_map}")
    write_node_to_label_map_to_csv(node_to_label_map, csv_file)
    return

def generate_louvain_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = ACTORS_LOUVAIN_CSV_TEMPLATE%k
        process = mp.Process(target=generate_louvain_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = DIRECTORS_LOUVAIN_CSV_TEMPLATE%k
        process = mp.Process(target=generate_louvain_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Top-Level

@trace
def generate_clusters_for_k_core_graphs(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> None:
    processes: List[mp.Process] = []
    processes = processes + generate_label_propagation_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_louvain_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join Cluster Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
        process.join()
    return

########
# Main #
########

def preprocess_data() -> None:
    movies_graph, movies_df = load_raw_data()
    full_projected_actors_graph, full_projected_directors_graph = project_graphs(movies_graph, movies_df)
    k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map = generate_k_core_graphs(full_projected_actors_graph, full_projected_directors_graph)
    generate_clusters_for_k_core_graphs(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    print()
    print('Done.')
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

