#!/usr/bin/python3

###########
# Imports #
###########

import os
import math
import pandas as pd
import networkx as nx
import multiprocessing as mp
import community as community_louvain
from typing import List, Tuple, Union

from misc_utilities import timer, debug_on_error, redirected_output, tqdm_with_message, trace

###########
# Globals #
###########

RAW_DATA_CSV = './netflix_titles.csv'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

if not os.path.isdir('./output/'):
    os.makedirs('./output/')
DIRECTOR_ACTOR_EDGE_LIST_CSV = './output/director_actor_edge_list.csv'
PROJECTED_ACTORS_CSV = './output/projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './output/projected_directors.csv'

K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE = './output/projected_actors_k_core_%d.csv'
K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE = './output/projected_directors_k_core_%d.csv'
#K_CORE_CHOICES_FOR_K = [10, 20, 30, 45, 60, 75, 100]
K_CORE_CHOICES_FOR_K = [45]

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

def generate_k_core_graph(full_projected_graph: nx.Graph, k: int, graph_node_type: str, output_queue: mp.SimpleQueue) -> None:
    k_core_graph = nx.k_core(full_projected_graph, k)
    k_core_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_graph)
    template = K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE if graph_node_type == 'actor' else K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE
    k_core_edgelist.to_csv(template%k, index=False)
    output_queue.put((k_core_graph, graph_node_type, k))
    return

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
            if graph_node_type == 'actor':
                k_to_actor_k_core_graph_map[k] = k_core_graph
            else:
                assert graph_node_type == 'director'
                k_to_director_k_core_graph_map[k] = k_core_graph
        for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join K-Core Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
            process.join()
    return k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map

########################
# Generate Communities #
########################

def write_node_to_label_map_to_csv(node_to_label_map: dict, csv_file: str) -> None:
    if len(node_to_label_map) == 0:
        with open(csv_file, 'w') as f:
            f.write('node,label')
    else:
        label_df = pd.DataFrame.from_dict(node_to_label_map, orient='index')
        label_df.to_csv(csv_file, index_label='node', header=['label'])
    return

# Label Propagation

ACTORS_LABEL_PROP_CSV_TEMPLATE = './output/projected_actors_k_core_%d_label_propagation.csv'
DIRECTORS_LABEL_PROP_CSV_TEMPLATE = './output/projected_directors_k_core_%d_label_propagation.csv'

def generate_label_propagation_csv(graph: nx.Graph, csv_file: str) -> None:
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(graph)
    node_to_label_map = dict()
    for label, nodes in enumerate(communities):
        for node in nodes:
            assert node not in node_to_label_map
            node_to_label_map[node] = label
    write_node_to_label_map_to_csv(node_to_label_map, csv_file)
    return

def generate_label_propagation_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_LABEL_PROP_CSV_TEMPLATE%k
        process = mp.Process(target=generate_label_propagation_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_LABEL_PROP_CSV_TEMPLATE%k
        process = mp.Process(target=generate_label_propagation_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Louvain

ACTORS_LOUVAIN_CSV_TEMPLATE = './output/projected_actors_k_core_%d_louvain.csv'
DIRECTORS_LOUVAIN_CSV_TEMPLATE = './output/projected_directors_k_core_%d_louvain.csv'

def generate_louvain_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_label_map = community_louvain.best_partition(graph)
    write_node_to_label_map_to_csv(node_to_label_map, csv_file)
    return

def generate_louvain_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_LOUVAIN_CSV_TEMPLATE%k
        process = mp.Process(target=generate_louvain_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_LOUVAIN_CSV_TEMPLATE%k
        process = mp.Process(target=generate_louvain_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Top-Level

def generate_communities_for_k_core_graphs(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> None:
    processes: List[mp.Process] = []
    processes = processes + generate_label_propagation_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_louvain_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join Community Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
        process.join()
    return

############################
# Generate Vertex Rankings #
############################

def write_node_to_value_map_to_csv(node_to_value_map: dict, csv_file: str) -> None:
    if len(node_to_value_map) == 0:
        with open(csv_file, 'w') as f:
            f.write('node,value')
    else:
        value_df = pd.DataFrame.from_dict(node_to_value_map, orient='index')
        assert len(value_df.columns) == 1
        value_df.to_csv(csv_file, index_label='node', header=['value'])
    return

# HITS

ACTORS_HITS_AUTHORITY_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_hits_authority.csv'
DIRECTORS_HITS_AUTHORITY_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_hits_authority.csv'

ACTORS_HITS_HUB_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_hits_hub.csv'
DIRECTORS_HITS_HUB_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_hits_hub.csv'

def generate_hits_csv(graph: nx.Graph, hub_csv_file: str, authority_csv_file: str) -> None:
    node_to_hub_value_map, node_to_authority_value_map = nx.hits(graph, normalized=True)
    write_node_to_value_map_to_csv(node_to_hub_value_map, hub_csv_file)
    write_node_to_value_map_to_csv(node_to_authority_value_map, authority_csv_file)
    return

def generate_hits_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        hub_csv_file = ACTORS_HITS_HUB_CSV_TEMPLATE%k
        authority_csv_file = ACTORS_HITS_AUTHORITY_CSV_TEMPLATE%k
        process = mp.Process(target=generate_hits_csv, args=(graph,hub_csv_file,authority_csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        hub_csv_file = DIRECTORS_HITS_HUB_CSV_TEMPLATE%k
        authority_csv_file = DIRECTORS_HITS_AUTHORITY_CSV_TEMPLATE%k
        process = mp.Process(target=generate_hits_csv, args=(graph,hub_csv_file,authority_csv_file))
        process.start()
        processes.append(process)
    return processes

# Square Clustering Coefficient

ACTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_square_clustering_coefficient.csv'
DIRECTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_square_clustering_coefficient.csv'

def generate_square_clustering_coefficient_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.square_clustering(graph)
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_square_clustering_coefficient_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE%k
        process = mp.Process(target=generate_square_clustering_coefficient_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_SQUARE_CLUSTERING_COEFFICIENT_CSV_TEMPLATE%k
        process = mp.Process(target=generate_square_clustering_coefficient_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Clustering Coefficient

ACTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_clustering_coefficient.csv'
DIRECTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_clustering_coefficient.csv'

def generate_clustering_coefficient_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.clustering(graph)
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_clustering_coefficient_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE%k
        process = mp.Process(target=generate_clustering_coefficient_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_CLUSTERING_COEFFICIENT_CSV_TEMPLATE%k
        process = mp.Process(target=generate_clustering_coefficient_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# PageRank

ACTORS_PAGERANK_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_pagerank.csv'
DIRECTORS_PAGERANK_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_pagerank.csv'

def generate_pagerank_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.pagerank(graph)
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_pagerank_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_PAGERANK_CSV_TEMPLATE%k
        process = mp.Process(target=generate_pagerank_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_PAGERANK_CSV_TEMPLATE%k
        process = mp.Process(target=generate_pagerank_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Closeness

ACTORS_CLOSENESS_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_closeness.csv'
DIRECTORS_CLOSENESS_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_closeness.csv'

def generate_closeness_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.closeness_centrality(graph) if len(graph.nodes) > 0 else dict()
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_closeness_centrality_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_CLOSENESS_CSV_TEMPLATE%k
        process = mp.Process(target=generate_closeness_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_CLOSENESS_CSV_TEMPLATE%k
        process = mp.Process(target=generate_closeness_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Betweenness

ACTORS_BETWEENNESS_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_betweenness.csv'
DIRECTORS_BETWEENNESS_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_betweenness.csv'

def generate_betweenness_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.betweenness_centrality(graph) if len(graph.nodes) > 0 else dict()
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_betweenness_centrality_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_BETWEENNESS_CSV_TEMPLATE%k
        process = mp.Process(target=generate_betweenness_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_BETWEENNESS_CSV_TEMPLATE%k
        process = mp.Process(target=generate_betweenness_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Eigenvector

ACTORS_EIGENVECTOR_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_eigenvector.csv'
DIRECTORS_EIGENVECTOR_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_eigenvector.csv'

def generate_eigenvector_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.eigenvector_centrality(graph) if len(graph.nodes) > 0 else dict() if len(graph.nodes) > 0 else dict()
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_eigenvector_centrality_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_EIGENVECTOR_CSV_TEMPLATE%k
        process = mp.Process(target=generate_eigenvector_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_EIGENVECTOR_CSV_TEMPLATE%k
        process = mp.Process(target=generate_eigenvector_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Degree

ACTORS_DEGREE_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_degree.csv'
DIRECTORS_DEGREE_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_degree.csv'

def generate_degree_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.degree_centrality(graph) if len(graph.nodes) > 0 else dict()
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_degree_centrality_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_DEGREE_CSV_TEMPLATE%k
        process = mp.Process(target=generate_degree_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_DEGREE_CSV_TEMPLATE%k
        process = mp.Process(target=generate_degree_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Katz

KATZ_ALHPA = 0.01
ACTORS_KATZ_CSV_TEMPLATE  = './output/projected_actors_k_core_%d_katz.csv'
DIRECTORS_KATZ_CSV_TEMPLATE  = './output/projected_directors_k_core_%d_katz.csv'

def generate_katz_csv(graph: nx.Graph, csv_file: str) -> None:
    node_to_value_map = nx.katz_centrality(graph, KATZ_ALHPA)
    write_node_to_value_map_to_csv(node_to_value_map, csv_file)
    return

def generate_katz_centrality_processes(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> List[mp.Process]:
    processes: List[mp.Process] = []
    for k, graph in k_to_actor_k_core_graph_map.items():
        csv_file = ACTORS_KATZ_CSV_TEMPLATE%k
        process = mp.Process(target=generate_katz_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    for k, graph in k_to_director_k_core_graph_map.items():
        csv_file = DIRECTORS_KATZ_CSV_TEMPLATE%k
        process = mp.Process(target=generate_katz_csv, args=(graph,csv_file))
        process.start()
        processes.append(process)
    return processes

# Top-Level

def generate_vertex_rankings_for_k_core_graphs(k_to_actor_k_core_graph_map: dict, k_to_director_k_core_graph_map: dict) -> None:
    processes: List[mp.Process] = []
    processes = processes + generate_katz_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_hits_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_square_clustering_coefficient_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_clustering_coefficient_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_pagerank_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_closeness_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_betweenness_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_eigenvector_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_degree_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    processes = processes + generate_katz_centrality_processes(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join Vertex Ranking Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
        process.join()
    return

########
# Main #
########

def preprocess_data() -> None:
    movies_graph, movies_df = load_raw_data()
    full_projected_actors_graph, full_projected_directors_graph = project_graphs(movies_graph, movies_df)
    k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map = generate_k_core_graphs(full_projected_actors_graph, full_projected_directors_graph)
    generate_communities_for_k_core_graphs(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    generate_vertex_rankings_for_k_core_graphs(k_to_actor_k_core_graph_map, k_to_director_k_core_graph_map)
    print()
    print('Done.')
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

