#!/usr/bin/python3

###########
# Imports #
###########

import math
import pandas as pd
import networkx as nx
import multiprocessing as mp
from typing import List, Tuple

from misc_utilities import timer, debug_on_error, redirected_output

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
K_CORE_CHOICES_FOR_K = [10, 20, 30, 45, 60, 75, 100]

#############
# Utilities #
#############

def _mp_queue_is_empty(mp_queue: mp.Queue): -> bool:
    '''NB: Destructive; changes order.'''
    try: 
        queue_item = output_mp_queue.get(block = False)
        output_mp_queue.put(queue_item)
        is_empty = False
    except mp.Queue.Empty:
        is_empty = True
    return True

#################
# Load Raw Data #
#################

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def load_raw_data() -> Tuple[nx.Graph, pd.DataFrame]:
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
    return movies_graph, movies_df

##################
# Project Graphs #
##################

def project_actor_graph(movies_graph: nx.Graph, movies_df.DataFrame, output_queue: mp.Queue):
    output_string = None
    def note_output_string(new_output_string_value: str) -> None:
        output_string = new_output_string_value
        return
    with redirected_output(note_output_string):
        with timer(section_name="Actor graph projection"):
            full_projected_actors_graph = nx.projected_graph(movies_graph, movies_df['actor'])
            projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_actors_graph)
            projected_actors_edgelist = projected_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
            projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
            print(f"Number of Actors: {len(full_projected_actors_graph.nodes)}")
            assert set(projected_actors_edgelist.stack().unique()).issubset(set(full_projected_actors_graph.nodes))
            #assert set(projected_actors_edgelist.stack().unique()).issubset(set(movies_df.actor.unique()))
    output_queue.put((full_projected_actors_graph, output_string))
    return 

def project_director_graph(movies_graph: nx.Graph, movies_df.DataFrame, output_queue: mp.Queue) -> None:
    output_string = None
    def note_output_string(new_output_string_value: str) -> None:
        output_string = new_output_string_value
        return
    with redirected_output(note_output_string):
        with timer(section_name="Director graph projection"):
            full_projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
            projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_directors_graph)
            projected_directors_edgelist = projected_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
            projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
            print(f"Number of Directors: {len(full_projected_directors_graph.nodes)}")
            assert set(projected_directors_edgelist.stack().unique()).issubset(set(full_projected_directors_graph.nodes))
            #assert set(projected_directors_edgelist.stack().unique()).issubset(set(movies_df.director.unique()))
    output_queue.put((full_projected_directors_graph, output_string))
    return

def project_graphs(movies_graph: nx.Graph, movies_df.DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    actor_output_queue = mp.Queue()
    actor_process = mp.Process(target=project_actor_graph, args=(movies_graph, movies_df, output_queue))
    actor_process.start()

    director_output_queue = mp.Queue()
    director_process = mp.Process(target=project_director_graph, args=(movies_graph, movies_df, output_queue))
    director_process.start()
    
    actor_process.join()
    director_process.join()
    full_projected_actors_graph, actor_printouts = actor_output_queue.get()
    full_projected_directors_graph, director_printouts = director_output_queue.get()
    print(actor_printouts)
    print(director_printouts)
    assert _mp_queue_is_empty(actor_output_queue)
    assert _mp_queue_is_empty(director_output_queue)
    
    return full_projected_actors_graph, full_projected_directors_graph

##########################
# Generate K-Core Graphs #
##########################

def generate_k_core_graph(full_projected_graph: nx.Graph, k: int, graph_node_type: str) -> None:
    k_core_graph = nx.k_core(full_projected_graph, k)
    k_core_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_graph)
    k_core_edgelist = k_core_actors_edgelist.rename(columns={'source': f'source_{graph_node_type}', 'target': f'target_{graph_node_type}'})
    template = K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE if graph_node_type == 'actor' else K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE
    k_core_edgelist.to_csv(template%k, index=False)
    return

def generate_k_core_graphs(full_projected_actors_graph: nx.Graph, full_projected_directors_graph: nx.Graph) -> None:
    sorted_k_core_choices_for_k = sorted(K_CORE_CHOICES_FOR_K)
    with timer(section_name="K-core computation"):
        processes: List[mp.Process] = []
        for k in sorted_k_core_choices_for_k:
            actor_process = mp.Process(f=generate_k_core_graph, args=(full_projected_actors_graph, k, 'actor'))
            actor_process.start()
            processes.append(actor_process)
            
            director_process = mp.Process(f=generate_k_core_graph, args=(full_projected_directors_graph, k, 'director'))
            director_process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
    return

########
# Main #
########

def preprocess_data() -> None:
    movies_graph, movies_df = load_raw_data()
    full_projected_actors_graph, full_projected_directors_graph = project_graphs(movies_graph, movies_df)
    generate_k_core_graphs(full_projected_actors_graph, full_projected_directors_graph)
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

