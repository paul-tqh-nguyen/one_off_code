
'''

Sections:
* Imports
* Globals

'''

###########
# Imports #
###########

import os
import multiprocessing as mp
import networkx as nx
from functools import reduce
from karateclub import Graph2Vec
from typing import Iterable, List

from misc_utilities import *

###########
# Globals #
###########

EDGES_FILE = './data/MUTAG_A.txt'
EDGE_LABELS_FILE = './data/MUTAG_edge_labels.txt'
NODE_LABELS_FILE = './data/MUTAG_node_labels.txt'
GRAPH_IDS_FILE = './data/MUTAG_graph_indicator.txt'
GRAPH_LABELS_FILE = './data/MUTAG_graph_labels.txt'

CHECKPOINT_DIR = './checkpoints'

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

###################
# Data Processing #
###################

def process_data() -> List[nx.Graph]:
    nx_graphs = nx.Graph()
    # with open(EDGES_FILE, 'r') as edges_file_handle:
    #     edges_file_lines = edges_file_handle.readlines()
    #     split_lines = eager_map(lambda s: s.split(','), edges_file_lines)
    #     assert set(map(len, split_lines)) == {2}
    #     edges = map(lambda l: (int(l[0]), int(l[1])), split_lines)
    #     del split_lines
    # with open(EDGE_LABELS_FILE, 'r') as edge_labels_file_handle:
    #     edge_labels = map(int, edge_labels_file_handle.readlines())
    #     edges = (src, dst, label for (src, dst), label in zip(edges, edge_labels))
    # nx_graph.add_weighted_edges_from(edges, weight='edge_label')
    # assert functools.reduce(lambda l, s: s.union(l), edges, set()) == set(nx_graph.nodes)
    # with open(NODE_LABELS_FILE, 'r') as node_labels_file_handle:
    #     node_id_to_node_label = dict(enumerate(map(int, node_labels_file_handle.readlines())))
    #     assert len(node_id_to_node_label) == len(nx_graph.nodes)
    #     nx.set_node_attributes(nx_graph, node_id_to_node_label, 'node_label')
    # with open(GRAPH_IDS_FILE, 'r') as graph_ids_file_handle:
    #     node_id_to_graph_id = dict(enumerate(map(int, graph_ids_file_handle.readlines())))
    #     assert len(node_id_to_graph_id) == len(nx_graph.nodes)
    #     nx.set_node_attributes(nx_graph, node_id_to_graph_id, 'graph_id')
    with open(GRAPH_LABELS_FILE, 'r') as graph_labels_file_handle:
        graph_id_to_graph_label = dict(enumerate(map(int, graph_labels_file_handle.readlines())))
        assert len(graph_id_to_graph_label) == 188
        node_id_to_graph_label = ((node_id, graph_id_to_graph_label[graph_id]) for node_id, graph_id in node_id_to_graph_id.items())
        assert len(node_id_to_graph_label) == len(nx_graph.nodes)
        nx.set_node_attributes(nx_graph, node_id_to_graph_label, 'graph_label')
    
    # @todo make this return a list of graphs
    return nx_graphs

#######################
# graph2vec Utilities #
#######################


class HyperParameterSearchObjective:
    def __init__(self, graphs: List[nx.Graph], process_id_queue: object):
        # process_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dyanmically
        self.process_id_queue = process_id_queue
        self.graphs = graphs
        
    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters: {,
            'wl_iterations': int(trial.suggest_int('wl_iterations', 1, 6)),
            'dimensions': int(trial.suggest_int('dimensions', 100, 500)),
            'workers': 1,
            'epochs': int(trial.suggest_int('epochs', 10, 15)),
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 1e-1),
            'min_count': 0,
            'seed': 1234,
        }
        return hyperparameters

    def train_model(**hyperparameters) -> float:
        grgaph2vec_trainer = Graph2Vec(**hyperparameters)
        grgaph2vec_trainer.fit(self.graphs)
        graph_embeddings = grgaph2vec_trainer.get_embedding()
        # @todo save embeddings
        # @todo finish this
        return best_validation_loss

    def checkpoint_directory_from_hyperparameters() -> str:
        checkpoint_dir = os.path.join(
            CHECKPOINT_DIR,
            f'wl_iterations_{int(wl_iterations)}_' \
            f'dimensions_{int(dimensions)}_' \
            f'epochs_{int(epochs)}_' \
            f'learning_rate_{learning_rate:.5g}'
        )
        return checkpoint_directory
    
    def __call__(self, trial: optuna.Trial) -> float:
        process_id = self.process_id_queue.get()
        
        hyperparameters = self.get_trial_hyperparameters(trial)
        
        checkpoint_dir = checkpoint_directory_from_hyperparameters(**hyperparameters)
        print(f'Starting training for {checkpoint_dir} on GPU {process_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = self.train_model(**hyperparameters)
        except Exception as exception:
            self.process_id_queue.put(process_id)
            raise exception
        self.process_id_queue.put(process_id)
        return best_validation_loss

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    nx_graph = process_data()
    # @todo set up hyperparameter search
    return

if __name__ == '__main__':
    main()
