
'''

Sections:
* Imports
* Globals

'''

# @todo update docstring

###########
# Imports #
###########

import os
import json
import more_itertools
import random
import joblib
import pickle
import optuna
import gensim
import karateclub
import numpy as np
import multiprocessing as mp
import networkx as nx
from collections import OrderedDict
from functools import reduce
from typing import Iterable, Dict, Tuple

from misc_utilities import *
from global_values import *
from mutag_classifier import MUTAGClassifier

# @todo make sure these imports are used

###########
# Globals #
###########

# Data Files

EDGES_FILE = './data/MUTAG_A.txt'
EDGE_LABELS_FILE = './data/MUTAG_edge_labels.txt'
NODE_LABELS_FILE = './data/MUTAG_node_labels.txt'
GRAPH_IDS_FILE = './data/MUTAG_graph_indicator.txt'
GRAPH_LABELS_FILE = './data/MUTAG_graph_labels.txt'

# @todo make sure all these globals are used
    
###################
# Data Processing #
###################

def process_data() -> Tuple[dict, dict]:
    with open(GRAPH_IDS_FILE, 'r') as graph_ids_file_handle:
        node_id_to_graph_id = dict(enumerate(map(int, graph_ids_file_handle.readlines()), start=1))
        graph_id_to_graph = {graph_id: nx.Graph() for graph_id in set(node_id_to_graph_id.values())}
        for node_id, graph_id in node_id_to_graph_id.items():
            graph_id_to_graph[graph_id].add_node(node_id)
    with open(NODE_LABELS_FILE, 'r') as node_labels_file_handle:
        node_labels_file_lines = node_labels_file_handle.readlines()
        assert len(node_labels_file_lines) == len(node_id_to_graph_id)
        for node_id, node_label in enumerate(map(int, node_labels_file_lines), start=1):
            graph_id = node_id_to_graph_id[node_id]
            graph = graph_id_to_graph[graph_id]
            graph.nodes[node_id]['node_label'] = node_label
    with open(EDGES_FILE, 'r') as edges_file_handle:
        edges_file_lines = edges_file_handle.readlines()
        split_lines = eager_map(lambda s: s.split(','), edges_file_lines)
        assert set(map(len, split_lines)) == {2}
        edges = map(lambda l: (int(l[0]), int(l[1])), split_lines)
    with open(EDGE_LABELS_FILE, 'r') as edge_labels_file_handle:
        edge_labels = map(int, edge_labels_file_handle.readlines())
    for (src_id, dst_id), edge_label in zip(edges, edge_labels):
        graph_id = node_id_to_graph_id[src_id]
        graph = graph_id_to_graph[graph_id]
        assert dst_id in graph.nodes
        graph.add_edge(src_id, dst_id, edge_label=edge_label)
    with open(GRAPH_LABELS_FILE, 'r') as graph_labels_file_handle:
        graph_id_to_graph_label = dict(enumerate(map(lambda label: 1 if label.strip()=='1' else 0, graph_labels_file_handle.readlines()), start=1))
        assert set(graph_id_to_graph_label.values()) == {0, 1}
        assert len(graph_id_to_graph_label) == 188
    graph_id_to_graph = {graph_id: nx.convert_node_labels_to_integers(graph) for graph_id, graph in graph_id_to_graph.items()}
    assert set(graph_id_to_graph.keys()) == set(graph_id_to_graph_label.keys())
    return graph_id_to_graph, graph_id_to_graph_label

#######################
# graph2vec Utilities #
#######################
    
class Graph2VecHyperParameterSearchObjective:
    def __init__(self, graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int], process_id_queue: object):
        # process_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dyanmically
        self.graph_id_to_graph: OrderedDict = OrderedDict(((graph_id, graph_id_to_graph[graph_id]) for graph_id in sorted(graph_id_to_graph.keys())))
        self.graph_id_to_graph_label: OrderedDict = OrderedDict(((graph_id, graph_id_to_graph_label[graph_id]) for graph_id in sorted(graph_id_to_graph_label.keys())))
        self.process_id_queue = process_id_queue

    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters = {
            'wl_iterations': int(trial.suggest_int('wl_iterations', 1, 6)),
            'dimensions': int(trial.suggest_int('dimensions', 100, 500)),
            'epochs': int(trial.suggest_int('epochs', 10, 15)),
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 1e-1),
        }
        return hyperparameters

    def train_model(self, wl_iterations: int, dimensions: int, epochs: int, learning_rate: float) -> float:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        graphs = self.graph_id_to_graph.values()
        assert all(nx.is_connected(graph) for graph in graphs)
        assert not any(nx.is_directed(graph) for graph in graphs)
        assert all(list(range(graph.number_of_nodes())) == sorted(graph.nodes()) for graph in graphs)
        
        weisfeiler_lehman_features = [karateclub.utils.treefeatures.WeisfeilerLehmanHashing(graph, wl_iterations, False, False) for graph in graphs]
        documents = [gensim.models.doc2vec.TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(weisfeiler_lehman_features)]
        
        model = gensim.models.doc2vec.Doc2Vec(
            documents,
            vector_size=dimensions,
            window=0,
            min_count=0,
            dm=0,
            sample=0.0001,
            workers=1,
            epochs=epochs,
            alpha=learning_rate,
            seed=RANDOM_SEED
        )
        
        graph_embedding_matrix: np.ndarray = np.array([model.docvecs[str(i)] for i in range(len(documents))])
        graph_id_to_graph_embeddings = VectorDict(self.graph_id_to_graph.keys(), graph_embedding_matrix)

        checkpoint_directory = self.__class__.checkpoint_directory_from_hyperparameters(wl_iterations, dimensions, epochs, learning_rate)
        if not os.path.isdir(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        
        saved_model_location = os.path.join(checkpoint_directory, DOC2VEC_MODEL_FILE_BASENAME)
        model.save(saved_model_location)

        keyed_embedding_pickle_location = os.path.join(checkpoint_directory, KEYED_EMBEDDING_PICKLE_FILE_BASENAME)
        with open(keyed_embedding_pickle_location, 'wb') as file_handle:
            pickle.dump(graph_id_to_graph_embeddings, file_handle)
        
        result_summary_json_file_location = os.path.join(checkpoint_directory, RESULT_SUMMARY_JSON_FILE_BASENAME)
        with open(result_summary_json_file_location, 'w') as file_handle:
            json.dump({
                'saved_model_location': saved_model_location,
                'keyed_embedding_pickle_location': keyed_embedding_pickle_location, 
            }, file_handle, indent=4)
        
        return 1 # stub

    @staticmethod
    def checkpoint_directory_from_hyperparameters(wl_iterations: int, dimensions: int, epochs: int, learning_rate: float) -> str:
        checkpoint_directory = os.path.join(
            GRAPH2VEC_CHECKPOINT_DIR,
            f'wl_iterations_{int(wl_iterations)}_' \
            f'dimensions_{int(dimensions)}_' \
            f'epochs_{int(epochs)}_' \
            f'learning_rate_{learning_rate:.5g}'
        )
        return checkpoint_directory
    
    def __call__(self, trial: optuna.Trial) -> float:
        process_id = self.process_id_queue.get()
        
        hyperparameters = self.get_trial_hyperparameters(trial)

        checkpoint_dir = self.__class__.checkpoint_directory_from_hyperparameters(**hyperparameters)
        print(f'Starting training for {checkpoint_dir} with sub-process #{process_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    loss = self.train_model(**hyperparameters)
        except Exception as exception:
            self.process_id_queue.put(process_id)
            raise exception
        self.process_id_queue.put(process_id)
        return loss

def get_number_of_graph2vec_hyperparameter_search_trials(study: optuna.Study) -> int:
    df = study.trials_dataframe()
    if len(df) == 0:
        number_of_remaining_trials = NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_TRIALS
    else:
        number_of_completed_trials = df.state.eq('COMPLETE').sum()
        number_of_remaining_trials = NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_TRIALS - number_of_completed_trials
    return number_of_remaining_trials

def graph2vec_hyperparameter_search(graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int]) -> None:
    set(graph_id_to_graph.keys()) == set(graph_id_to_graph_label.keys())
    study = optuna.create_study(study_name=GRAPH2VEC_STUDY_NAME, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=GRAPH2VEC_DB_URL, direction='minimize', load_if_exists=True)
    number_of_trials = get_number_of_graph2vec_hyperparameter_search_trials(study)
    optimize_kawrgs = dict(
        n_trials=number_of_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )
    with mp.Manager() as manager:
        process_id_queue = manager.Queue()
        more_itertools.consume((process_id_queue.put(process_id) for process_id in range(NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_PROCESSES)))
        optimize_kawrgs['func'] = Graph2VecHyperParameterSearchObjective(graph_id_to_graph, graph_id_to_graph_label, process_id_queue)
        optimize_kawrgs['n_jobs'] = NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_PROCESSES
        with joblib.parallel_backend('multiprocessing', n_jobs=NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_PROCESSES):
            study.optimize(**optimize_kawrgs)
    return

####################
# MUTAG Classifier #
####################

class MUTAGClassifierHyperParameterSearchObjective:
    def __init__(self, graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int], gpu_id_queue: object):
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dyanmically
        self.graph_id_to_graph = graph_id_to_graph
        self.graph_id_to_graph_label = graph_id_to_graph_label
        self.gpu_id_queue = gpu_id_queue
        self._graph2vec_study_df = optuna.create_study(study_name=GRAPH2VEC_STUDY_NAME, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=GRAPH2VEC_DB_URL, direction='minimize', load_if_exists=True).trials_dataframe()

    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        graph2vec_trial_indices = self._graph2vec_study_df[self._graph2vec_study_df.state.eq('COMPLETE')].index.tolist()
        hyperparameters = {
            'batch_size': int(trial.suggest_int('batch_size', 1, 32)),
            'graph2vec_trial_index': int(trial.suggest_categorical('graph2vec_trial_index', graph2vec_trial_indices)),
            'number_of_layers': int(trial.suggest_int('number_of_layers', 1, 5)),
            'gradient_clip_val': trial.suggest_uniform('gradient_clip_val', 1.0, 25.0), 
            'dropout_probability': trial.suggest_uniform('dropout_probability', 0.0, 1.0),
        }
        assert set(hyperparameters.keys()) == set(MUTAGClassifier.hyperparameter_names)
        return hyperparameters

    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get()

        hyperparameters = self.get_trial_hyperparameters(trial)
        checkpoint_dir = MUTAGClassifier.checkpoint_directory_from_hyperparameters(**hyperparameters) # @todo do something with this
        LOGGER.info(f'Starting MUTAG classifier training for {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = MUTAGClassifier.train_model(gpus=[gpu_id], graph_id_to_graph=self.graph_id_to_graph, graph_id_to_graph_label=self.graph_id_to_graph_label, **hyperparameters)
        except Exception as exception:
            if self.gpu_id_queue is not None:
                self.gpu_id_queue.put(gpu_id)
            raise exception
        if self.gpu_id_queue is not None:
            self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def get_number_of_mutag_classifier_hyperparameter_search_trials(study: optuna.Study) -> int:
    df = study.trials_dataframe()
    if len(df) == 0:
        number_of_remaining_trials = NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS
    else:
        number_of_completed_trials = df.state.eq('COMPLETE').sum()
        number_of_remaining_trials = NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS - number_of_completed_trials
    return number_of_remaining_trials

def mutag_classifier_hyperparameter_search(graph_id_to_graph: Dict[int, nx.Graph], graph_id_to_graph_label: Dict[int, int]) -> None:
    study = optuna.create_study(study_name=MUTAG_CLASSIFIER_STUDY_NAME, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=MUTAG_CLASSIFIER_DB_URL, direction='minimize', load_if_exists=True)
    number_of_trials = get_number_of_mutag_classifier_hyperparameter_search_trials(study)
    optimize_kawrgs = dict(
        n_trials=number_of_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )
    with mp.Manager() as manager:
        gpu_id_queue = manager.Queue()
        more_itertools.consume((gpu_id_queue.put(gpu_id) for gpu_id in GPU_IDS))
        optimize_kawrgs['func'] = MUTAGClassifierHyperParameterSearchObjective(graph_id_to_graph, graph_id_to_graph_label, gpu_id_queue)
        optimize_kawrgs['n_jobs'] = len(GPU_IDS)
        with joblib.parallel_backend('multiprocessing', n_jobs=len(GPU_IDS)):
            study.optimize(**optimize_kawrgs)
    return
    
##########
# Driver #
##########

@debug_on_error
def main() -> None:
    graph_id_to_graph, graph_id_to_graph_label = process_data()
    graph2vec_hyperparameter_search(graph_id_to_graph, graph_id_to_graph_label)
    mutag_classifier_hyperparameter_search(graph_id_to_graph, graph_id_to_graph_label)
    return

if __name__ == '__main__':
    main()
