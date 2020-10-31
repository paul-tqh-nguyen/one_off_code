
'''

@todo update this string

'''

###########
# Imports #
###########

import argparse
import functools
import nvgpu
import random
import more_itertools
import joblib
import optuna
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from typing import Dict, Tuple, Set

from misc_utilities import *
from link_predictor import LinkPredictor

# @todo make sure these imports are used

###########
# Globals #
###########

GPU_IDS = eager_map(int, nvgpu.available_gpus())

STUDY_NAME = 'link-predictor'
DB_URL = 'sqlite:///link-predictor.db'

HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION = './docs/hyperparameter_search_results.json'
NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS = 10_000
RESULT_SUMMARY_JSON_FILE_BASENAME = 'result_summary.json'

###################
# Data Processing #
###################

def process_data() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    remaining_graph = nx.Graph()
    with open('./data/facebook_combined.txt', 'r') as f:
        lines = f.readlines()
    edges = [tuple(eager_map(int, line.split())) for line in lines]
    remaining_graph.add_edges_from(edges)

    assert len(remaining_graph.nodes) == 4039
    assert len(remaining_graph.edges) == 88234
    
    nodes = list(remaining_graph.nodes())
    num_edges_to_sample = len(remaining_graph.edges) // 2
    assert num_edges_to_sample == 44117
    assert set(nodes) == set(range(4039))
    
    positive_edges = set()
    negative_edges = set()
    
    while len(negative_edges) < num_edges_to_sample:
        random_edge = (random.choice(nodes), random.choice(nodes))
        if len(set(random_edge)) == 2:
            random_edge = tuple(sorted(random_edge))
            if random_edge not in negative_edges:
                negative_edges.add(random_edge)
    
    while len(positive_edges) < num_edges_to_sample:
        random_edge = random.choice(edges)
        if random_edge not in positive_edges:
            remaining_graph.remove_edge(*random_edge)
            if nx.is_connected(remaining_graph):
                positive_edges.add(random_edge)
            else:
                remaining_graph.add_edge(*random_edge)

    positive_edges = np.array(list(positive_edges))
    negative_edges = np.array(list(negative_edges))
    
    assert nx.is_connected(remaining_graph)
    return remaining_graph, positive_edges, negative_edges

########################################
# Link Predictor Hyperparameter Search #
########################################

class LinkPredictorHyperParameterSearchObjective:
    def __init__(self, gpu_id_queue: object): # @todo update these inputs
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classes are generated dynamically
        self.gpu_id_queue = gpu_id_queue

    def get_trial_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters = { # @todo update these
            'wl_iterations': int(trial.suggest_int('wl_iterations', 1, 20)),
        }
        assert set(hyperparameters.keys()) == set(LinkPredictor.hyperparameter_names)
        return hyperparameters
    
    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get()

        hyperparameters = self.get_trial_hyperparameters(trial)
        checkpoint_dir = LinkPredictor.checkpoint_directory_from_hyperparameters(**hyperparameters)
        LOGGER.info(f'Starting link predictor training for {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = LinkPredictor.train_model(gpus=[gpu_id], **hyperparameters) # @todo update this line
        except Exception as exception:
            if self.gpu_id_queue is not None:
                self.gpu_id_queue.put(gpu_id)
            raise exception
        if self.gpu_id_queue is not None:
            self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def get_number_of_link_predictor_hyperparameter_search_trials(study: optuna.Study) -> int:
    df = study.trials_dataframe()
    if len(df) == 0:
        number_of_remaining_trials = NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS
    else:
        number_of_completed_trials = df.state.eq('COMPLETE').sum()
        number_of_remaining_trials = NUMBER_OF_LINK_PREDICTOR_HYPERPARAMETER_TRIALS - number_of_completed_trials
    return number_of_remaining_trials

def load_hyperparameter_search_study() -> optuna.Study:
    return optuna.create_study(study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=DB_URL, direction='minimize', load_if_exists=True)

def hyperparameter_search_study_df() -> pd.DataFrame:
    return load_hyperparameter_search_study().trials_dataframe()

def link_predictor_hyperparameter_search() -> None: # @todo update this signature
    study = load_hyperparameter_search_study()
    number_of_trials = get_number_of_link_predictor_hyperparameter_search_trials(study)
    optimize_kawrgs = dict(
        n_trials=number_of_trials,
        gc_after_trial=True,
        catch=(Exception,),
    )
    with mp.Manager() as manager:
        gpu_id_queue = manager.Queue()
        more_itertools.consume((gpu_id_queue.put(gpu_id) for gpu_id in GPU_IDS))
        optimize_kawrgs['func'] = LinkPredictorHyperParameterSearchObjective(graph_id_to_graph, graph_id_to_graph_label, gpu_id_queue)
        optimize_kawrgs['n_jobs'] = len(GPU_IDS)
        with joblib.parallel_backend('multiprocessing', n_jobs=len(GPU_IDS)):
            with training_logging_suppressed():
                study.optimize(**optimize_kawrgs)
    return

#########################################
# Hyperparameter Search Result Analysis #
#########################################

def analyze_hyperparameter_search_results() -> None:
    df = hyperparameter_search_study_df()
    df = df.loc[df.state=='COMPLETE']
    params_prefix = 'params_'
    assert set(LinkPredictor.hyperparameter_names) == {column_name[len(params_prefix):] for column_name in df.columns if column_name.startswith(params_prefix)}
    result_summary_dicts = []
    for row in df.itertuples():
        hyperparameter_dict = {hyperparameter_name: getattr(row, params_prefix+hyperparameter_name) for hyperparameter_name in LinkPredictor.hyperparameter_names}
        checkpoint_dir = LinkPredictor.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
        result_summary_file_location = os.path.join(checkpoint_dir, RESULT_SUMMARY_JSON_FILE_BASENAME)
        with open(result_summary_file_location, 'r') as f:
            result_summary_dict = json.load(f)
            result_summary_dict['duration_seconds'] = row.duration.seconds
        result_summary_dicts.append(result_summary_dict)
    with open(HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION, 'w') as f:
        json.dump(result_summary_dicts, f, indent=4)
    LOGGER.info(f'Hyperparameter result summary saved to {HYPERPARAMETER_ANALYSIS_JSON_FILE_LOCATION} .')
    return

#################
# Default Model #
#################

def train_default_model(graph: nx.Graph, positive_edges: np.ndarray, negative_edges: np.ndarray) -> None:
    LinkPredictor.train_model(
        gpus=GPU_IDS,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        graph=graph,
        p=1,
        q=0.5,
        walks_per_node=5,
        walk_length=5,
        node2vec_epochs=50,
        node2vec_learning_rate=1e-3,
        embedding_size=300,
        link_predictor_learning_rate=1e-3,
        link_predictor_batch_size=256,
        link_predictor_gradient_clip_val=1.0,
    )
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-train-default-model', action='store_true', help='Train the default classifier.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Perform several trials of hyperparameter search for the link predictor.')
    parser.add_argument('-analyze-hyperparameter-search-results', action='store_true', help=f'Analyze completed hyperparameter search trials.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,map(bool,vars(args).values())))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.train_default_model:
        graph, positive_edges, negative_edges = process_data()
        train_default_model(graph, positive_edges, negative_edges)
    elif args.hyperparameter_search:
        graph, positive_edges, negative_edges = process_data()
        link_predictor_hyperparameter_search(graph, positive_edges, negative_edges)
    elif args.analyze_hyperparameter_search_results:
        analyze_hyperparameter_search_results()
    else:
        raise ValueError('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
