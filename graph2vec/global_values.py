
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
import gensim
import logging
import torch
import numpy as np
from typing import Union, Iterable

from misc_utilities import *

# @todo make sure these imports are used

###########
# Logging #
###########

LOGGER_NAME = 'mutag_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './logs.txt'
LOGGER_STREAM_HANDLER = logging.StreamHandler()

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    return

_initialize_logger()

@contextmanager
def training_logging_suppressed() -> Generator:
    logger = logging.root.manager.loggerDict['lightning']
    lightning_original_disability = logger.disabled
    logger.disabled = True
    logger_stream_handler_original_stream = LOGGER_STREAM_HANDLER.stream
    with open(os.devnull, 'w') as dev_null:
        LOGGER_STREAM_HANDLER.setStream(dev_null)
        yield
    logger.disabled = lightning_original_disability
    LOGGER_STREAM_HANDLER.setStream(logger_stream_handler_original_stream)
    return

###########
# Globals #
###########

RANDOM_SEED = 1234

KEYED_EMBEDDING_PICKLE_FILE_BASENAME = 'doc2vec_keyed_embedding.pickle'
DOC2VEC_MODEL_FILE_BASENAME = 'doc2vec.model'
RESULT_SUMMARY_JSON_FILE_BASENAME = 'result_summary.json'

# graph2vec Globals

GRAPH2VEC_CHECKPOINT_DIR = './checkpoints_graph2vec'
GRAPH2VEC_STUDY_NAME = 'graph2vec'
GRAPH2VEC_DB_URL = 'sqlite:///graph2vec.db'

NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_TRIALS = 10 # 9999 # @todo enable this
NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_PROCESSES = 50

if not os.path.isdir(GRAPH2VEC_CHECKPOINT_DIR):
    os.makedirs(GRAPH2VEC_CHECKPOINT_DIR)

# MUTAG Classifier Globals
    
MUTAG_CLASSIFIER_CHECKPOINT_DIR = './checkpoints_mutag_classifier'
MUTAG_CLASSIFIER_STUDY_NAME = 'classifier-mutag'
MUTAG_CLASSIFIER_DB_URL = 'sqlite:///classifier-mutag.db'

NUMBER_OF_MUTAG_CLASSIFIER_HYPERPARAMETER_TRIALS = 1 # 9999 # @todo enable this
GPU_IDS = [0, 1, 2, 3]

if not os.path.isdir(MUTAG_CLASSIFIER_CHECKPOINT_DIR):
    os.makedirs(MUTAG_CLASSIFIER_CHECKPOINT_DIR)

###############
# Vector Dict #
###############

class VectorDict():
    '''Index into matrix by keys'''

    def __init__(self, keys: Iterable, matrix: Union[torch.tensor, np.ndarray]):
        assert len(matrix.shape) == 2
        assert len(keys) == matrix.shape[0]
        self.key_to_index_map = dict(map(reversed, enumerate(keys)))
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
        self.matrix = matrix

    def __getitem__(self, key: Union[int, torch.Tensor, Iterable]) -> np.ndarray:
        if isinstance(key, int):
            item = self.matrix[self.key_to_index_map[key]]
        else:
            item = self.matrix[[self.key_to_index_map[int(sub_key)] for sub_key in key]]
        return item

    def to(self, device_spec: Union[str, torch.device]) -> 'VectorDict':
        self.matrix.to(device_spec)
        return self

###################
# Nadam Optimizer #
###################

def monkey_patch_nadam() -> None:

    # Stolen from https://github.com/pytorch/pytorch/pull/1408
    
    class Nadam(torch.optim.Optimizer):
        """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).
        It has been proposed in `Incorporating Nesterov Momentum into Adam`__.
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 2e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            schedule_decay (float, optional): momentum schedule decay (default: 4e-3)
        __ http://cs229.stanford.edu/proj2015/054_report.pdf
        __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        """
    
        def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, schedule_decay=4e-3):
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, schedule_decay=schedule_decay)
            super(Nadam, self).__init__(params, defaults)
    
        def step(self, closure=None):
            """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()
    
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]
    
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['m_schedule'] = 1.
                        state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                        state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
    
                    # Warming momentum schedule
                    m_schedule = state['m_schedule']
                    schedule_decay = group['schedule_decay']
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    eps = group['eps']
    
                    state['step'] += 1
    
                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)
    
                    momentum_cache_t = beta1 * \
                        (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
                    momentum_cache_t_1 = beta1 * \
                        (1. - 0.5 *
                         (0.96 ** ((state['step'] + 1) * schedule_decay)))
                    m_schedule_new = m_schedule * momentum_cache_t
                    m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                    state['m_schedule'] = m_schedule_new
    
                    # Decay the first and second moment running average coefficient
                    bias_correction2 = 1 - beta2 ** state['step']
    
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    exp_avg_sq_prime = exp_avg_sq.div(1. - bias_correction2)
    
                    denom = exp_avg_sq_prime.sqrt_().add_(group['eps'])
    
                    p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
                    p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)
    
            return loss

    torch.optim.Nadam = Nadam
    return

monkey_patch_nadam()
