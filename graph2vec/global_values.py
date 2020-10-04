
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

###########
# Globals #
###########

RANDOM_SEED = 1234

# graph2vec Globals

GRAPH2VEC_CHECKPOINT_DIR = './checkpoints_graph2vec'
GRAPH2VEC_STUDY_NAME = 'graph2vec'
GRAPH2VEC_DB_URL = 'sqlite:///graph2vec.db'

NUMBER_OF_GRAPH2VEC_HYPERPARAMETER_TRIALS = 1 # 9999 # @todo enable this
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

    def __init__(self, keys: Iterable, matrix: np.ndarray):
        assert len(matrix.shape) == 2
        assert len(keys) == matrix.shape[0]
        self.key_to_index_map = dict(map(reversed, enumerate(keys)))
        self.matrix = matrix

    def __getitem__(self, key) -> np.ndarray:
        return self.matrix[self.key_to_index_map[key]]

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

#######################
# Monkey Patch Gensim #
#######################

# def monkey_patch_gensim_doc2vec_compute_loss():
#     '''The official gensim doc2vec interface doesn't support the compute_loss attribute. '''
#     def train(self, documents=None, corpus_file=None, total_examples=None, total_words=None,
#               epochs=None, start_alpha=None, end_alpha=None,
#               word_count=0, queue_factor=2, report_delay=1.0, callbacks=()):
        
#         kwargs = {}

#         if corpus_file is None and documents is None:
#             raise TypeError("Either one of corpus_file or documents value must be provided")

#         if corpus_file is not None and documents is not None:
#             raise TypeError("Both corpus_file and documents must not be provided at the same time")

#         if documents is None and not os.path.isfile(corpus_file):
#             raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)

#         if documents is not None and not isinstance(documents, Iterable):
#             raise TypeError("documents must be an iterable of list, got %r instead" % documents)

#         if corpus_file is not None:
#             # Calculate offsets for each worker along with initial doctags (doctag ~ document/line number in a file)
#             offsets, start_doctags = self._get_offsets_and_start_doctags_for_corpusfile(corpus_file, self.workers)
#             kwargs['offsets'] = offsets
#             kwargs['start_doctags'] = start_doctags

#         kwargs['compute_loss'] = getattr(self, 'compute_loss', False)
#         with open('/tmp/test.py', 'a') as f: # @todo remove this
#             import traceback ; traceback.print_stack(file=f)
#             f.write('\n\n')
#             f.write(f"Doc2Vec.train\n")
#             f.write(f"self.compute_loss {repr(self.compute_loss)}\n")
#             f.write(f"kwargs {repr(kwargs)}\n")
#             f.write('\n\n')
#         super(gensim.models.doc2vec.Doc2Vec, self).train(
#             sentences=documents, corpus_file=corpus_file, total_examples=total_examples, total_words=total_words,
#             epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
#             queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks, **kwargs)
    
#     gensim.models.doc2vec.Doc2Vec.train = train
    
# monkey_patch_gensim_doc2vec_compute_loss()
