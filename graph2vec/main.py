
'''

Sections:
* Imports
* Globals

'''

###########
# Imports #
###########

import multiprocessing as mp
import pandas as pd
import networkx as nx
from pandarallel import pandarallel
from functools import reduce
from typing import Tuple

from misc_utilities import *

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

EDGES_FILE = './data/Mutag.edges'
NODE_LABELS_FILE = './data/Mutag.node_labels'
LINK_LABELS_FILE = './data/Mutag.link_labels'

GRAPH_LABELS_FILE = './data/Mutag.graph_labels'
GRAPH_INDEX_FILE = './data/Mutag.graph_idx'

###################
# Data Processing #
###################

def process_data() -> None:
    nx_graph = nx.Graph()
    with open(EDGES_FILE, 'r') as edges_file_handle:
        edges_file_lines = edges_file_handle.readlines()
        split_lines = map(lambda s: s.split(','), edges_file_lines)
        edges = eager_map(lambda l: (int(l[0]), int(l[1])), split_lines)
        nodes = functools.reduce(lambda l, s: s.union(l), edges, set())
        with open(LINK_LABELS_FILE, 'r') as link_labels_file_handle:
            labels = eager_map(int, link_labels_file_handle.readlines())
            assert len(labels) == len(edges)
            nx_graph.add_weighted_edges_from(((src, dst, label) for (src, dst), label in zip(edges, labels)), weight='label')
    with open(NODE_LABELS_FILE, 'r') as node_labels_file_handle:
        assert nodes == set(nx_graph.nodes) == set(map(lambda line: int(line.split(',')[1]), node_labels_file_handle.readlines()))
    with open(GRAPH_INDEX_FILE, 'r') as f:
        f.readlines()
    edge_count = 1
    node_count = 1
    graph_count = 1
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    process_data()
    return

if __name__ == '__main__':
    main()
