#!/usr/bin/python3
'#!/usr/bin/python3 -OO' # @ todo use this

'''
'''

# @todo fill in the doc string

###########
# Imports #
###########

import os
import multiprocessing as mp
import networkx as nx
import pandas as pd
from typing import Tuple

from misc_utilities import *

###########
# Globals #
###########

RAW_DATA_CSV = './raw_data.csv'

RELEVANT_COLUMNS = [
    'PASSENGERS',
    'DISTANCE',
    'REGION',
    'YEAR',
    'QUARTER',
    'MONTH',
    'DISTANCE_GROUP',
    'CLASS',
    'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM',
    'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME',   'DEST_STATE_ABR',   'DEST_STATE_NM',
]

EDGE_COLOR = "#cccccc"
DEFAULT_NODE_COLOR = '#008080'
NUMBER_OF_SPRING_LAYOUT_ITERATIONS = 10

DEFAULT_NODE_SIZE = 10
MINIMUM_NODE_SIZE = 1
NODE_SIZE_GROWTH_POTENTIAL = 300 - MINIMUM_NODE_SIZE

KATZ_ALHPA = 0.01

###################
# Preprocess Data #
###################

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relevant_df = df[RELEVANT_COLUMNS]
    relevant_df = relevant_df[df.PASSENGERS != 0.0]
    passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']).PASSENGERS.sum().reset_index()
    origin_city_market_id_info_df = df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']].rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID', 'ORIGIN': 'AIRPORT', 'ORIGIN_CITY_NAME': 'CITY_NAME'})
    dest_city_market_id_info_df = df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']].rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID', 'DEST': 'AIRPORT', 'DEST_CITY_NAME': 'CITY_NAME'})
    city_market_id_info_df = pd.concat([origin_city_market_id_info_df, dest_city_market_id_info_df])
    city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})
    return passenger_flow_df, city_market_id_info_df

###########################
# Visualization Utilities #
###########################

def draw_graph_to_file(output_location: str, graph: nx.Graph, **kwargs) -> None:
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        layout = nx.spring_layout(graph, iterations=NUMBER_OF_SPRING_LAYOUT_ITERATIONS)
        nx.draw_networkx_edges(graph,
                               layout, 
                               width=1,
                               alpha=1.0,
                               edge_color=EDGE_COLOR)
        assert len(kwargs) <= 1
        if 'label_to_nodes_map' in kwargs:
            label_to_nodes_map = kwargs['label_to_nodes_map']
            for label, nodes in label_to_nodes_map.items():
                node_color = random_hex_color()
                nx.draw_networkx_nodes(graph,
                                       layout,
                                       nodelist=nodes,
                                       node_color=node_color,
                                       node_size=DEFAULT_NODE_SIZE,
                                       ax=plot)
            plot.set_title(f'{len(label_to_nodes_map)} communities')
        elif 'node_to_value_map' in kwargs:
            node_to_value_map = kwargs['node_to_value_map']
            if len(node_to_value_map) > 0:
                max_value = max(node_to_value_map.values())
                for node, value in node_to_value_map.items():
                    normalized_value = value/max_value
                    nx.draw_networkx_nodes(graph,
                                           layout,
                                           nodelist=[node],
                                           node_color=DEFAULT_NODE_COLOR,
                                           node_size=MINIMUM_NODE_SIZE+NODE_SIZE_GROWTH_POTENTIAL*normalized_value,
                                           ax=plot)
        else:
            assert len(kwargs)==0
            nx.draw_networkx_nodes(graph,
                                   layout,
                                   nodelist=graph.nodes,
                                   node_color=DEFAULT_NODE_COLOR,
                                   node_size=DEFAULT_NODE_SIZE,
                                   ax=plot)
        figure.savefig(output_location)
    return

def write_passenger_flow_vertex_ranking_to_csv(node_to_label_map: dict, city_market_id_info_df: dict, csv_file: str) -> None:
    if len(node_to_label_map) == 0:
        with open(csv_file, 'w') as f:
            f.write('node,label')
    else:
        label_df = pd.DataFrame.from_dict(node_to_label_map, orient='index').rename(columns={0:'value'})
        label_df = label_df.join(city_market_id_info_df).sort_values('value', ascending=False)
        label_df.to_csv(csv_file, index_label='CITY_MARKET_ID')
    return

def visualize_passenger_flow_graph(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    draw_graph_to_file('./output_data/passenger_flow_graph.png', passenger_flow_graph)
    return

def visualize_passenger_flow_graph_betweenness_centrality(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    betweenness_centrality = nx.betweenness_centrality(passenger_flow_graph)
    draw_graph_to_file('./output_data/betweenness_centrality.png', passenger_flow_graph, node_to_value_map=betweenness_centrality)
    write_passenger_flow_vertex_ranking_to_csv(betweenness_centrality, city_market_id_info_df, './output_data/betweenness_centrality.csv')
    return

def visualize_passenger_flow_graph_eigenvector_centrality(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    eigenvector_centrality = nx.eigenvector_centrality(passenger_flow_graph, max_iter=1000)
    draw_graph_to_file('./output_data/eigenvector_centrality.png', passenger_flow_graph, node_to_value_map=eigenvector_centrality)
    write_passenger_flow_vertex_ranking_to_csv(eigenvector_centrality, city_market_id_info_df, './output_data/eigenvector_centrality.csv')
    return

def visualize_passenger_flow_graph_hits(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    node_to_hub_value_map, node_to_authority_value_map = nx.hits(passenger_flow_graph, max_iter=1000, normalized=True)
    draw_graph_to_file('./output_data/hits_hub.png', passenger_flow_graph, node_to_value_map=node_to_hub_value_map)
    draw_graph_to_file('./output_data/hits_authority.png', passenger_flow_graph, node_to_value_map=node_to_authority_value_map)
    write_passenger_flow_vertex_ranking_to_csv(node_to_hub_value_map, city_market_id_info_df, './output_data/hits_hub.csv')
    write_passenger_flow_vertex_ranking_to_csv(node_to_authority_value_map, city_market_id_info_df, './output_data/hits_authority.csv')
    return

def visualize_passenger_flow_graph_katz(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    katz_centrality = nx.katz_centrality(passenger_flow_graph, KATZ_ALHPA)
    draw_graph_to_file('./output_data/katz_centrality.png', passenger_flow_graph, node_to_value_map=katz_centrality)
    write_passenger_flow_vertex_ranking_to_csv(katz_centrality, city_market_id_info_df, './output_data/katz_centrality.csv')
    return

def visualize_passenger_flow_graph_pagerank(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    pagerank = nx.pagerank(passenger_flow_graph)
    draw_graph_to_file('./output_data/pagerank.png', passenger_flow_graph, node_to_value_map=pagerank)
    write_passenger_flow_vertex_ranking_to_csv(pagerank, city_market_id_info_df, './output_data/pagerank.csv')
    return

def visualize_passenger_flow_graph_vertex_rankings(passenger_flow_graph: nx.DiGraph, city_market_id_info_df: dict) -> None:
    processes: List[mp.Process] = []
    visualization_functions = [
        visualize_passenger_flow_graph,
        visualize_passenger_flow_graph_betweenness_centrality,
        visualize_passenger_flow_graph_eigenvector_centrality,
        visualize_passenger_flow_graph_hits,
        visualize_passenger_flow_graph_katz,
        visualize_passenger_flow_graph_pagerank,
    ]
    if not os.path.isdir('./output_data/'):
        os.makedirs('./output_data/')
    for visualization_function in visualization_functions:
        process = mp.Process(target=visualization_function, args=(passenger_flow_graph, city_market_id_info_df))
        process.start()
        processes.append(process)
    for process in tqdm_with_message(processes, post_yield_message_func = lambda index: f'Join Process {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
        process.join()
    return


##########
# Driver #
##########

@debug_on_error
def main() -> None:
    raw_data_df = pd.read_csv(RAW_DATA_CSV)
    passenger_flow_df, city_market_id_info_df = preprocess_data(raw_data_df)
    passenger_flow_graph = nx.from_pandas_edgelist(passenger_flow_df, source='ORIGIN_CITY_MARKET_ID', target='DEST_CITY_MARKET_ID', edge_attr='PASSENGERS', create_using=nx.DiGraph)
    visualize_passenger_flow_graph_vertex_rankings(passenger_flow_graph, city_market_id_info_df)
    return

if __name__ == '__main__':
    main()
