#!/usr/bin/python3

"""
"""

# @todo update the doc string

###########
# Imports #
###########

import math
import tqdm
import matplotlib.cm
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple

from misc_utilities import *

###########
# Globals #
###########

with warnings_suppressed():
    tqdm.tqdm.pandas()

MAX_NUMBER_OF_CLUSTERS_TO_TRY = 10

# https://www.kaggle.com/carrie1/ecommerce-data
RAW_DATA_CSV_FILE_LOCATION = './data/data.csv'

RFM_PCA_VISUALIZATION_OUTPUT_PNG_FILE_LOCATION = './rfm_pca.png'
RFM_PCA_VISUALIZATION_WITH_CLUSTERS_OUTPUT_PNG_FILE_LOCATION_TEMPLATE = './rfm_pca_{cluster_count}_clusters.png'
RFM_CLUSTER_SILHOUETTE_SCORE_OUPUT_PNG_FILE_LOCATION = './rfm_silhouette_scores.png'
CUSTOMER_SIMILARITY_LABEL_OUTPUT_PNG_FILE_LOCATION = './customer_similarity_communities.png'

###################
# Data Processing #
###################

def clean_data(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df.InvoiceDate = data_df.InvoiceDate = pd.to_datetime(data_df.InvoiceDate, format="%m/%d/%Y %H:%M")
    data_df.drop(data_df.index[data_df.CustomerID != data_df.CustomerID], inplace=True)
    data_df = data_df.astype({'CustomerID': int}, copy=False)
    assert len(data_df[data_df.isnull().any(axis=1)])==0, "Raw data contains NaN"
    return data_df

def append_last_purchase_date_data(rfm_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    customer_latest_purchse_date_df = data_df.groupby('CustomerID').agg({'InvoiceDate': max}).rename(columns={'InvoiceDate': 'LastPurchaseDate'})
    rfm_df = rfm_df.join(customer_latest_purchse_date_df, on='CustomerID')
    latest_date = data_df.InvoiceDate.max()
    rfm_df['DaysSinceLastPurchase'] = rfm_df.LastPurchaseDate.progress_map(lambda date: (latest_date - date).days)
    rfm_df.drop(columns=['LastPurchaseDate'], inplace=True)
    return rfm_df

def append_invoice_count_data(rfm_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    customer_frequency_series = data_df.groupby('CustomerID').InvoiceNo.count()
    customer_frequency_series.name = 'InvoiceCount'
    rfm_df = rfm_df.join(customer_frequency_series, on='CustomerID')
    return rfm_df

def append_overall_purchase_amount_data(rfm_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    total_purchase_amount_series = data_df.apply(lambda row: row.Quantity * row.UnitPrice, axis=1)
    total_purchase_amount_df = data_df[['CustomerID']].copy()
    total_purchase_amount_df['InvoicePurchaseAmount'] = total_purchase_amount_series
    overall_purchase_amount_df = total_purchase_amount_df.groupby('CustomerID').agg({'InvoicePurchaseAmount': sum}).rename(columns={'InvoicePurchaseAmount': 'CustomerOverallPurchaseAmount'})
    rfm_df = rfm_df.join(overall_purchase_amount_df, on='CustomerID')
    return rfm_df

def generate_rfm_df(data_df: pd.DataFrame) -> pd.DataFrame:
    rfm_df = pd.DataFrame({'CustomerID': data_df.CustomerID.unique()}).set_index('CustomerID')
    print('Calculating recency data.')
    rfm_df = append_last_purchase_date_data(rfm_df, data_df)
    print('Calculating frequency data.')
    rfm_df = append_invoice_count_data(rfm_df, data_df)
    print('Calculating monetary data.')
    rfm_df = append_overall_purchase_amount_data(rfm_df, data_df)
    rfm_df.drop(rfm_df[rfm_df.CustomerOverallPurchaseAmount <= 0].index, inplace=True)
    assert len(rfm_df[rfm_df.DaysSinceLastPurchase < 0]) == 0
    assert len(rfm_df[rfm_df.InvoiceCount <= 0]) == 0
    assert len(rfm_df[rfm_df.CustomerOverallPurchaseAmount <= 0]) == 0
    rfm_df.rename(columns={'DaysSinceLastPurchase': 'recency', 'InvoiceCount': 'frequency', 'CustomerOverallPurchaseAmount': 'monetary'}, inplace=True)
    print('Normalizing RFM data.')
    rfm_df.recency = rfm_df.recency.progress_map(lambda number_of_days_since_last_purchase: 1+number_of_days_since_last_purchase)
    rfm_df = rfm_df.progress_apply(np.log10)
    assert len(rfm_df[rfm_df.isnull().any(axis=1)])==0, "RFM data contains NaN"
    return rfm_df

def generate_customer_similarity_labels_via_louvain(data_df: pd.DataFrame) -> pd.DataFrame:
    print('Creating bipartite customer-to-product graph.')
    bipartite_graph = nx.from_pandas_edgelist(data_df, 'CustomerID', 'StockCode')
    print('Creating customer similarity graph.')
    customer_similarity_graph = nx.projected_graph(bipartite_graph, data_df.CustomerID.unique())
    print('Determining Louvain communities.')
    customer_id_to_label = community_louvain.best_partition(customer_similarity_graph)
    custom_label_df = pd.DataFrame.from_dict(customer_id_to_label, orient='index', columns=['community_label'])
    return custom_label_df

##################
# Visualize Data #
##################

def visualize_rfm_via_pca(rfm_df: pd.DataFrame, customer_louvain_community_label_df: pd.DataFrame) -> None:
    print('Visualizing principal components.')
    rfm_np = rfm_df.to_numpy()
    scaler = StandardScaler(copy=False)
    scaler.fit(rfm_np)
    rfm_np = scaler.transform(rfm_np)
    pca = PCA(n_components=2, copy=False)
    pca.fit(rfm_np)
    rfm_np = pca.transform(rfm_np)
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        plot.set_title('RFM PCA Visualization with 2 Principal Components')
        plot.set_xlabel('PCA 1')
        plot.set_ylabel('PCA 2')
        plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
        plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
        plot.scatter(rfm_np[:,0], rfm_np[:,1], alpha=0.25)
        figure.savefig(RFM_PCA_VISUALIZATION_OUTPUT_PNG_FILE_LOCATION)
    cluster_label_to_silhouette_score = {}
    for cluster_count in tqdm_with_message(range(2, MAX_NUMBER_OF_CLUSTERS_TO_TRY+1), post_yield_message_func=lambda index: f'Visualizing {cluster_count} clusters'):
        kmeans = KMeans(init='k-means++', n_clusters=cluster_count, n_init=100)
        labels = kmeans.fit_predict(rfm_np)
        cluster_centers = kmeans.cluster_centers_
        with temp_plt_figure(figsize=(20.0,10.0)) as figure:
            plot = figure.add_subplot(111)
            plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
            plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
            cluster_label_to_color_map = matplotlib.cm.rainbow(np.linspace(0, 1, cluster_count))
            colors = np.array([cluster_label_to_color_map[label] for label in labels])
            plot.scatter(rfm_np[:,0], rfm_np[:,1], c=colors, alpha=0.25)
            silhouette_score_value = silhouette_score(rfm_np, labels)
            cluster_label_to_silhouette_score[cluster_count] = silhouette_score_value
            plot.set_title(f'RFM PCA Visualization with 2 Principal Components and {cluster_count} Clusters (Silhouette Score of {silhouette_score_value})')
            plot.set_xlabel('PCA 1')
            plot.set_ylabel('PCA 2')
            figure.savefig(RFM_PCA_VISUALIZATION_WITH_CLUSTERS_OUTPUT_PNG_FILE_LOCATION_TEMPLATE.format(cluster_count=cluster_count))
    print('Visualizing Silhouette scores.')
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        plot.set_title(f'Silhouette Scores')
        plot.set_xlabel('Cluster Count')
        plot.set_ylabel('Silhouette Score')
        cluster_counts, silhouette_scores = zip(*sorted(cluster_label_to_silhouette_score.items(), key=lambda pair: pair[0]))
        plot.plot(cluster_counts, silhouette_scores, '-')
        plot.set_xlim(left=0, right=11)
        silhouette_y_axis_upper_bound = math.ceil(max(cluster_label_to_silhouette_score.values()) * 1.1 * 100) / 100
        plot.set_ylim(bottom=0.0, top=silhouette_y_axis_upper_bound)
        y_tick_delta = 0.05
        plot.set_yticks(np.arange(0, silhouette_y_axis_upper_bound+y_tick_delta, step=y_tick_delta))
        plot.set_xticks(range(max(cluster_label_to_silhouette_score.keys())+2))
        plot.grid(True)
        figure.savefig(RFM_CLUSTER_SILHOUETTE_SCORE_OUPUT_PNG_FILE_LOCATION)
    print('Visualizing customer purchase-similarity communities.')
    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        plot = figure.add_subplot(111)
        plot.axvline(c='grey', lw=1, ls='--', alpha=0.5)
        plot.axhline(c='grey', lw=1, ls='--', alpha=0.5)
        number_of_communities = len(customer_louvain_community_label_df.community_label.unique())
        cluster_label_to_color_map = matplotlib.cm.rainbow(np.linspace(0, 1, number_of_communities))
        labels = (customer_louvain_community_label_df.loc[customer_id].community_label for customer_id in rfm_df.index)
        colors = np.array([cluster_label_to_color_map[label] for label in labels])
        plot.scatter(rfm_np[:,0], rfm_np[:,1], c=colors, alpha=0.25)
        plot.set_title(f'Customer Purchase-Similarity Communities via Louvain ({number_of_communities} Communities)')
        plot.set_xlabel('PCA 1')
        plot.set_ylabel('PCA 2')
        figure.savefig(CUSTOMER_SIMILARITY_LABEL_OUTPUT_PNG_FILE_LOCATION)
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    print('Importing & cleaning data.')
    data_df = pd.read_csv('./data/data.csv', encoding="ISO-8859-1")
    data_df = clean_data(data_df)
    print('Generating RFM data.')
    rfm_df = generate_rfm_df(data_df)
    print('Generating similar purchase graph.')
    customer_louvain_community_label_df = generate_customer_similarity_labels_via_louvain(data_df)
    print('Visualizing cluster data.')
    visualize_rfm_via_pca(rfm_df, customer_louvain_community_label_df)
    return
        
if __name__ == '__main__':
    main()
