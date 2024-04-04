import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import json
from matplotlib.collections import PatchCollection


#Intra-Network Connectivity
def calculate_intra_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
    """
    Intra-Network Connectivity

    Inputs:
    - orginal single-subject timeseries
    - new timeseries (discarding the first four columns of orig_timeseries and adding a new column with the mapped values of network labels)
    - path to the network names
    - network names
    
    Parameters:
    - Key of network numbers mapped to network names
    - For each network, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
    - Calculates the correlation matrix for the selected data using network_data.transpose().corr() then gets the average correlations

    Returns:
    - Intra-network connectivity average correlation matrix
    """
    intra_network_connectivity = {}
    ordered_networks = network_names.keys()
    for network in ordered_networks:
        if network != 0:
            network_data = original_timeseries[new_timeseries['Yeo_7network'] == network]
            correlation_matrix = network_data.transpose().corr()
            average_corr = np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)])
            intra_network_connectivity[network] = average_corr
    return intra_network_connectivity


#Inter-Network Connectivity
def calculate_inter_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
    """
    Inter-Network Connectivity
    
    Inputs:
    - orginal single-subject timeseries
    - new timeseries
    - path to the network names
    - network names

    Parameters:
    - Key of network numbers mapped to network names (ordered_networks)
    - Iterates over pairs of networks (network_i, network_j)
    - For each network pair, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
    - Calculates the correlation between each pair of rows (one from each network) and then gets the average correlations

    Returns:
    - Inter-network connectivity average correlation matrix
    """
    inter_network_connectivity = {}
    ordered_networks = network_names.keys()
    for i, network_i in enumerate(ordered_networks):
        for j, network_j in enumerate(ordered_networks):
            if network_i != network_j and network_i != 0 and network_j != 0:
                data_i = original_timeseries[new_timeseries['Yeo_7network'] == network_i]
                data_j = original_timeseries[new_timeseries['Yeo_7network'] == network_j]
                correlation_values = []
                for index_i, row_i in data_i.iterrows():
                    for index_j, row_j in data_j.iterrows():
                        correlation_values.append(row_i.corr(row_j))
                average_corr = np.mean(correlation_values)
                key = (network_i, network_j)
                inter_network_connectivity[key] = average_corr
    return inter_network_connectivity

# #Inter-Network Connectivity Counter 
# def count_inter_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
#     """
#     Inter-Network Connectivity
    
#     Inputs:
#     - orginal single-subject timeseries
#     - new timeseries
#     - path to the network names
#     - network names

#     Parameters:
#     - Key of network numbers mapped to network names (ordered_networks)
#     - Iterates over pairs of networks (network_i, network_j)
#     - For each network pair, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
#     - Calculates the correlation between each pair of rows (one from each network) and then gets the average correlations

#     Returns:
#     - Inter-network connectivity average correlation matrix
#     """
#     inter_network_connectivity = {}
#     inter_network_counter = {}  # Counter for the number of correlation values
#     ordered_networks = network_names.keys()
#     for i, network_i in enumerate(ordered_networks):
#         for j, network_j in enumerate(ordered_networks):
#             if network_i != network_j and network_i != 0 and network_j != 0:
#                 data_i = original_timeseries[new_timeseries['Yeo_7network'] == network_i]
#                 data_j = original_timeseries[new_timeseries['Yeo_7network'] == network_j]
#                 correlation_values = []

#                 for index_i, row_i in data_i.iterrows():
#                     for index_j, row_j in data_j.iterrows():
#                         correlation_values.append(row_i.corr(row_j))

#                 for index_i, row_i in data_i.iterrows():
#                     for index_j, row_j in data_j.iterrows():
#                         correlation_values.append(row_i.corr(row_j))
#                 average_corr = np.mean(correlation_values)
#                 key = (network_i, network_j)
#                 inter_network_connectivity[key] = average_corr
#                 # Update the counter
#                 if key in inter_network_counter:
#                     inter_network_counter[key] += len(correlation_values)
#                 else:
#                     inter_network_counter[key] = len(correlation_values)
#     return inter_network_counter

# connectivity_count = count_inter_network_connectivity()


#Thresholded Connectivity
def threshold_dataframe(df, threshold):
    """
    Takes a DataFrame and replaces all numeric values below a given threshold with 0.
    Non-numeric values are left as is.

    Parameters:
    df (pd.DataFrame): DataFrame to apply thresholding to.
    threshold (float): Threshold value to determine whether to keep a cell's numeric value or set it to 0.

    Returns:
    pd.DataFrame: New DataFrame with thresholding applied.
    """
    # Apply a function to the entire DataFrame to threshold values
    # This function will ignore non-numeric values
    thresholded_df = df.map(lambda x: x if isinstance(x, (int, float)) and x >= threshold else 0 if isinstance(x, (int, float)) else x)
    
    return thresholded_df

def generate_edges_and_weights(df, excluded_columns):
    regions = [col for col in df.columns if col not in excluded_columns]
    edges_with_weights = {}
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            weight = df.at[i, region2]
            edges_with_weights[(region1, region2)] = weight
    return edges_with_weights

def generate_edges_and_weights_with_network(df, excluded_columns, network_name):
    regions = [col for col in df.columns if col not in excluded_columns]
    region_to_network = pd.Series(df[network_name].values, index=df.region).to_dict()
    edges_with_weights = {}
    for i, region1 in enumerate(regions):
        for region2 in regions:
            network1 = region_to_network.get(region1)
            network2 = region_to_network.get(region2)
            weight = df.at[i, region2]
            edges_with_weights[(network1, network2)] = weight
    return edges_with_weights

def count_network_pairs_above_threshold(edges_weights_with_networks, threshold):
    network_pair_counts = {}
    for network_pair, weight in edges_weights_with_networks.items():
        if weight >= threshold:
            if network_pair not in network_pair_counts:
                network_pair_counts[network_pair] = 0
            network_pair_counts[network_pair] += 1
    return network_pair_counts