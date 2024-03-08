''' Step 2: Create a target column using the user-fed target column name, column to categorize and the number of bins to get as even of a distribution as possible. 
NOTE: Using qcut for Quantile cut.
'''
import torch
torch.manual_seed(0)
import random
random.seed(0)

import pandas as pd
import os

from load_data import load_data

def create_class_label(nodes_path, edges_path, target_col_name, col_to_cat, num_of_bins):
    nodes_data, edges_data = load_data(nodes_path, edges_path)

    # Dropping redundant rows, if any
    nodes_data = nodes_data.loc[nodes_data['chromosome'] != 0]

    nodes_data[target_col_name] = pd.qcut(nodes_data[col_to_cat], num_of_bins, labels=list(range(1, num_of_bins+1)))
    print(f"Target class distribution: {nodes_data.groupby(nodes_data[target_col_name])[target_col_name].count()}")
    print(f'Ranges for classes: {nodes_data.groupby(target_col_name)[col_to_cat].agg(["min", "max"])}')

    return nodes_data, edges_data






