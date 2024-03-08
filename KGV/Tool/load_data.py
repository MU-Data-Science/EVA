''' Step 1: Read input files which can be either .csv or .parquet. Return dataframes for edges and nodes. 
NOTE: In subsequent code, call load_data instead of concat_data.
'''

import os
import pandas as pd

def concat_data(filepath):
    # If reading only a single file
    if os.path.isfile(filepath):
        print(f"Reading: {filepath}")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
    else:
        # If reading a folder
        for root, dirs, files in os.walk(filepath):
            print(f"Reading: {os.path.join(root, files[0])}")
            if files[0].endswith('.csv'):
                df = pd.read_csv(os.path.join(root, files[0]))
            elif files[0].endswith('.parquet'):
                df = pd.read_parquet(os.path.join(root, files[0]))
            for f in files[1:]:
                print(f"Reading: {os.path.join(root, f)}")
                if f.endswith('.csv'):
                    df_temp = pd.read_csv(os.path.join(root, f))
                elif f.endswith('.parquet'):
                    df_temp = pd.read_parquet(os.path.join(root, f))
                df = pd.concat([df, df_temp], axis=0)
    return df

def load_data(nodes_path, edges_path):
    print("--> Processing node files... ")
    nodes_data = concat_data(nodes_path)
    print("--> Processing edge files... ")
    edges_data = concat_data(edges_path)
    
    return nodes_data, edges_data