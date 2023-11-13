# Purpose: Map node features to unique integers
# Execution: python3 mapping_node_features.py

import pandas as pd
import math

df = pd.read_csv('/Users/shivikaprasanna/Desktop/node_features.csv')
df_to_map = df.loc[ : , df.columns != 'src_id']
val_dict = {}
label_dict = {}

for k in df_to_map.keys():
    val_dict = {value:key for key , value in enumerate(df_to_map[k].unique())}
    df_to_map[k] = df_to_map[k].map(val_dict)

for idx, row in df_to_map.iterrows():
    if row['variantId'] != 0:
        df_to_map.loc[idx, 'Label'] = 1
    else:
        df_to_map.loc[idx, 'Label'] = 0

src_id = df['src_id']
df_to_map = df_to_map.join(src_id)
df_to_map.to_csv('/Users/shivikaprasanna/Desktop/node_features_mapped.csv', index=None)
