import os
import pandas as pd

def concat_data(filepath):
    if os.path.isfile(filepath):
        print(f"Reading: {filepath}")
        df = pd.read_parquet(filepath)
    else:
        for root, dirs, files in os.walk(filepath):
            print(f"Reading: {os.path.join(root, files[0])}")
            df = pd.read_parquet(os.path.join(root, files[0]))
            for f in files[1:]:
                print(f"Reading: {os.path.join(root, f)}")
                df_temp = pd.read_parquet(os.path.join(root, f))
                df = pd.concat([df, df_temp], axis=0)
    return df