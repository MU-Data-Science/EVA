import pandas as pd

from concat_data import concat_data

# If the train, val and test sets have already been saved into parquet files, use this function
def read_from_existing_files(rejected_nodes_data_path, train_set_path, val_set_path, test_set_path, edge_path):
    rejected_nodes_data = pd.read_parquet(rejected_nodes_data_path)
    rejected_nodes_data.drop('_merge', axis=1, inplace=True)
    train_set = pd.read_parquet(train_set_path)
    val_set = pd.read_parquet(val_set_path)
    test_set = pd.read_parquet(test_set_path)

    total_selected_len = len(train_set) + len(val_set) + len(test_set)

    edges_data = concat_data(edge_path)
    print(f"length of edges_data: {len(edges_data)}")

    nodes_data = pd.concat([train_set, val_set, test_set, rejected_nodes_data])

    total_selected_len = len(train_set) + len(val_set) + len(test_set)

    return nodes_data, edges_data, total_selected_len

# If reading data from scratch, use this function
def read_from_raw_data(node_path, edge_path):
    nodes_data = concat_data(node_path)
    edges_data = concat_data(edge_path)

    nodes_data = nodes_data.loc[nodes_data['chromosome'] != 0]

    min_class_samples = nodes_data.groupby('ann_impact')['ann_impact'].count().min()
    class_0 = nodes_data.loc[nodes_data['ann_impact'] == 7]
    selected_class_0 = class_0[:min_class_samples]
    rejected_class_0 = class_0[min_class_samples:]

    class_1 = nodes_data.loc[nodes_data['ann_impact'] == 13]
    selected_class_1 = class_1[:min_class_samples]
    rejected_class_1 = class_1[min_class_samples:]

    class_2 = nodes_data.loc[nodes_data['ann_impact'] == 20]
    selected_class_2 = class_2[:min_class_samples]
    rejected_class_2 = class_2[min_class_samples:]

    class_3 = nodes_data.loc[nodes_data['ann_impact'] == 141]
    selected_class_3 = class_3[:min_class_samples]
    rejected_class_3 = class_3[min_class_samples:]

    class_4 = nodes_data.loc[nodes_data['ann_impact'] == 0]
    selected_class_4 = class_4[:min_class_samples]
    rejected_class_4 = class_4[min_class_samples:]

    shuffled_selected_classes = pd.concat([selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4])

    nodes_data = pd.concat([shuffled_selected_classes.sample(frac=1), rejected_class_0, rejected_class_1, rejected_class_2, rejected_class_3, rejected_class_4])

    print("Writing train, val and test sets to files...")
    total_selected_len = len(selected_class_0) + len(selected_class_1) + len(selected_class_2) + len(selected_class_3) + len(selected_class_4)
    n_train = int(total_selected_len * 0.6)
    n_val = int(total_selected_len * 0.2)

    nodes_data[:n_train].to_parquet(f'/mydata/dgl/general/effect_train_set.parquet')
    nodes_data[n_train:n_train+n_val].to_parquet(f'/mydata/dgl/general/effect_val_set.parquet')
    nodes_data[n_train+n_val:total_selected_len].to_parquet(f'/mydata/dgl/general/effect_test_set.parquet')

    return nodes_data, edges_data, total_selected_len