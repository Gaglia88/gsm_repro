import pandas as pd
import numpy as np
import sparker
from sklearn.svm import SVC
from sparker.common_node_pruning import compute_statistics, calc_cbs, calc_weights, do_reset, WeightTypes
from sparker.pruning_utils import PruningUtils
from sklearn.linear_model import LogisticRegression
import random
import pickle
import os
from progressive_utils import Utils
import time


def blocking(d):    
    tstart = time.time()
    profiles1, profiles2, profiles, new_gt, max_profile_id, separator_ids = Utils.load_data(d, base_folder="/home/app/")
    num_profiles = profiles.count()
    purging_threshold = 1.0
    if "purging_threshold" in d:
        purging_threshold = float(d["purging_threshold"])
    profile_blocks, profile_blocks_filtered, blocks_after_filtering = Utils.blocking_cleaning(profiles, separator_ids, pf=purging_threshold)
    return profiles, profile_blocks_filtered, blocks_after_filtering, new_gt, separator_ids

def make_process(base_path, d, feature_set=["cfibf", "raccb", "rs", "nrs"], budget=50):
    dataset = d["name"]
    profiles, profile_blocks_filtered, blocks_after_filtering, new_gt, separator_ids = blocking(d)
    # Generates the features
    features = sparker.FeatureGenerator.generate_features(profiles, blocks_after_filtering, separator_ids, new_gt,
                                                          False)
                                                          
    # Converts the features into a pandas dataframe
    pd_features1 = features.toPandas()
    pd_features = pd_features1[(["p1", "p2", "is_match"] + feature_set)]

    # Computes the probability of being a match for each pair of records
    fold = 1
    X_train, X_test, y_train, y_test = Utils.get_train_test(pd_features[["is_match"] + feature_set], budget, "is_match", fold)
    est = LogisticRegression(random_state=42)  # SVC(probability=True, random_state=42)
    trained = est.fit(X_train, np.ravel(y_train))

    probabilities = pd.DataFrame(trained.predict_proba(X_test), columns=trained.classes_) \
        .rename({0: "0", 1: "p_match"}, axis=1)['p_match']

    weights = pd.concat(
        [pd_features.loc[X_test.index][['p1', 'p2', 'is_match']].copy().reset_index().drop("index", axis=1),
         probabilities], axis=1)

    # Fetures without those used for training
    features_no_train = pd_features1.loc[pd_features1.index.difference(X_train.index)]
    

    # Index that maps every profile with the blocks in which it is contained
    profile_index = profile_blocks_filtered.map(
        lambda x: (x.profile_id, list(map(lambda y: y.block_id, x.blocks)))).collectAsMap()

    # Index that maps every block with the profiles that contains
    block_index = blocks_after_filtering.map(lambda b: (b.block_id, b.profiles)).collectAsMap()
    
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    
    if not os.path.isdir(f'{base_path}/{dataset}'):
        os.mkdir(f'{base_path}/{dataset}')
    
    f = open(f'{base_path}/{dataset}/profile_index_{dataset}.pickle', 'wb')
    pickle.dump(profile_index, f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/block_index_{dataset}.pickle', 'wb')
    pickle.dump(block_index, f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/separator_ids_{dataset}.pickle', 'wb')
    pickle.dump(separator_ids, f)
    f.close()
    
    f = open(f'{base_path}/{dataset}/new_gt_{dataset}.pickle', 'wb')
    pickle.dump(new_gt, f)
    f.close()
    
    features_no_train.to_parquet(f'{base_path}/{dataset}/features.parquet')
    weights.to_parquet(f'{base_path}/{dataset}/weights.parquet')
    
    return ""
    
if __name__ == '__main__':
    clean_datasets = Utils.load_datasets(dtype='clean')
    base_path = "/home/app/progressive/files"
    
    for d in clean_datasets:
        if not os.path.isdir(f'files/{d["name"]}'):
            make_process(base_path, d)
    
    
    dirty_datasets = Utils.load_datasets(dtype='dirty')
    for d in dirty_datasets:
        print(d)
        if not os.path.isdir(f'files/{d["name"]}'):
            make_process(base_path, d)

    
