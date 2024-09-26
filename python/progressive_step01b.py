import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import random
import pickle
import os
from progressive_utils import Utils
import time
import shutil

def make_process(base_path, d, feature_set=["cfibf", "raccb", "rs", "nrs"], budget=50):
    dataset = d["name"]
    
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    
    if not os.path.isdir(f'{base_path}/{dataset}'):
        os.mkdir(f'{base_path}/{dataset}')
    
    if  os.path.isfile(f'{base_path}/{dataset}/profile_index_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/block_index_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/separator_ids_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/new_gt_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/features.parquet') and os.path.isfile(f'{base_path}/{dataset}/weights.parquet'):
        print(f"[GSM REPRO] Progressive step 1b - All files for {dataset} are done")
    else:    
        # Converts the features into a pandas dataframe
        pd_features1 = pd.read_parquet(f'{base_path}/{dataset}/features_tmp')
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
        
        features_no_train.to_parquet(f'{base_path}/{dataset}/features.parquet')
        weights.to_parquet(f'{base_path}/{dataset}/weights.parquet')
        shutil.rmtree(f'{base_path}/{dataset}/features_tmp')
    
    return ""
    
if __name__ == '__main__':
    clean_datasets = Utils.load_datasets(dtype='clean')
    base_path = "/home/app/progressive/files"
    
    for d in clean_datasets:
        #if not os.path.isdir(f'files/{d["name"]}'):
        make_process(base_path, d)
    
    
    dirty_datasets = Utils.load_datasets(dtype='dirty')
    for d in dirty_datasets:
        print(d)
        #if not os.path.isdir(f'files/{d["name"]}'):
        make_process(base_path, d)

    
