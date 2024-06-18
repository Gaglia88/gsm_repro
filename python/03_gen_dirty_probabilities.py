from utils import Utils
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Process, Manager
from configparser import ConfigParser

def run_process(d, r, out, features_set_id, X_train, X_test, y_train, y_test, feature_set, train_set_size, return_dict, log_dict):
    """
    Generate the probabilities for a specific dataset/run/feature set.
    d: dataset data
    r: run number
    features_set_id: id of the current feature set
    features: dataframe of features
    feature_set: features to use
    train_set_size: size of the training set
    return_dict: returning data
    """
    try:
        print(f"{d['name']} - {features_set_id} - {r}")
        t1 = time.time()
        # Train the estimator
        est = SVC(probability=True, random_state=42)
        #est = LogisticRegression(random_state=42)
        trained = est.fit(X_train[feature_set], np.ravel(y_train))
        #Output edges
        edges = X_test[['p1', 'p2']].copy()
        # Generate the matching probabilities for each edge
        probabilities = trained.predict_proba(X_test[feature_set])
        edges['p_match'] = probabilities[:,1]
        # Generate the prediction for each edge
        edges['pred'] = trained.predict(X_test[feature_set])
        edges['is_match'] = y_test
        t2 = time.time()
        return_dict[f"{d['name']}{train_set_size}{features_set_id}{r}"] = f"{d['name']},{features_set_id},{r},{t2-t1}\n"
        edges.to_parquet(f"/home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet", index=False)
    except Exception as e:
        log_dict[f"{d['name']}{train_set_size}{features_set_id}{r}"] = repr(e)

"""
Generate all the probabilities for each dataset.
"""
if __name__ == "__main__":
    # Read the default configuration
    config = ConfigParser()
    config.read('/home/app/config/config.ini')

    #Number of parallel process that compute the features
    num_parallel_proc = config.getint('gen_all_probabilities', 'parallel_process')
    # Number of repetitions for each feature (change the training set)
    repetitions = 3
    # Training set sizes
    training_set_sizes = [50, 500]
    # Top feature_sets
    feat_to_use = [78, 128, 187]
    
    feat_to_use_tsize = {}
    feat_to_use_tsize[50] = [78, 187]
    feat_to_use_tsize[500] = [128]
    
    # Load the feature sets
    features_sets = pd.read_csv('/home/app/config/feature_sets.csv', sep=";")
    features_sets = features_sets[features_sets['conf_id'].isin(feat_to_use)]
    
    # Load the datasets
    datasets = Utils.load_datasets(dtype='dirty')
    
    # Logging
    log = open('/home/app/logs/errors_gen_dirty_probabilities.txt', 'wt')
    
    # Write the runtime for each dataset/feature set
    out = open('/home/app/results/dirty_feature_set_rt.csv', 'at')
    out.write("dataset,features_set_id,run,time\n")
    
    # Create the default folder
    if not os.path.isdir('/home/app/probabilities'):
        os.mkdir('/home/app/probabilities')
        
        
    # Create a pool of processes, one process for each dataset-featureset-iteration
    manager = Manager()
    # Resulting values from the process
    return_dict = manager.dict()
    log_dict = manager.dict()
    
    # For each dataset
    for d in datasets:
        print(f"Read features for {d['name']}")
        # Load the features
        features = pd.read_parquet(f"/home/app/features/{d['name']}")
        
        # For each training set size
        for train_set_size in training_set_sizes:
            # Build the folders
            if not os.path.isdir(f"/home/app/probabilities/{d['name']}"):
                os.mkdir(f"/home/app/probabilities/{d['name']}")
            if not os.path.isdir(f"/home/app/probabilities/{d['name']}/{train_set_size}"):
                os.mkdir(f"/home/app/probabilities/{d['name']}/{train_set_size}")
            
            # For each repetition
            for r in range(0, repetitions):
                print(f"Repetition {r}")
                # Generate the training set
                X_train, X_test, y_train, y_test = Utils.get_train_test(features, train_set_size, "is_match", r)
            
                # For each feature set
                for index, feature_set in features_sets.iterrows():
                    features_set_id = feature_set['conf_id']
                    if features_set_id in feat_to_use_tsize[train_set_size]:
                        features_set = list(map(lambda x: x.strip(), feature_set['conf'].split(',')))
                        if not os.path.isfile(f"/home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet"):
                            p = Process(target=run_process, args=(d, r, out, features_set_id, X_train, X_test, y_train, y_test, features_set, train_set_size, return_dict, log_dict,))
                            p.start()
                            p.join()
                            # Write the returning values
                            for k in list(return_dict.keys()):
                                value = return_dict.pop(k, "")
                                out.write(value)
                            for k in list(log_dict.keys()):
                                value = log_dict.pop(k, "")
                                if len(value) > 0:
                                    log.write(f"{k} - {value}\n")
                            out.flush()
                            log.flush()

    """
    # Run the processes, num_parallel_proc at a time
    while len(processes) > 0:
        run_processes = []
        # Launch num_parallel_proc
        while len(run_processes) < num_parallel_proc and len(processes) > 0:
            p = processes.pop()
            run_processes.append(p)
            p.start()
            
        # Wait the execution of all processes
        for p in run_processes:
            p.join()
    
        # Write the returning values
        for k in list(return_dict.keys()):
            value = return_dict.pop(k, "")
            out.write(value)
        for k in list(log_dict.keys()):
            value = log_dict.pop(k, "")
            if len(value) > 0:
                log.write(f"{k} - {value}\n")
        out.flush()
        log.flush()
    """            
    out.close()
    
    # Final check
    
    # For each dataset
    for d in datasets:
        # For each training set size
        for train_set_size in training_set_sizes:
            # For each feature set
            for index, feature_set in features_sets.iterrows():
                features_set_id = feature_set['conf_id']
                if features_set_id in feat_to_use_tsize[train_set_size]:
                    # for each repetition
                    for r in range(0, repetitions):
                        # Check the existence of the probabilities
                        if not os.path.isfile(f"/home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet"):
                            log.write(f"NOT FOUND: probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet\n")
    log.close()
                