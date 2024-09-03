from utils import Utils
import os
import pandas as pd
from configparser import ConfigParser
import sys

def check_features():
    datasets = Utils.load_datasets()
    for d in datasets:
        if not os.path.isdir(f"/home/app/features/{d['name']}"):
            raise Exception(f"[GSM REPRO] Missing features for the dataset {d['name']}")
            return False
    return True

def check_dirty_probabilities():
    datasets = Utils.load_datasets(dtype='dirty')

    repetitions = 3
    # Training set sizes
    training_set_sizes = [50, 500]
    # Top feature_sets
    feat_to_use = [78, 128, 187]
    
    feat_to_use_tsize = {}
    feat_to_use_tsize[50] = [78, 187]
    feat_to_use_tsize[500] = [128]
    
    features_sets = pd.read_csv('/home/app/config/feature_sets.csv', sep=";")
    features_sets = features_sets[features_sets['conf_id'].isin(feat_to_use)]


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
                            log.write(f"[GSM REPRO] Missing probability: probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet\n")
                            return False
    return True
            
def check_clean_probabilities():
    # Read the default configuration
    config = ConfigParser()
    config.read('/home/app/config/config.ini')

    # Number of repetitions for each feature (change the training set)
    repetitions = config.getint('gen_all_probabilities', 'repetitions')
    # Training set size
    train_set_size = config.getint('gen_all_probabilities', 'train_set_size')
    
    # Load the feature sets
    features_sets = pd.read_csv('/home/app/config/feature_sets.csv', sep=";")
    
    # Load the datasets
    datasets = Utils.load_datasets(dtype='clean')


    # For each dataset
    for d in datasets:
        # For each feature set
        for index, feature_set in features_sets.iterrows():
            features_set_id = feature_set['conf_id']
            # for each repetition
            for r in range(0, repetitions):
                # Check the existence of the probabilities
                if not os.path.isfile(f"/home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet"):
                    raise Exception(f"[GSM REPRO] Missing probability: /home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet\n")
                    return False
    return True

def check_train_size_probabilities():
    # Read the default configuration
    config = ConfigParser()
    config.read('/home/app/config/config.ini')

    # Number of repetitions for each feature (change the training set)
    repetitions = config.getint('gen_all_probabilities', 'repetitions')
    # Training set sizes
    training_set_sizes = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    # Top feature_sets
    feat_to_use = [78, 128, 187]
    
    # Load the feature sets
    features_sets = pd.read_csv('/home/app/config/feature_sets.csv', sep=";")
    features_sets = features_sets[features_sets['conf_id'].isin(feat_to_use)]
    
    # Load the datasets
    datasets = Utils.load_datasets(dtype='clean')

    # For each dataset
    for d in datasets:
        # For each training set size
        for train_set_size in training_set_sizes:
            # For each feature set
            for index, feature_set in features_sets.iterrows():
                features_set_id = feature_set['conf_id']
                # for each repetition
                for r in range(0, repetitions):
                    # Check the existence of the probabilities
                    if not os.path.isfile(f"/home/app/probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet"):
                        raise Exception(f"[GSM REPRO] Missing probability: probabilities/{d['name']}/{train_set_size}/{d['name']}_fs{features_set_id}_run{r}.parquet\n")
                        return False
    return True

def check_file_exists(path):
    if not os.path.isfile(path):
        raise Exception(f'[GSM REPRO] Missing file: {path}')
        return False
    return True

def check_calc_cbs():
    # Load the datasets
    datasets = Utils.load_datasets(dtype='clean')
    for d in datasets:
        fname = f"/home/app/cbs_stats/{d['name']}.csv"
        if not os.path.isfile(fname):
            raise Exception(f"[GSM REPRO] Missing file: {fname}")
            return False
    return True

def check_progressive_01():
    base_path = "/home/app/progressive/files"
    datasets = Utils.load_datasets()
    for d in datasets:
        dataset = d["name"]
        if  os.path.isfile(f'{base_path}/{dataset}/profile_index_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/block_index_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/separator_ids_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/new_gt_{dataset}.pickle') and os.path.isfile(f'{base_path}/{dataset}/features.parquet') and os.path.isfile(f'{base_path}/{dataset}/weights.parquet'):
            continue;
        else:
            raise Exception(f"[GSM REPRO] Missing files for progressive experiments (step 01) for dataset {dataset}")
            return False
    return True

def check_progressive_02():
    base_path = "/home/app/progressive/files"
    all_features=["cfibf", "raccb", "js", "rs", "aejs", "nrs", "wjs"]
    datasets = Utils.load_datasets()
    for d in datasets:
        dataset = d["name"]
        outdir = f"{base_path}/comparisons/{dataset}/"
        
        all_done = True
        for i in range(0, 5):
            if not os.path.isfile(f'{outdir}comp_sup_mb_run_{i}.pickle'):
                all_done = False
                break
            
            for f in all_features:
                if not os.path.isfile(f'{outdir}comp_{f}_run_{i}.pickle'):
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            raise Exception(f"[GSM REPRO] Missing files for progressive experiments (step 02) for dataset {dataset}")
            return False
    return True

if __name__ == "__main__":
    exp = int(sys.argv[1])
    
    res = False
    
    if exp==1:
        res = check_features()
    elif exp==2:
        res = check_file_exists('/home/app/results/features_calc_time.csv')
    elif exp==3:
        res = check_clean_probabilities()
    elif exp==4:
        res = check_train_size_probabilities()
    elif exp==5:
        res = check_dirty_probabilities()
    elif exp==6:
        res = check_file_exists('/home/app/results/algorithm_selection_java.csv')
    elif exp==7:
        res = check_file_exists('/home/app/results/feature_selection_java.csv')
    elif exp==8:
        res = check_file_exists('/home/app/results/train_size_selection_java.csv')
    elif exp==9:
        res = check_file_exists('/home/app/results/table4.csv')
    elif exp==10:
        res = check_file_exists('/home/app/results/scalability.csv')
    elif exp==11:
        res = check_calc_cbs()
    elif exp==12:
        res = check_file_exists('/home/app/results/blast_thresholds.csv')
    elif exp==13:
        res = check_progressive_01()
    elif exp==14:
        res = check_progressive_02()
    elif exp==15:
        res1 = check_file_exists('/home/app/results/progressive_clean_results.csv')
        res2 = check_file_exists('/home/app/results/progressive_dirty_results.csv')
        res = res1 and res2
        
    if res:
        out = open("/home/app/progress.txt", "wt")
        out.write(str(exp))
        out.close()