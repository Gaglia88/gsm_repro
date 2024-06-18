import pandas as pd
import pickle
import os
from progressive_utils import Utils

all_features = ["cfibf", "raccb", "js", "rs", "aejs", "nrs", "wjs"]
clean_datasets = []
dirty_datasets = []
datasets = Utils.load_datasets('../datasets/datasets.json')
num_matches = dict()
for d in datasets:
    num_matches[d["name"]] = len(pd.read_json(f"../{d['base_path']}/{d['gt']}", lines=True))
    if d["type"] == "dirty":
        dirty_datasets.append(d["name"])
    else:
        clean_datasets.append(d["name"])
        
def calc_auc(res, gt_size, num):
    noOfMatches = 0.0
    progressiveRecall = 0.0
    n = min(num, len(res))
    for r in res[0:n]:
        if r == 1:
            noOfMatches += 1
        progressiveRecall += noOfMatches
    auc = progressiveRecall / gt_size / (n + 1)
    return auc

def get_auc(res, gt_size, levels=[1, 5, 10, 20]):
    return [calc_auc(res, gt_size, gt_size * l) for l in levels]
    
def calc_dataset_auc(basepath, dataset, avg=True):
    global all_features, num_matches
    outdir = f"{basepath}/files/comparisons/{dataset}/"
    #df = pd.DataFrame(columns=["method", "auc_1", "auc_5", "auc_10", "auc_20"])
    res = []
    for i in range(0, 5):
        if os.path.isfile(f'{outdir}comp_sup_mb_run_{i}.pickle'):
            f = open(f'{outdir}comp_sup_mb_run_{i}.pickle', 'rb')
            sup_mb = pickle.load(f)
            f.close()

            auc = get_auc(sup_mb, num_matches[dataset]-25)

            #serie = pd.Series((["sup_mb"]+auc), index = df.columns)
            res.append((["sup_mb"]+auc))
            #print(serie)ì
            #df = df.append(serie, ignore_index=True)
            #df = pd.concat([df, serie.transpose()], ignore_index=True)

            #display(df)

            for fn in all_features:
                if os.path.isfile(f'{outdir}comp_{fn}_run_{i}.pickle'):
                    f = open(f'{outdir}comp_{fn}_run_{i}.pickle', 'rb')
                    data = pickle.load(f)
                    f.close()

                    auc = get_auc(data, num_matches[dataset]-25)

                    res.append(([fn]+auc))
                    
                    #serie = pd.Series(([fn]+auc), index = df.columns)
                    #df = df.append(serie, ignore_index=True)
                    #df = pd.concat([df, serie], ignore_index=True)
    df = pd.DataFrame(res, columns=["method", "auc_1", "auc_5", "auc_10", "auc_20"])
    if avg:
        res = df.groupby('method').mean().reset_index()
    else:
        res = df
    res['dataset'] = dataset
    return res
    
def load_data(basepath, datasets, avg=True):
    df = pd.DataFrame(columns=["method", "dataset", "auc_1", "auc_5", "auc_10", "auc_20"])

    for d in datasets:
        df1 = calc_dataset_auc(basepath, d, avg)
        #df = df.append(df1)
        df = pd.concat([df, df1], ignore_index=True)
    return df
    
    
basepath = "/home/app/progressive/"
df_clean = load_data(basepath, clean_datasets)

df_clean.to_csv('/home/app/results/progressive_clean_results.csv', sep=";", index=False)

df_dirty = load_data(basepath, dirty_datasets)
df_dirty.to_csv('/home/app/results/progressive_dirty_results.csv', sep=";", index=False)