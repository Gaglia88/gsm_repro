"""
A Graph-Based Blocking Approach for Entity Matching Using Contrastively Learned Embeddings 
Authors: John Bosco Mugeni & Toshiyuki Amagasa 
Institute: University of Tsukuba (ibaraki, Japan).

Published: ACM SIGAPP Applied Computing Review (Feb 2023)
"""


import argparse
import time
#from preprocessing_datasets import load_dataset
from evaluation import calc_index
from graph_clustering.knn_graph_clusteriser import all_in_one_clusteriser
#from vector_models import model
from transformers import AutoTokenizer
from simcse import SimCSE
import pandas as pd
import json
import os

def load_datasets(path='/home/app/datasets/datasets.json', dtype=''):
    f = open(path)
    datasets = json.load(f)
    f.close()
    if len(dtype) > 0:
        datasets = list(filter(lambda d: d['type']==dtype, datasets))
    return datasets

def convert_dataset(d1, d2, gt):
    d1 = d1.copy()
    d2 = d2.copy()
    gt = gt.copy()
    d1 = d1.reset_index().rename({'index': 'ltable_id'}, axis=1)
    d2 = d2.reset_index().rename({'index': 'rtable_id'}, axis=1)
    tmp = gt.merge(d1, left_on='id1', right_on='realProfileID')[['ltable_id', 'id2']]
    new_gt = tmp.merge(d2, left_on='id2', right_on='realProfileID')[['ltable_id', 'rtable_id']]
    
    d1 = d1.drop('realProfileID', axis=1)
    d2 = d2.drop('realProfileID', axis=1)
    d1 = d1.rename({'ltable_id': 'id'}, axis=1)
    d2 = d2.rename({'rtable_id': 'id'}, axis=1)
    return d1, d2, new_gt

def flatten(row):
    return " ".join(map(lambda x: str(x), row.values)).strip().lower()

def get_repr(df):
    return df.apply(flatten, axis=1)

def get_text_repr(df):
    df1 = df[['id']].copy()
    df1['text'] = get_repr(df.drop('id', axis=1))
    return df1

def fillnan(df):
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna('unk')
    return df

def load_data(dataset):
    base_path = "/home/app/"+dataset["base_path"]+"/"
    d1 = fillnan(pd.read_json(f"{base_path}dataset1.json", lines=True).rename({'id':'id12345'}, axis=1))
    d2 = fillnan(pd.read_json(f"{base_path}dataset2.json", lines=True).rename({'id':'id12345'}, axis=1))
    gt = pd.read_json(f"{base_path}groundtruth.json", lines=True)

    # Convert dataset and groundtruth in the form ltable_id, rtable_id
    table1, table2, table_match = convert_dataset(d1, d2, gt)

    table1 = get_text_repr(table1).rename({'id':'ltable_id'}, axis=1)
    table2 = get_text_repr(table2).rename({'id':'rtable_id'}, axis=1)

    table3 = pd.concat([table1, table2], ignore_index=True)

    pairs = set()
    for pair in zip(table_match['ltable_id'], table_match['rtable_id']):
            ordered_pair = tuple(sorted((table3.loc[table3['ltable_id'] == pair[0]].index[0], table3.loc[table3['rtable_id'] == pair[1]].index[0])))
            pairs.add(ordered_pair)

    table3.drop(labels=['ltable_id'], axis=1, inplace=True)
    table3.drop(labels=['rtable_id'], axis=1, inplace=True)

    return dataset['name'], table3, pairs

parser = argparse.ArgumentParser(description='Graph-based Block Clustering')

parser.add_argument("--verbose", type=int, default='0',choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str,default='walmart-amazon-clean', help='dataset')
parser.add_argument("--blocker", type=str,default='louvian', help='clustering algorithm')
parser.add_argument("--num_clusters", type=int,default='30', help='used in knn graph')
parser.add_argument("--attributes", default=["text"], nargs='+')
parser.add_argument("--model_name", type=str, default='princeton-nlp/sup-simcse-bert-base-uncased',help="model_path or name")
parser.add_argument("--upper_limit", type=int, default='30', help='upper limit for n_neighbors' )
parser.add_argument("--lower_limit", type=int, default='2', help='lower limit for n_neighbors' )
parser.add_argument("--step", type=int, default='1', help='step size' )
parser.add_argument("--max_seq_length", type=int, default=256, help='models sequence length')

hp, _ = parser.parse_known_args()

key_values = {
    'dataset': hp.dataset,
    'cluster_method': hp.blocker,
    'verbose': hp.verbose,
    'num_clusters': hp.num_clusters,
    'attributes_list': hp.attributes,
    'model_name': hp.model_name,
}

######################################
########### Blocking program ##########
######################################

datasets = load_datasets(dtype='clean')

if not os.path.isdir('/home/app/results/'):
    os.makedirs('/home/app/results/', exist_ok=True)
        
out = open('/home/app/results/contextual_blocker.csv', 'wt')
out.write("dataset;recall;precision;f1;ov_time\n")
for d in datasets:
    print(f"Processing {d['name']}")
    # 1) LOAD and PREPROCESS the dataset
    dataset_name, table, pairs = load_data(d)

    # 2) DO the embedding
    stime = time.time()
    
    attributes = hp.attributes # get attr names
    records = table[attributes].agg(' '.join, axis=1)
    simCSE = SimCSE(hp.model_name) # pre-trained model 
    vectors = simCSE.encode(list(records), device="cuda", max_length=hp.max_seq_length)
    num_clusters = 30
    data = all_in_one_clusteriser(vectors, hp.blocker, num_clusters)
    reduction_ratio, pair_completeness, reference_metric, pair_quality, fmeasure = calc_index(data, table, pairs)
    etime = time.time()
    
    out.write(f"{d['name']};{pair_completeness};{pair_quality};{fmeasure};{etime-stime}\n")
    out.flush()
    
out.close()


