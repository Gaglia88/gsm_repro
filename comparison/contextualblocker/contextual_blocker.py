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

def load_data(data):
    dname = data['dataset']
    d1 = pd.read_json(f"/data/luca/datasets/{dname}/dataset1.json", lines=True).fillna("")
    d2 = pd.read_json(f"/data/luca/datasets/{dname}/dataset2.json", lines=True).fillna("")
    gt = pd.read_json(f"/data/luca/datasets/{dname}/groundtruth.json", lines=True).fillna("")

    # Convert dataset and groundtruth in the form ltable_id, rtable_id
    table1, table2, table_match = convert_dataset(d1, d2, gt)

    table1 = get_text_repr(table1)
    table2 = get_text_repr(table2)

    table3 = pd.concat([table1, table2])
    
    pairs = set()
    for pair in zip(table_match['ltable_id'], table_match['rtable_id']):
            ordered_pair = tuple(sorted((table3.loc[table3['id'] == pair[0]].index[0], table3.loc[table3['id'] == pair[1]].index[0])))
            pairs.add(ordered_pair)

    table3.drop(labels=['id'], axis=1, inplace=True)

    return dname, table3, pairs

parser = argparse.ArgumentParser(description='Graph-based Block Clustering')

parser.add_argument("--verbose", type=int, default='0',choices=[0, 1, 2], help="increase output verbosity")
parser.add_argument("--dataset", type=str,default='walmart-amazon-clean', help='dataset')
parser.add_argument("--blocker", type=str,default='louvian', help='clustering algorithm')
parser.add_argument("--num_clusters", type=int,default='6', help='used in knn graph')
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

prog_start = time.time()
# 1) LOAD and PREPROCESS the dataset

dataset_name, table, pairs = load_data(key_values)

# 2) DO the embedding

start = time.time()
attributes = hp.attributes # get attr names
records = table[attributes].agg(' '.join, axis=1)
#transformer = model(hp) # uncomment if needed
simCSE = SimCSE(hp.model_name) # pre-trained model 

print(f'constucting embedding space chosen attributes: {attributes}')
vectors = simCSE.encode(list(records), device="cuda", max_length=hp.max_seq_length)
print("TIME: {:.2f}".format(time.time() - start))

time_list = list()
key_values = {}

for num_clusters in range(hp.lower_limit, hp.upper_limit+1, hp.step):
    print()
    
    key_values["num_clusters"] = num_clusters
    print(f"building blocks with: {num_clusters} clusters")
    
    start = time.time()
    data = all_in_one_clusteriser(vectors, hp.blocker, num_clusters)
    reduction_ratio, pair_completeness, reference_metric, pair_quality, fmeasure = calc_index(data, table, pairs)
    
    
    print("(RR) Reduction ratio is: {:.2f}".format(reduction_ratio))
    print("(PC) Pair completeness is: {:.2f}".format(pair_completeness))
    print("(PQ) Pair quality is: {:.2f}".format(pair_quality))
    print("(PQ) fmeasure is: {:.2f}".format(fmeasure))
    print("(RM) Reference metric (Harmonic mean RR and PC) is: {:.2f}".format(reference_metric))
    
    end = time.time()
    blocking_time = end - start
    time_list.append(blocking_time)  # blocking time for 30 loops
    
    print(">> Blocking time was roughly {:.2f} seconds for {} tuples!".format(blocking_time, vectors.shape[0]))
    print("*" * 50)



