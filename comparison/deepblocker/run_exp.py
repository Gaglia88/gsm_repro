#!/usr/bin/env python
# coding: utf-8

# In[5]:

import traceback
import pandas as pd
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
import time
import json
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import os

# In[10]:


base_path = "/home/app/"

def load_datasets(path='/home/app/datasets/datasets.json', dtype=''):
    f = open(path)
    datasets = json.load(f)
    f.close()
    if len(dtype) > 0:
        datasets = list(filter(lambda d: d['type']==dtype, datasets))
    return datasets


# In[11]:


clean_datasets = load_datasets(dtype='clean')
models = ["AutoEncoderTupleEmbedding"]

def load_data(path, idfield="realProfileID"):
    p1 = pd.read_json(f"{path}/dataset1.json", lines=True).fillna("").astype('string').rename({"id": "idxyzw1"}, axis =1)
    p2 = pd.read_json(f"{path}/dataset2.json", lines=True).fillna("").astype('string').rename({"id": "idxyzw2"}, axis =1)
    gt = pd.read_json(f"{path}/groundtruth.json", lines=True).astype('string')

    if "id1" in p1.columns.values:
        raise Exception("id1 in p1")
    if "id2" in p2.columns.values:
        raise Exception("id2 in p2")
    if "rtable_id" in p1.columns.values:
        raise Exception("rtable_id in p1")
    if "ltable_id" in p2.columns.values:
        raise Exception("ltable_id in p2")



    p1 = p1.reset_index().rename({"index": "id"}, axis =1)
    p2 = p2.reset_index().rename({"index": "id"}, axis =1)

    tmp = gt.merge(p1, left_on="id1", right_on=idfield)[["id", "id2"]].rename({"id": "ltable_id"}, axis=1)
    gt_conv = tmp.merge(p2, left_on="id2", right_on=idfield)[["ltable_id", "id"]].rename({"id": "rtable_id"}, axis=1)
    
    if len(gt_conv) != len(gt):
        raise Exception("gt conv failed")
    
    return p1, p2, gt_conv

def removeID(cols, idname="realProfileID"):
    return [x for x in cols if x != idname]

def flatten(row):
    return " ".join(map(lambda x: str(x), row.values))

def get_repr(df):
    return df[removeID(df.columns)].apply(flatten, axis=1)

def get_text_repr(df):
    df1 = df[['id']].copy()
    df1['text'] = get_repr(df.drop('id', axis=1))
    return df1


def do_process(dataset, k, idfield = "realProfileID", model="AutoEncoderTupleEmbedding"):
    global base_path
    
    try:
        dname = dataset["name"]
        # Path in which the data are stored
        d = base_path+dataset["base_path"]
        
        # Load data
        d1, d2, gt = load_data(d)
        
        # Put all values together
        d1_txt = get_text_repr(d1)
        d2_txt = get_text_repr(d2)
        
        # Remove any empty value, otherwise deep_blocker not working
        d1_txt['text'] = d1_txt['text'].str.strip()
        d2_txt['text'] = d2_txt['text'].str.strip()
        d1_txt = d1_txt[d1_txt['text'].str.len()>0]
        d2_txt = d2_txt[d2_txt['text'].str.len()>0]
        

        stime = time.time()
        # Model to use
        if model == "CTTTupleEmbedding":
            tuple_embedding_model = CTTTupleEmbedding()
        elif model == "AutoEncoderTupleEmbedding":
            tuple_embedding_model = AutoEncoderTupleEmbedding()
        else:
            tuple_embedding_model = HybridTupleEmbedding()

        # Se k
        topK_vector_pairing_model = ExactTopKVectorPairing(K=k)
        # Initialize DeepBlocker
        db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)


        ttime = time.time()

        # Columns to block, all
        cols_to_block = ["text"]

        # Generate the candidate set
        candidate_set_df = db.block_datasets(d1_txt, d2_txt, cols_to_block)

        etime = time.time()

        # Compute precision/recall
        res = blocking_utils.compute_blocking_statistics(candidate_set_df, gt, d1_txt, d2_txt)

        return dname, model, res['recall'], res['precision'], len(candidate_set_df), ttime-stime, etime-ttime, etime-stime
    except:
        print(traceback.format_exc())
        return dname, model, 'error', 0, 0, 0, 0, 0


# In[17]:

if __name__ == "__main__":
    if not os.path.isdir('/home/app/results/'):
        os.makedirs('/home/app/results/', exist_ok=True)


    out = open('/home/app/results/deepblocker.csv', 'wt')
    out.write("k;dataset;model_name;recall;precision;candidates;train_time;block_time;overall_time\n")
    for model in models:
        for i in range(0, len(clean_datasets)):
            print(f"Processing {clean_datasets[i]['name']}")
            dataset, model_name, recall, precision, candidates, train_time, block_time, overall_time = do_process(clean_datasets[i], 
                                                                                                 clean_datasets[i]['deepblocker_k'], 
                                                                                                 model=model)        
            out.write(f"{clean_datasets[i]['deepblocker_k']};{dataset};{model_name};{recall};{precision};{candidates};{train_time};{block_time};{overall_time}\n")
            out.flush()
    out.close()


# In[ ]:




