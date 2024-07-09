import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sparkly.index import IndexConfig, LuceneIndex
from sparkly.search import Searcher
from pathlib import Path
import shutil
import os
import time
import json
import traceback

spark = SparkSession.builder.getOrCreate()

def load_datasets(path='/home/app/datasets/datasets.json', dtype=''):
    f = open(path)
    datasets = json.load(f)
    f.close()
    if len(dtype) > 0:
        datasets = list(filter(lambda d: d['type']==dtype, datasets))
    return datasets

def transform(tbl, id_field):
    return tbl.fillna(" ").select(id_field, F.concat_ws(*tbl.columns).alias('text'))

def fix(tableA):
    a = [(c, c.replace('.', '').replace('/', '').replace(':', '')) for c in tableA.columns if c != '_id']
    return tableA.rename(dict(a), axis=1)

def convert_dataset(d1, d2, gt):
    d1 = d1.copy()
    d2 = d2.copy()
    gt = gt.copy()
    d1 = d1.astype('str')
    d2 = d2.astype('str')
    gt = gt.astype('str')
    
    d1 = d1.reset_index().rename({'index': 'ltable_id'}, axis=1)
    d2 = d2.reset_index().rename({'index': 'rtable_id'}, axis=1)
    tmp = gt.merge(d1, left_on='id1', right_on='realProfileID')[['ltable_id', 'id2']]
    new_gt = tmp.merge(d2, left_on='id2', right_on='realProfileID')[['ltable_id', 'rtable_id']]
    
    d1 = d1.drop('realProfileID', axis=1)
    d2 = d2.drop('realProfileID', axis=1)
    d1 = d1.rename({'ltable_id': '_id'}, axis=1)
    d2 = d2.rename({'rtable_id': '_id'}, axis=1)
    return d1, d2, new_gt

def block(dataset, limit=50):
    id_field="_id"
    # Path in which the data are stored
    base_path = "/home/app/"+dataset["base_path"]+"/"
    # the analyzers used to convert the text into tokens for indexing
    analyzers = ['3gram']

    d1 = pd.read_json(f"{base_path}dataset1.json", lines=True).rename({'id':'id12345'}, axis=1).fillna("")
    d2 = pd.read_json(f"{base_path}dataset2.json", lines=True).rename({'id':'id12345'}, axis=1).fillna("")
    gt = pd.read_json(f"{base_path}groundtruth.json", lines=True)

    tableA, tableB, table_match = convert_dataset(d1, d2, gt)

    table_a = spark.createDataFrame(fix(tableA))
    table_b = spark.createDataFrame(fix(tableB))
    gold = spark.createDataFrame(table_match)

    table_a = transform(table_a, id_field)
    table_b = transform(table_b, id_field)
    
    tstart = time.time()

    # the index config, '_id' column will be used as the unique 
    # id column in the index. Note id_col must be an integer (32 or 64 bit)
    config = IndexConfig(id_col='_id')
    # add the 'name' column to be indexed with analyzer above
    config.add_field('text', analyzers)
    # create a new index stored at /tmp/example_index/

    if os.path.isdir('/tmp/example_index/'):
        shutil.rmtree('/tmp/example_index/')

    index = LuceneIndex('/tmp/example_index/', config)
    # index the records from table A according to the config we created above
    index.upsert_docs(table_a)

    # get a query spec (template) which searches on 
    # all indexed fields
    query_spec = index.get_full_query_spec()
    # create a searcher for doing bulk search using our index
    searcher = Searcher(index)
    # search the index with table b
    candidates = searcher.search(table_b, query_spec, id_col='_id', limit=limit).cache()

    candidates.show()
    # output is rolled up 
    # search record id -> (indexed ids + scores + search time)
    #
    # explode the results to compute recall
    pairs = candidates.select(
                        F.explode('ids').alias('a_id'),
                        F.col('_id').alias('b_id')
                    )


    # number of matches found
    true_positives = gold.intersect(pairs).count()
    # precentage of matches found
    recall = true_positives / gold.count()
    precision = true_positives/pairs.count()
    tend = time.time()
    f1 = (2*precision*recall)/(precision+recall)
    candidates.unpersist()
    
    return f"{dataset['name']};{limit};{recall};{precision};{f1};{tend-tstart}\n"
    

if __name__ == "__main__":
    datasets = load_datasets(dtype="clean")
    
    if not os.path.isdir('/home/app/results/'):
        os.makedirs('/home/app/results/', exist_ok=True)

    out = open('/home/app/results/sparkly_k_best.csv', 'wt')
    out.write("dataset;k;recall;precision;f1;runtime\n")
    for d in datasets:
        try:
            print(d['name'])
            out.write(block(d, d['sparkly_k']))
            out.flush()
        except Exception as e:
            print(traceback.format_exc())
            pass
    out.close()