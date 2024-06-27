import pandas as pd
import time
import subprocess
import pickle

def calc_metrics(dataset, basepath):
    pf = open('blocking_result.pkl', 'rb')
    a = pickle.load(pf)
    pf.close()
    gt = pd.read_csv(f'/home/app/comparison/sudowoodo/data/{basepath}/{dataset}/groundtruth.csv')
    candidates = pd.DataFrame(a, columns=['ltable_id', 'rtable_id', 'w']).drop("w", axis=1)

    found = len(candidates.merge(gt))
    recall = 0.0
    precision = 0.0
    f1 = 0.0
    try:
        recall = found/len(gt)
    except:
        pass
    try:
        precision = found/len(candidates)
    except:
        pass
    try:
        f1 = 2*(precision*recall) / (precision+recall)
    except:
        pass
    
    return recall, precision, f1

def process(dataset, basepath='em'):
    stime = time.time()
    p1 = subprocess.Popen(['python', 'train_bt.py', '--task_type', basepath, '--task', dataset, 
    '--logdir', 'result_blocking/', '--ssl_method', 'barlow_twins', '--batch_size', '64', 
    '--lr', '5e-5', '--lm', 'distilbert', '--n_ssl_epochs', '5', '--n_epochs', '5', '--max_len', '128', 
    '--projector', '4096', '--da', 'del', '--blocking', '--fp16', '--save_ckpt', '--k', '50', '--run_id', '0'])
    p1.wait()
    
    etrain = time.time()
    
    d = f"/home/app/comparison/sudowoodo/data/{basepath}/{dataset}"
    
    p2 = subprocess.Popen(['python', 'blocking.py', '--task', d, '--logdir', 'result_blocking', '--batch_size', '512', '--max_len', '128', '--projector', '4096', '--lm', 'distilbert', '--fp16', '--k', '50', '--ckpt_path', f'result_blocking/{dataset}/ssl.pt'])
    p2.wait()
    
    etime = time.time()
    
    recall, precision, f1 = calc_metrics(dataset, basepath)
    
    return recall, precision, f1, (etrain-stime), (etime-etrain), (etime-stime)


datasets = ['abtBuy', 'DblpAcm', 'movies', 'imdb_tvdb', 'tmdb_tvdb', 'scholarDblp', 'walmartAmazon', 'amazonGoogleProducts', 'imdb_tmdb']


if __name__ == "__main__":
    out = open('/home/app/results/sudowoodo_results.csv', 'wt')
    out.write("dataset, recall, precision, f1, train_time, block_time, ov_time\n")
    for d in datasets:
        try:
            print(d)
            recall, precision, f1, train_time, block_time, ov_time = process(d)
            out.write(f"{d}, {recall}, {precision}, {f1}, {train_time}, {block_time}, {ov_time}\n")
            out.flush()
        except:
            out.write(d)
            pass
    out.close()
    
    out = open('/home/app/results/sudowoodo_results_50.csv', 'wt')
    out.write("dataset, recall, precision, f1, train_time, block_time, ov_time\n")
    for d in datasets:
        try:
            print(d)
            recall, precision, f1, train_time, block_time, ov_time = process(d, basepath="em_50")
            out.write(f"{d}, {recall}, {precision}, {f1}, {train_time}, {block_time}, {ov_time}\n")
            out.flush()
        except:
            out.write(d)
            pass
    out.close()