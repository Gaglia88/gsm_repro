#!/bin/bash
set -e

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec > >(tee gsm_comparison_log.txt) 2>&1
# Everything below will go to the file 'gsm_comparison_log.txt':

# Free space is on by default, no needed files are removed, set 0 will disable it
export FREE_SPACE=1

# Checks if datasets exists, if not download them
if [ ! -e "/home/app/datasets/downloaded.txt" ]; then
   wget -P /home/app/ https://dbgroup.ing.unimore.it/gsm_repro/gsm_repro_datasets.tar.gz
   tar -xvf /home/app/gsm_repro_datasets.tar.gz -C /home/app/
   rm /home/app/gsm_repro_datasets.tar.gz
fi

source /opt/conda/etc/profile.d/conda.sh

conda activate py10

echo "[GSM REPRO] Deepblocker"
cd deepblocker
# Checks if embeddings exists, if not download them
if [ ! -d "/home/app/comparison/deepblocker/embedding" ]; then
   echo "[GSM REPRO] Downloads fasttext embeddings"
   mkdir /home/app/comparison/deepblocker/embedding
   cd /home/app/comparison/deepblocker/embedding
   wget -q -P /home/app/comparison/deepblocker/ https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
   unzip /home/app/comparison/deepblocker/wiki.en.zip -d /home/app/comparison/deepblocker/embedding
   cd ..
fi

python run_exp.py
cd ..

echo "[GSM REPRO] Sparkly"
cd sparkly
python run_best_k.py
python run_k10.py
conda deactivate
cd ..

echo "[GSM REPRO] Contextual blocker"
conda activate py9
cd contextualblocker
python run_exp.py
conda deactivate
cd ..


echo "[GSM REPRO] Sudowoodo"
conda activate py37
cd sudowoodo
# Download needed datasets
if [ ! -d "/home/app/comparison/sudowoodo/data" ]; then
   echo "[GSM REPRO] Download Sudowoodo data"
   wget -q -P /home/app/comparison/sudowoodo/ https://sparc20.ing.unimore.it/gsm_repro/sudowoodo_data.tar.gz
   tar -xvf /home/app/comparison/sudowoodo/sudowoodo_data.tar.gz -C /home/app/comparison/sudowoodo/
   rm /home/app/comparison/sudowoodo/sudowoodo_data.tar.gz
fi
if [ ! -d "apex" ]; then
   git clone https://github.com/NVIDIA/apex
fi
conda install -c nvidia -y cuda-nvcc
cd apex
python setup.py install
cd ..
python run_exp.py
# Remove data for space saving
if [[ "$FREE_SPACE" -eq 1 ]]; then
   rm -rf /home/app/comparison/sudowoodo/mlruns
   rm -rf /home/app/comparison/sudowoodo/result_blocking
   rm -rf /home/app/comparison/sudowoodo/data
   rm -rf /home/app/comparison/deepblocker/embedding
   rm /home/app/comparison/deepblocker/wiki.en.zip
   rm -rf /home/app/comparison/sudowoodo/apex
   rm /home/app/comparison/sudowoodo/blocking_result.pkl
fi
conda deactivate
cd ..
