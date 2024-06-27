#!/bin/bash
set -e

# Checks if datasets exists, if not download them
if [ ! -d "/home/app/datasets" ]; then
   wget -P /home/app/ https://dbgroup.ing.unimore.it/gsm_repro/gsm_repro_datasets.tar.gz
   tar -xvf /home/app/gsm_repro_datasets.tar.gz -C /home/app/
   rm /home/app/gsm_repro_datasets.tar.gz
fi

source /opt/conda/etc/profile.d/conda.sh

conda activate py10

echo "Deepblocker"
cd deepblocker
# Checks if embeddings exists, if not download them
if [ ! -d "embedding" ]; then
   echo "Downloads fasttext embeddings"
   mkdir embedding
   cd embedding
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
   unzip wiki.en.zip
   cd ..
fi
python run_exp.py
cd ..

echo "Sparkly"
cd sparkly
python run_best_k.py
python run_k10.py
conda deactivate
cd ..

echo "Contextual blocker"
conda activate py9
cd contextualblocker
python run_exp.py
conda deactivate
cd ..


echo "Sudowoodo"
conda activate py37
cd sudowoodo
# Download needed datasets
if [ ! -d "data" ]; then
   echo "Download Sudowoodo data"
   wget https://sparc20.ing.unimore.it/gsm_repro/sudowoodo_data.tar.gz
   tar -xvf sudowoodo_data.tar.gz
   rm sudowoodo_data.tar.gz
fi
if [ ! -d "apex" ]; then
   git clone https://github.com/NVIDIA/apex
fi
conda install -c nvidia -y cuda-nvcc
cd apex
python setup.py install
cd ..
python run_exp.py
rm -rf mlruns
rm -rf result_blocking
conda deactivate
cd ..