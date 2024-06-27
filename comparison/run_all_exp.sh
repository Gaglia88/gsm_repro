conda activate py10

echo "Deepblocker"
cd deepblocker
# Checks if embeddings exists, if not download them
if [ ! -d "embeddings" ]; then
   mkdir embeddings
   cd embeddings
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
   unzip wiki.en.zip
   rm wiki.en.zip
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

print "Contextual blocker"
conda activate py9
cd contextualblocker
python run_exp.py
conda deactivate
cd ..

print "Sudowoodo"
conda activate py37
cd sudowoodo
# Download needed datasets
if [ ! -d "data" ]; then
   mkdir data
   cd data
   wget https://sparc20.ing.unimore.it/gsm_repro/sudowoodo_data.tar.gz
   tar -xvf sudowoodo_data.tar.gz
   rm sudowoodo_data.tar.gz
   cd ..
fi
python run_exp.py
rm -rf mlruns
rm -rf result_blocking
conda deactivate
cd ..