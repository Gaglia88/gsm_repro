#Max memory for scala/java scripts
MAX_MEMORY="100g"
# Local Spark directory, do not change it
SPARK_DIR="spark-3.0.1-bin-hadoop2.7"

export SPARK_LOCAL_IP="localhost"
export SBT_OPTS="-Xmx$MAX_MEMORY"

# Checks if datasets exists, if not download them
if [ ! -d "datasets" ]; then
   wget https://dbgroup.ing.unimore.it/gsm_repro/gsm_repro_datasets.tar.gz
   tar -xvf gsm_repro_datasets.tar.gz
   rm gsm_repro_datasets.tar.gz
fi

# Checks if spark exists, if not download it
if [ ! -d "$SPARK_DIR" ]; then
   wget https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz;
   tar -xvf spark-3.0.1-bin-hadoop2.7.tgz;
   rm spark-3.0.1-bin-hadoop2.7.tgz;
fi

cd scala
# Builds all the features and generate data for Table 1
echo "Builds all the features and generate data for Table 1"
sbt "runMain Experiments.BuildFeatures"
cd ..


cd python
# Generate the probabilities for each feature set
echo "Generate the probabilities for each feature set"
python3.7 01_gen_all_probabilities.py

# Generate the probabilities varying the traning set size
echo "Generate the probabilities varying the traning set size"
python3.7 02_var_training_set_size.py

# Generate the probabilities for the dirty datasets (for the scalability experiment)
echo "Generate the probabilities for the dirty datasets (for the scalability experiment)"
python3.7 03_gen_dirty_probabilities.py
cd ..

cd java
# Perform the algorithm selection (data for Figures 6, 7, 10, 11)
echo "Perform the algorithm selection (data for Figures 6, 7, 10, 11)"
java -Xmx${MAX_MEMORY} -cp "supMB.jar:lib/*" supervisedMB.AlgorithmSelection

# Perform the feature selection (data for Tables 2, 3 and for Figures 8, 9)
echo "Perform the feature selection (data for Tables 2, 3 and for Figures 8, 9)"
java -Xmx${MAX_MEMORY} -cp "supMB.jar:lib/*" supervisedMB.FeatureSelection

# Perform the training size selection (data for Figures 12, 13)
echo "Perform the training size selection (data for Figures 12, 13)"
java -Xmx${MAX_MEMORY} -cp "supMB.jar:lib/*" supervisedMB.TrainSizeSelection

# Generate data for Table 4
echo "Generate data for Table 4"
java -Xmx${MAX_MEMORY} -cp "supMB.jar:lib/*" supervisedMB.Table4

# Perform the scalability experiment (data for Figure 16)
echo "Perform the scalability experiment (data for Figure 16)"
java -Xmx${MAX_MEMORY} -cp "supMB.jar:lib/*" supervisedMB.Scalability

cd ..

cd scala
# Compute the runtime for generating the different feature sets, needed for Figures 8, 9
echo "Compute the runtime for generating the different feature sets, needed for Figures 8, 9"
sbt "runMain Experiments.CalcFeaturesTime"

# Compute the CBS for the different datasets, needed for Figures 17, 18
echo "Compute the CBS for the different datasets, needed for Figures 17, 18"
sbt "runMain Experiments.CalcCBS"

# Compute and store BLAST thresholds, for Figure 14.
echo "Compute and store BLAST thresholds, for Figure 14."
sbt "runMain Experiments.ThresholdCalc"

cd ..

cd python

# Perform the progressive experiments
echo "Perform the progressive experiments - Step 1"
spark-submit --conf spark.driver.memory=${MAX_MEMORY} --conf spark.executor.memory=${MAX_MEMORY} --conf spark.local.dir=/home/app/tmp/ --conf spark.driver.maxResultSize=-1 progressive_step01.py
echo "Perform the progressive experiments - Step 2"
python3.7 progressive_step02.py
echo "Perform the progressive experiments - Step 3"
python3.7 progressive_step03.py