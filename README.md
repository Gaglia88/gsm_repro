## Reproducibility experiments for Generalized Supervised Meta-blocking
This repository contains the code to reproduce the experiments performed in [1].

### Structure
* *datasets* contains the datasets used in the project. They will be downloaded at the first execution of `run_all_tests.sh` script. It is possible to edit the file *datasets/datasets.json* to add/remove them. For each dataset, the parameter *purging_threshold* (default value 1.0) affects the number of retained comparisons, decreasing it will discard more comparisons, but reducing recall. It can be lowered for the larger datasets if the available RAM is not enough (see also requirements section).
*  *config* contains the list of available feature sets and a configuration file. The configuration file lets to set the maximum memory for Spark execution, the number of parallel processes that run at a time in the different experiments, and the number of repetitions for each experiment.
* *docker* contains the docker files to create a machine to reproduce the main experiments.
* *java* contains the Java code, needed for the experiments.
* *python* contains the Python code, needed for the experiments.
* *scala* contains the Scala code, needed for the experiments.
* *notebooks* contains the notebooks needed to generate the charts/tables of the paper.
* *comparison* contains the code needed to reproduce the comparison with other frameworks (Table 5 of the paper).

### Requirements
* The original experiments were performed on a machine with four Intel Xeon E5-2697 2.40 GHz (72 cores), 216 GB of RAM, running Ubuntu 18.04.
With less memory, the largest datasets could cause out-of-memory issues. In particular, 200K and 300K datasets could cause this issue, a solution to test everything is to lower the *purging_threshold* parameter in the file *datasets/datasets.json*. This will cause a more aggressive pruning, reducing the number of retained comparisons, but also recall.
* [Docker](http://www.docker.com) is needed to run the experiments.
* To perform the comparison with the other tools (Table 5 in the paper) a GPU is needed. Also Docker need to be configured to use it, this require to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Reproduce the experiments
#### 1. Comparison with other frameworks
In the paper we compared the performance of Generalized Supervised Meta-blocking with [Sudowoodo](https://github.com/megagonlabs/sudowoodo), [ContextualBlocker](https://github.com/boscoj2008/ContextualBlocker-for-EM), [DeepBlocker](https://github.com/qcri/DeepBlocker) and [Sparkly](https://github.com/anhaidgroup/sparkly).
To perform this comparison, a different docker machine is needed, due to the heterogeneity of the different setups.

##### Step 1
Move in the `comparison` folder and run the command `start_docker.sh`, this will create and start a new docker machine called `gsm_comparison`.

##### Step 2
Inside the docker machine, move in the `comparison` folder and run the script `run_all_exp.sh`, this will process all datasets with the four frameworks.

##### Step 3
Close the docker machine and proceed with the main experiments.

#### 2. Main experiments
##### Step 1
Open the file `config/config.ini` and set, based on your system,
* `max_memory` is the maximum memory used by Apache Spark.
* `parallel_process` is the number of parallel processes that run in some experiments. A higher value requires more memory, so it can cause out-of-memory problems.
* `repetitions` number of repetitions for each experiment. The default value is 10, a lower value requires less time to complete all the experiments.
**Do not touch the other parameters in the configuration file.**

##### Step 2
Open the file `run_all_tests.sh` and set the `MAX_MEMORY` parameter based on your system configuration. This is the maximum memory that can be used by the Java Virtual Machine.

##### Step 3
Open a shell, move to the `docker` folder, and run the script `start_docker.sh`. The script will run the docker-machine and login inside it.

##### Step 4
Inside the docker machine, run the script `run_all_tests.sh`. It will run all the experiments. The resulting files will be placed inside the *results* folder.

##### Step 5
When all the experiments are completed, run the script `start_notebook.sh`.
This will start Jupyter inside the docker-machine which has a port forwarding on port 8888.
Opening from the browser *your machine ip:8888* will open the notebook environment.
Running the notebook it is possible to reproduce all the Figures/Tables of the paper [1].

### References
[1] Gagliardelli, L., Papadakis, G., Simonini, G., Bergamaschi, S., & Palpanas, T. (2023). GSM: A generalized approach to Supervised Meta-blocking for scalable entity resolution. _Information Systems_, 102307.
