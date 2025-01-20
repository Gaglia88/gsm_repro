FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ADD . /home/app
WORKDIR /home/app

RUN apt-get update
RUN apt-get install -y build-essential git curl wget unzip
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN . "$HOME/.cargo/env"    

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda create -n py10 python=3.10 -y

RUN wget https://downloads.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz
RUN tar -xvf pylucene-9.4.1-src.tar.gz
RUN rm pylucene-9.4.1-src.tar.gz

RUN wget  https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.3%2B7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.3_7.tar.gz
RUN tar -xvf OpenJDK17U-jdk_x64_linux_hotspot_17.0.3_7.tar.gz
RUN rm OpenJDK17U-jdk_x64_linux_hotspot_17.0.3_7.tar.gz
RUN mv /home/app/jdk-17.0.3+7/ /usr/bin/temurin-jdk-17.0.3+7/

ENV JAVA_HOME /usr/bin/temurin-jdk-17.0.3+7/
ENV PATH $JAVA_HOME/bin:$PATH
ENV JCC_JDK /usr/bin/temurin-jdk-17.0.3+7/
ENV PYTHON=/opt/conda/envs/py10/bin/python
ENV PREFIX_PYTHON=/opt/apache/pylucene/_install
ENV JCC="$PYTHON -m jcc --shared"
ENV NUM_FILES=16
ENV TORCH_CUDA_ARCH_LIST="compute capability"

RUN conda init && . ~/.bashrc && conda activate py10 && pip install setuptools==69.5.1

WORKDIR /home/app/pylucene-9.4.1/jcc
RUN conda init && . ~/.bashrc && conda activate py10 && python setup.py build
RUN conda init && . ~/.bashrc && conda activate py10 && python setup.py install

WORKDIR /home/app/pylucene-9.4.1
RUN conda init && . ~/.bashrc && conda activate py10 && make
RUN conda init && . ~/.bashrc && conda activate py10 && make install

WORKDIR /home/app

RUN conda init && . ~/.bashrc && conda activate py10 && pip install scikit-learn==1.5.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install torch==2.3.1
RUN conda init && . ~/.bashrc && conda activate py10 && pip install fasttext==0.9.3
RUN conda init && . ~/.bashrc && conda activate py10 && pip install torchtext==0.18.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install joblib==1.2.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install mmh3==3.0.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install numba==0.56.4
RUN conda init && . ~/.bashrc && conda activate py10 && pip install numpy==1.21.5
RUN conda init && . ~/.bashrc && conda activate py10 && pip install numpydoc==1.5.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install psutil==5.9.4
RUN conda init && . ~/.bashrc && conda activate py10 && pip install pyarrow==10.0.1
RUN conda init && . ~/.bashrc && conda activate py10 && pip install pyspark==3.3.1
RUN conda init && . ~/.bashrc && conda activate py10 && pip install scipy==1.8.0
RUN conda init && . ~/.bashrc && conda activate py10 && pip install tqdm==4.64.1
RUN conda init && . ~/.bashrc && conda activate py10 && pip install pandas==1.5.2

RUN conda create -n py9 python=3.9 -y
RUN conda init && . ~/.bashrc && conda activate py9 && pip install simcse==0.4
RUN conda init && . ~/.bashrc && conda activate py9 && pip install cdlib==0.4.0
RUN conda init && . ~/.bashrc && conda activate py9 && pip install networkx==2.6.0 

RUN  conda create -n py37 python=3.7.10
#RUN conda init && . ~/.bashrc && conda activate py37 && conda install -c nvidia -y cuda-nvcc
RUN conda init && . ~/.bashrc && conda activate py37 && pip install numpy==1.21.6
RUN conda init && . ~/.bashrc && conda activate py37 && pip install regex==2024.4.16
RUN conda init && . ~/.bashrc && conda activate py37 && pip install scipy==1.7.3
RUN conda init && . ~/.bashrc && conda activate py37 && pip install sentencepiece==0.2.0
RUN conda init && . ~/.bashrc && conda activate py37 && pip install scikit-learn==1.0.2
RUN conda init && . ~/.bashrc && conda activate py37 && pip install spacy==3.7.5
RUN conda init && . ~/.bashrc && conda activate py37 && pip install tensorboardX==2.6.2.2
RUN conda init && . ~/.bashrc && conda activate py37 && pip install jsonlines==3.1.0
RUN conda init && . ~/.bashrc && conda activate py37 && pip install nltk==3.8.1
RUN conda init && . ~/.bashrc && conda activate py37 && pip install torch==1.13.1
RUN conda init && . ~/.bashrc && conda activate py37 && pip install tqdm==4.66.4
RUN conda init && . ~/.bashrc && conda activate py37 && pip install transformers==4.30.2
RUN conda init && . ~/.bashrc && conda activate py37 && pip install pandas==1.3.5
RUN conda init && . ~/.bashrc && conda activate py37 && pip install mlflow==1.30.1
#RUN git clone https://github.com/NVIDIA/apex
#RUN conda init && . ~/.bashrc && conda activate py37 && cd apex && python setup.py install
