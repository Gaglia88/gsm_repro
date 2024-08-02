FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y build-essential git curl wget

RUN apt-get install -y python3.7
RUN apt-get install -y python3-distutils
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py
RUN rm get-pip.py
RUN python3.7 -m pip install pip

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import
RUN chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg
RUN apt-get update
RUN apt-get install sbt=1.3.13

ADD . /home/app
WORKDIR /home/app

RUN apt-get install -y openjdk-11-jdk
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

ENV PATH $JAVA_HOME/bin:$PATH

RUN pip install --upgrade pip

RUN pip install pandas==1.2.2
RUN pip install scikit-learn==0.24.1
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install networkx==2.6.3
RUN pip install jupyter==1.0.0
RUN pip install matplotlib==3.5.3

#RUN wget https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
#RUN tar -xvf spark-3.0.1-bin-hadoop2.7.tgz
#RUN rm spark-3.0.1-bin-hadoop2.7.tgz
ENV SPARK_HOME="/home/app/spark-3.0.1-bin-hadoop2.7"
ENV PATH="${PATH}:${SPARK_HOME}/bin"
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3.7
ENV PYSPARK_PYTHON=/usr/bin/python3.7
