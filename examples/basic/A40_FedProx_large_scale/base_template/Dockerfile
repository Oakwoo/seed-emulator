FROM nvidia/cuda:12.3.1-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget
COPY STAGED_PATH TARGET_PATH
RUN export CONDA_DIR=/opt/conda && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && export PATH=$CONDA_DIR/bin:$PATH && conda init && conda create -y -n py310 python=3.10
RUN /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && pip3 install -e TARGET_PATH'

