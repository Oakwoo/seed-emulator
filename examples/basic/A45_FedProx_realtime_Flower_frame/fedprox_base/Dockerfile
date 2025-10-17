FROM nvidia/cuda:12.3.1-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget
COPY 8fcf2779c94929e9859b1d71ebe5b210 /fedprox_realtime
RUN export CONDA_DIR=/opt/conda && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && export PATH=$CONDA_DIR/bin:$PATH && conda init && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && conda create -y -n py310 python=3.10
RUN /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && pip3 install -e /fedprox_realtime'

