FROM 4fae3969fde5c17a0217e0d4a8134ea5
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl dnsutils ipcalc iproute2 iputils-ping jq mtr-tiny nano netcat tcpdump termshark vim-nox wget zsh
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
COPY 082b96ec819c95ae773daebde675ef80 /start.sh
COPY d18858afc6bb66ec3a19d872077acfd2 /seedemu_sniffer
COPY 17ac2d812a99a91e7f747e1defb72a29 /seedemu_worker
RUN chmod +x /start.sh
RUN chmod +x /seedemu_sniffer
RUN chmod +x /seedemu_worker
COPY e01e36443f9f72c6204189260d0bd276 /ifinfo.txt
COPY d3d51fdf7f4bad30dc5db560a01ce629 /interface_setup
COPY 6a1e5579ad5a86c362c6aae90c32215e /tmp/etc-hosts
COPY 53be52b6016ab1f879517691a07f59ac /quickstart-pytorch
RUN export CONDA_DIR=/opt/conda && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && export PATH=$CONDA_DIR/bin:$PATH && conda init && conda create -y -n py310 python=3.10
RUN /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && pip3 install -e /quickstart-pytorch'
CMD ["/start.sh"]

