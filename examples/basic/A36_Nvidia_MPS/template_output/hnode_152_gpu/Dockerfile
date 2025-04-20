FROM 4fae3969fde5c17a0217e0d4a8134ea5
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
RUN apt-get update && apt-get install -y --no-install-recommends curl dnsutils ipcalc iproute2 iputils-ping jq mtr-tiny nano netcat python3 python3-pip tcpdump termshark vim-nox zsh
RUN pip3 install torch && pip3 install numpy
COPY 082b96ec819c95ae773daebde675ef80 /start.sh
COPY ddf3c39b6b6ea1cb444eba92b04a53f2 /MPS_test
COPY b25d14e93b5400b5e13d52550c1fccb1 /mps_run
COPY 62405c7af776a1a145d268069ab8c9a6 /cuda_memory
COPY d18858afc6bb66ec3a19d872077acfd2 /seedemu_sniffer
COPY 17ac2d812a99a91e7f747e1defb72a29 /seedemu_worker
RUN chmod +x /start.sh
RUN chmod +x /MPS_test
RUN chmod +x /cuda_memory
RUN chmod +x /seedemu_sniffer
RUN chmod +x /seedemu_worker
COPY e01e36443f9f72c6204189260d0bd276 /ifinfo.txt
COPY d3d51fdf7f4bad30dc5db560a01ce629 /interface_setup
COPY 5cda357fa940acc13afff446a1f7d3b6 /GPUTest.py
COPY e432ef745a4d93a7059b78b10f645710 /GPUTest_Time.py
CMD ["/start.sh"]

