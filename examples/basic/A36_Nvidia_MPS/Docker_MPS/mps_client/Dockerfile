FROM nvidia/cuda:12.3.1-base-ubuntu20.04
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
COPY 082b96ec819c95ae773daebde675ef80 /start.sh
COPY ddf3c39b6b6ea1cb444eba92b04a53f2 /MPS_test
COPY b25d14e93b5400b5e13d52550c1fccb1 /mps_run
COPY 62405c7af776a1a145d268069ab8c9a6 /cuda_memory
RUN chmod +x /start.sh
RUN chmod +x /MPS_test
RUN chmod +x /cuda_memory
CMD ["/start.sh"]
