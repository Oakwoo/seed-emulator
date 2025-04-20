FROM nvidia/cuda:12.3.1-base-ubuntu20.04
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
COPY 082b96ec819c95ae773daebde675ef80 /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
