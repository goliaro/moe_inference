FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/* \
	/etc/apt/sources.list.d/cuda.list \
	/etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    	apt-get install -y --no-install-recommends \
        software-properties-common && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    	apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        libssl-dev \
        curl \
        unzip \
        cmake \
        net-tools \
        sudo \
        htop \
	    nano
RUN apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    	apt-get install -y --no-install-recommends nsight-systems-2022.4.2

# To profile with the Torch Profiler, uncomment below :
# ADD https://api.github.com/repos/gabrieleoliaro/fairseq/git/refs/heads/moe ff_version.json
# RUN git clone -b moe https://github.com/gabrieleoliaro/fairseq && cd fairseq && pip install --editable ./
# To profile with the nvprof profiler, uncomment below :
ADD https://api.github.com/repos/facebookresearch/fairseq/git/refs/heads/moe ff_version.json
RUN git clone -b moe https://github.com/facebookresearch/fairseq && cd fairseq && pip install --editable ./

RUN pip install boto3 iopath fairscale

