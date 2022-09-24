FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV LANG C.UTF-8

RUN git clone -b moe https://github.com/pytorch/fairseq && cd fairseq && pip install --editable ./

RUN conda install boto3 iopath fairscale

