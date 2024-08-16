ARG AWS_REGION
# Dockerfile for Sagemaker with p4d and p4de instances
# Run `docker build` from the root of this directory.

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV PATH="/opt/ml/code:${PATH}"

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install retrie
RUN pip install -r /opt/ml/code/requirements.txt
RUN pip uninstall flash-attn -y
RUN pip install --upgrade flash-attn
# RUN pip uninstall llm-foundry -y
# RUN pip install llm-foundry==0.4.0
RUN pip install --upgrade s3fs
RUN rm /opt/ml/code/requirements.txt
RUN pip uninstall transformer_engine -y

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY . /opt/ml/code/

# RUN git clone  https://github.com/stanford-futuredata/megablocks.git
# RUN cd megablocks && pip install -e .

RUN cp /opt/ml/code/training/train.py /opt/ml/code/train.py
RUN cp /opt/ml/code/tools/eval_expdb.py /opt/ml/code/eval_expdb.py

# Prevent sagemaker from installing requirements again.
# RUN rm /opt/ml/code/setup.py
RUN rm /opt/ml/code/requirements.txt