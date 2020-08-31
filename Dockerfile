FROM python:3.7

ARG project_dir=/home/jovyan
WORKDIR $project_dir

# opencv-devのインストール
RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM jupyter/scipy-notebook:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt