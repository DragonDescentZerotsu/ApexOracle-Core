# hash:sha256:e57660e6489ac9281de11e29c7745fbafd8edad9ef2cc40c442db5eee4a60fba
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

COPY postInstall /
RUN bash /postInstall
