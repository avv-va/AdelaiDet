FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# 20.04 ships Python 3.8: detectron2 0.6 requires python>=3.7, and torch 1.10
# cu111 has cp38 wheels.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
		python3-opencv ca-certificates python3-dev python3-pip git wget sudo ninja-build \
	&& rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# Non-root user, mirroring the upstream Dockerfile.
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN python3 -m pip install --user --upgrade pip

# tensorboard/cmake first (cmake from apt is too old), then the pinned torch.
RUN pip install --user tensorboard cmake
RUN pip install --user torch==1.10 torchvision==0.11.1 \
	-f https://download.pytorch.org/whl/cu111/torch_stable.html

# Prebuilt detectron2 0.6 for this exact torch/cuda combo -- avoids a slow
# from-source build. AdelaiDet only needs detectron2 importable; it compiles
# its own ops (adet._C) below.
RUN pip install --user detectron2==0.6 \
	-f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# No GPU is visible during `docker build`, so force CUDA extension compilation
# and declare the target arch(es). 8.0 = A100; extra arches keep the image
# portable to other common GPUs (V100 7.0, T4/2080 7.5, 3090/A10 8.6).
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Bring in the repo (see .dockerignore for what's excluded) and build AdelaiDet
# in editable/develop mode so its CUDA ops are compiled into the image.
COPY --chown=appuser:sudo . /home/appuser/AdelaiDet
WORKDIR /home/appuser/AdelaiDet
RUN rm -rf build **/*.so && pip install --user -e .

# Pure-python runtime deps that setup.py doesn't pin correctly:
#  - Pillow<10: detectron2 0.6 uses Image.LINEAR, removed in Pillow 10.
#  - rapidfuzz<3: AdelaiDet imports rapidfuzz.string_metric, removed in v3.
#    (Only needed for the text-spotting evaluator; pinned here so the package
#    stays importable regardless.)
RUN pip install --user "Pillow<10" "rapidfuzz<3"

# Fixed model cache, mirroring upstream.
ENV FVCORE_CACHE="/tmp"
# detectron2 looks for datasets under ./datasets or $DETECTRON2_DATASETS.
ENV DETECTRON2_DATASETS="/home/appuser/AdelaiDet/datasets"
