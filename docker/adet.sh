#!/usr/bin/env bash
# Build and run the AdelaiDet Docker image (for training BoxInst on PhenoBench).
#
#   docker/adet.sh build          # build the image
#   docker/adet.sh run            # drop into an interactive shell (default)
#   docker/adet.sh run <cmd...>   # run a command in the container, e.g.:
#       docker/adet.sh run python tools/train_net.py \
#           --config-file configs/BoxInst/phenobench_R_50_1x.yaml --num-gpus 8
#
# The runtime flags below (GPU access, shm size, ulimits, dataset/weight/output
# mounts) are the reason this script exists -- they cannot live in the Dockerfile.
set -euo pipefail

IMAGE="adet"
# Repo root = parent of this script's directory, regardless of where it's called.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmd="${1:-run}"
shift || true

case "$cmd" in
	build)
		docker build \
			-f "$ROOT/docker/adet.Dockerfile" \
			--build-arg "USER_ID=$(id -u)" \
			-t "$IMAGE" \
			"$ROOT"
		;;
	run)
		# Create host dirs so the bind-mounts don't materialize as root-owned.
		mkdir -p "$ROOT/datasets/phenobench/annotations" \
			"$ROOT/pretrained_models" "$ROOT/output"
		# Source of the phenobench-yolo dataset (read from outside the repo).
		# Override with PHENOBENCH_YOLO=... if it lives elsewhere. The whole root
		# (images/ + labels/ + classes.txt) is mounted so the converter can run
		# in-container; its images/ subdir is also mounted where the registered
		# phenobench_{train,val} datasets expect them.
		PHENOBENCH_YOLO="${PHENOBENCH_YOLO:-/home/ava/data/phenobench-yolo}"
		# tools/ and configs/ are bind-mounted so edits to the (pure-python)
		# training script and config files take effect live, with no rebuild.
		# adet/ stays baked into the image (it holds the compiled adet._C.so).
		docker run --rm -it \
			--gpus all \
			--shm-size=16g \
			--ulimit memlock=-1 --ulimit stack=67108864 \
			-v "$ROOT/datasets:/home/appuser/AdelaiDet/datasets" \
			-v "$ROOT/pretrained_models:/home/appuser/AdelaiDet/pretrained_models" \
			-v "$ROOT/output:/home/appuser/AdelaiDet/output" \
			-v "$ROOT/tools:/home/appuser/AdelaiDet/tools" \
			-v "$ROOT/configs:/home/appuser/AdelaiDet/configs" \
			-v "$PHENOBENCH_YOLO:/home/appuser/AdelaiDet/datasets/phenobench-yolo:ro" \
			-v "$PHENOBENCH_YOLO/images:/home/appuser/AdelaiDet/datasets/phenobench/images:ro" \
			"$IMAGE" \
			"${@:-/bin/bash}"
		;;
	*)
		echo "usage: $0 {build|run [cmd...]}" >&2
		exit 2
		;;
esac
