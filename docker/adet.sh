#!/usr/bin/env bash
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
		# Leading args that are existing directories are datasets to mount at
		# datasets/<basename> (read-only). Everything after them is the command.
		DATASET_MOUNTS=()
		declare -A LINK_DIRS=()
		while [ $# -gt 0 ] && [ -d "$1" ]; do
			ds_path="$(cd "$1" && pwd)"       # absolute, symlinks resolved
			ds_name="$(basename "$ds_path")"
			# Host mountpoint must exist first so the nested bind-mount (inside
			# the datasets/ mount) is owned by the user, not root.
			mkdir -p "$ROOT/datasets/$ds_name"
			DATASET_MOUNTS+=(-v "$ds_path:/home/appuser/AdelaiDet/datasets/$ds_name:ro")
			echo "mounting dataset '$ds_name' from $ds_path" >&2
			while IFS= read -r link_dir; do
				[ -n "$link_dir" ] && LINK_DIRS["$link_dir"]=1
			done < <(find "$ds_path" -type l -exec readlink -f {} + 2>/dev/null | xargs -r -n1 dirname | sort -u)
			shift
		done
		for link_dir in ${LINK_DIRS[@]+"${!LINK_DIRS[@]}"}; do
			DATASET_MOUNTS+=(-v "$link_dir:$link_dir:ro")
			echo "mounting symlink target dir $link_dir" >&2
		done
		mkdir -p "$ROOT/pretrained_models" "$ROOT/output"
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
			${DATASET_MOUNTS[@]+"${DATASET_MOUNTS[@]}"} \
			"$IMAGE" \
			"${@:-/bin/bash}"
		;;
	*)
		echo "usage: $0 {build|run [cmd...]}" >&2
		exit 2
		;;
esac
