#!/usr/bin/env python3
"""
Inspect the output distribution of the ResNet-50 backbone (the "first layer")
for a given AdelaiDet config, WITHOUT training anything.

The config's backbone is `build_fcos_resnet_fpn_backbone`, which is a ResNet-50
`bottom_up` followed by an FPN. This script runs a few images through only the
ResNet-50 `bottom_up` and reports the distribution (min/max/mean/std/percentiles
+ a histogram) of the activations at each output stage (res3, res4, res5).

Example (inside the Docker container, from the repo root):

    python3 tools/inspect_backbone_dist.py \
        --config-file configs/BoxInstSemanticStatic/voc_23_verdant_1box_R_50_1x.yaml \
        --num-images 8 \
        --output output/backbone_dist

    # or point it at explicit image(s) / a directory instead of the dataset:
    python3 tools/inspect_backbone_dist.py \
        --config-file configs/BoxInstSemanticStatic/voc_23_verdant_1box_R_50_1x.yaml \
        --input /home/ava/data/VOC_23_verdant_1box/images/test
"""
import argparse
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.modeling import build_model

from adet.config import get_cfg


# ---- dataset registration -------------------------------------------------
# Mirror the COCO-style registration done in tools/train_net.py so that
# cfg.DATASETS.TEST resolves. Only needed when reading images via the dataset
# (i.e. when --input is not given).
def _register_datasets():
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    specs = {
        "voc23_verdant_1box_train": (
            "datasets/voc23_verdant_1box/images/train",
            "datasets/voc23_verdant_1box/annotations/train.json",
        ),
        "voc23_verdant_1box_val": (
            "datasets/voc23_verdant_1box/images/val",
            "datasets/voc23_verdant_1box/annotations/val.json",
        ),
        "voc23_verdant_1box_test": (
            "datasets/voc23_verdant_1box/images/test",
            "datasets/voc23_verdant_1box/annotations/test.json",
        ),
    }
    for name, (img_root, json_file) in specs.items():
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {"thing_classes": ["bird", "boat"]}, json_file, img_root)


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_image_paths(args, cfg):
    """Return a list of image file paths to run through the backbone."""
    if args.input:
        paths = []
        for pattern in args.input:
            if os.path.isdir(pattern):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    paths.extend(glob.glob(os.path.join(pattern, ext)))
            else:
                paths.extend(glob.glob(pattern))
        return sorted(paths)[: args.num_images]

    # Otherwise pull from the registered test dataset.
    _register_datasets()
    from detectron2.data import DatasetCatalog

    dataset_name = cfg.DATASETS.TEST[0]
    records = DatasetCatalog.get(dataset_name)
    return [r["file_name"] for r in records[: args.num_images]]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default="configs/BoxInstSemanticStatic/voc_23_verdant_1box_R_50_1x.yaml",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Image file(s), glob(s), or a directory. If omitted, uses cfg.DATASETS.TEST.",
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--output",
        default="output/backbone_dist",
        help="Directory to write the histogram figure(s) to.",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Extra config overrides, e.g. MODEL.WEIGHTS path.pth",
    )
    args = parser.parse_args()

    cfg = setup_cfg(args)
    os.makedirs(args.output, exist_ok=True)

    # Build the full model so backbone weights/structure match the config, then
    # load the checkpoint (ImageNet-pretrained R-50 by default per the config).
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print(f"Loaded weights: {cfg.MODEL.WEIGHTS}")

    # The ResNet-50 is the backbone's bottom_up module (FPN sits on top of it).
    resnet = model.backbone.bottom_up
    device = model.device

    # Same normalization the model applies in forward(): (x - pixel_mean)/pixel_std.
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(3, 1, 1)

    # Test-time resize (shortest edge), matching inference preprocessing.
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    input_format = cfg.INPUT.FORMAT  # "BGR" for detectron2 defaults

    paths = load_image_paths(args, cfg)
    if not paths:
        raise SystemExit("No input images found. Pass --input or check the dataset paths.")
    print(f"Running {len(paths)} image(s) through ResNet-50 bottom_up...")

    # Accumulate activations per stage across all images.
    out_features = list(resnet._out_features)  # e.g. ["res3", "res4", "res5"]
    accum = {k: [] for k in out_features}

    for path in paths:
        img = utils.read_image(path, format=input_format)  # HxWxC
        img = aug.get_transform(img).apply_image(img)
        tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).to(device)
        tensor = (tensor - pixel_mean) / pixel_std
        feats = resnet(tensor.unsqueeze(0))  # dict: stage -> (1, C, H, W)
        for k in out_features:
            accum[k].append(feats[k].flatten().float().cpu())

    # Report + plot.
    n_stages = len(out_features)
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 4), squeeze=False)
    print("\n=== ResNet-50 output distribution ===")
    for ax, k in zip(axes[0], out_features):
        vals = torch.cat(accum[k]).numpy()
        pcts = np.percentile(vals, [0, 1, 25, 50, 75, 99, 100])
        print(
            f"\n[{k}] shape/level activations: n={vals.size:,}\n"
            f"  mean={vals.mean():.4f}  std={vals.std():.4f}\n"
            f"  min={pcts[0]:.4f}  p1={pcts[1]:.4f}  p25={pcts[2]:.4f}  "
            f"median={pcts[3]:.4f}  p75={pcts[4]:.4f}  p99={pcts[5]:.4f}  max={pcts[6]:.4f}\n"
            f"  frac==0 (post-ReLU): {(vals == 0).mean():.3f}"
        )
        ax.hist(vals, bins=100, color="steelblue")
        ax.set_title(f"{k}  (mean={vals.mean():.2f}, std={vals.std():.2f})")
        ax.set_xlabel("activation value")
        ax.set_ylabel("count")
        ax.set_yscale("log")

    fig.suptitle("ResNet-50 backbone output distribution")
    fig.tight_layout()
    out_path = os.path.join(args.output, "resnet50_output_hist.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved histogram to {out_path}")


if __name__ == "__main__":
    main()
