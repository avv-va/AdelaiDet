import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/annotations/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/annotations/rects_val.json"),
    "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/annotations/rects_test.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/annotations/abcnet_art_train.json"), 
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/annotations/abcnet_lsvt_train.json"), 
    "chnsyn_train": ("ChnSyn/syn_130k_images", "ChnSyn/annotations/chn_syntext.json"),
    "icdar2013_train": ("icdar2013/train_images", "icdar2013/ic13_train.json"),
    "icdar2015_train": ("icdar2015/train_images", "icdar2015/ic15_train.json"),
    "icdar2015_test": ("icdar2015/test_images", "icdar2015/ic15_test.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ---------------------------------------------------------------------------
# Each split is registered as "<name>_<split>". Stable logical
# "<name>_train"/"<name>_val"/"<name>_test" aliases are also registered (falling
# back to whatever single split exists).

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Distinct colors for the semantic (stuff) visualization; cycled if a
# dataset has more classes than entries.
_GENERIC_PALETTE = [
    [0, 200, 0], [255, 60, 60], [0, 130, 255], [255, 150, 0],
    [180, 0, 255], [0, 200, 200], [255, 0, 150], [150, 150, 0],
    [120, 80, 0], [0, 100, 60],
]


def _read_classes(classes_txt):
    with open(classes_txt) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _load_coco_dict(json_file):
    """Return the parsed json if it looks like a COCO detection file, else None."""
    try:
        with open(json_file) as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return None
    if isinstance(data, dict) and data.get("images"):
        return data
    return None


def _image_root_for(dataset_dir, split, coco):
    # A file_name carrying a path separator is relative to the dataset dir
    # (downloaded layout); a bare basename lives under images/<split> (converted).
    file_name = coco["images"][0].get("file_name", "")
    if "/" in file_name or os.sep in file_name:
        return dataset_dir
    cand = os.path.join(dataset_dir, "images", split)
    return cand if os.path.isdir(cand) else os.path.join(dataset_dir, "images")


def _find_image(images_dir, stem):
    for ext in _IMG_EXTS:
        cand = os.path.join(images_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def _load_yolo_split(images_dir, labels_dir):
    """Build detectron2 dataset dicts directly from YOLO labels (no COCO json).

    YOLO labels are `cls cx cy w h [...]` normalized to [0, 1]; only the first
    five fields are used. Each box becomes a rectangle-polygon segmentation so
    the mask loader is satisfied -- BoxInst/CondInst derive real masks from the
    boxes. In-memory equivalent of tools/convert_phenobench_to_coco.py.
    """
    import logging

    from PIL import Image
    from detectron2.structures import BoxMode

    logger = logging.getLogger(__name__)
    dicts = []
    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    for img_id, lf in enumerate(label_files):
        stem = os.path.splitext(lf)[0]
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except OSError as e:  # unreadable image (e.g. truncated cache tile)
            logger.warning("skipping unreadable image %s: %s", img_path, e)
            continue

        annos = []
        with open(os.path.join(labels_dir, lf)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue  # need at least cls cx cy w h
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
                # normalized center/size -> absolute top-left xywh, clipped.
                x0 = max(0.0, min((cx - w / 2.0) * W, W))
                y0 = max(0.0, min((cy - h / 2.0) * H, H))
                x1 = max(0.0, min((cx + w / 2.0) * W, W))
                y1 = max(0.0, min((cy + h / 2.0) * H, H))
                bw, bh = x1 - x0, y1 - y0
                if bw <= 1.0 or bh <= 1.0:
                    continue  # degenerate box
                annos.append(
                    {
                        "bbox": [x0, y0, bw, bh],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": cls,  # YOLO is already 0-indexed
                        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                        "iscrowd": 0,
                    }
                )
        dicts.append(
            {
                "file_name": img_path,
                "image_id": img_id,
                "height": H,
                "width": W,
                "annotations": annos,
            }
        )
    return dicts


def _discover_splits(dataset_dir):
    """Return {split: spec} for a dataset dir, where spec is one of:

        ("coco", json_file, image_root)     a COCO instances json
        ("yolo", images_dir, labels_dir)    raw YOLO labels

    COCO jsons take precedence when a split has both.
    """
    splits = {}

    # raw YOLO: images/<split> + labels/<split>
    images_root = os.path.join(dataset_dir, "images")
    labels_root = os.path.join(dataset_dir, "labels")
    if os.path.isdir(images_root) and os.path.isdir(labels_root):
        for split in sorted(os.listdir(labels_root)):
            img_dir = os.path.join(images_root, split)
            lbl_dir = os.path.join(labels_root, split)
            if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
                splits[split] = ("yolo", img_dir, lbl_dir)

    # COCO jsons: <dir>/annotations/*.json and <dir>/*.json (override YOLO)
    candidates = []
    ann_dir = os.path.join(dataset_dir, "annotations")
    if os.path.isdir(ann_dir):
        candidates += [os.path.join(ann_dir, f) for f in os.listdir(ann_dir)]
    candidates += [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    for jf in sorted(candidates):
        if not jf.endswith(".json"):
            continue
        coco = _load_coco_dict(jf)
        if coco is None:
            continue
        split = os.path.splitext(os.path.basename(jf))[0]
        splits[split] = ("coco", jf, _image_root_for(dataset_dir, split, coco))
    return splits


def register_generic_datasets(root="datasets"):
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        dataset_dir = os.path.join(root, name)
        classes_txt = os.path.join(dataset_dir, "classes.txt")
        if not os.path.isfile(classes_txt):
            continue
        thing_classes = _read_classes(classes_txt)
        splits = _discover_splits(dataset_dir)
        if not thing_classes or not splits:
            continue

        metadata = {
            "thing_classes": thing_classes,
            "stuff_classes": thing_classes,
            "stuff_colors": [
                _GENERIC_PALETTE[i % len(_GENERIC_PALETTE)]
                for i in range(len(thing_classes))
            ],
        }

        def _register(reg_name, split):
            if reg_name in DatasetCatalog.list():
                return
            spec = splits[split]
            if spec[0] == "coco":
                _, json_file, image_root = spec
                register_coco_instances(reg_name, dict(metadata), json_file, image_root)
            else:
                _, img_dir, lbl_dir = spec
                DatasetCatalog.register(
                    reg_name, lambda: _load_yolo_split(img_dir, lbl_dir)
                )
                MetadataCatalog.get(reg_name).set(
                    evaluator_type="coco", image_root=img_dir, **dict(metadata)
                )

        for split in splits:
            _register("{}_{}".format(name, split), split)

        # Stable aliases so configs need not know the on-disk split name.
        pick = lambda *prefs: next(
            (s for s in prefs if s in splits), sorted(splits)[0]
        )
        _register("{}_train".format(name), pick("train"))
        _register("{}_val".format(name), pick("val", "test", "train"))
        _register("{}_test".format(name), pick("test", "val", "train"))


register_all_coco()
register_generic_datasets()
