import os

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

# Box-supervised custom datasets (phenobench, VOC_23_verdant). Boxes come from
# YOLO labels converted to COCO by tools/convert_phenobench_to_coco.py; BoxInst /
# CondInst derive masks from the boxes. `stuff_classes`/`stuff_colors` mirror the
# thing classes so the CondInstSemantic dense (num_classes, H, W) output can be
# rendered by demo.py's Visualizer.draw_sem_seg (registration is lazy, so a
# missing json only matters when the dataset is actually loaded, not at demo time).
_PREDEFINED_SPLITS_BOX_SUPERVISED = {
    "phenobench_train": ("phenobench/images/train", "phenobench/annotations/train.json"),
    "phenobench_val": ("phenobench/images/val", "phenobench/annotations/val.json"),
    "voc23_verdant_train": ("voc23_verdant/images/train", "voc23_verdant/annotations/train.json"),
    "voc23_verdant_val": ("voc23_verdant/images/val", "voc23_verdant/annotations/val.json"),
    "voc23_verdant_test": ("voc23_verdant/images/test", "voc23_verdant/annotations/test.json"),
    "voc23_verdant_1box_train": ("voc23_verdant_1box/images/train", "voc23_verdant_1box/annotations/train.json"),
    "voc23_verdant_1box_val": ("voc23_verdant_1box/images/val", "voc23_verdant_1box/annotations/val.json"),
    "voc23_verdant_1box_test": ("voc23_verdant_1box/images/test", "voc23_verdant_1box/annotations/test.json"),
}

# One metadata dict per dataset family, keyed by name prefix.
_BOX_SUPERVISED_METADATA = {
    "phenobench": {
        "thing_classes": ["crop", "weed"],
        "stuff_classes": ["crop", "weed"],
        "stuff_colors": [[0, 200, 0], [255, 60, 60]],
    },
    "voc23_verdant": {
        "thing_classes": ["bird", "boat"],
        "stuff_classes": ["bird", "boat"],
        "stuff_colors": [[0, 130, 255], [255, 150, 0]],
    },
    "voc23_verdant_1box": {
        "thing_classes": ["bird", "boat"],
        "stuff_classes": ["bird", "boat"],
        "stuff_colors": [[0, 130, 255], [255, 150, 0]],
    },
}


def _metadata_for(name):
    # Longest matching prefix wins (voc23_verdant_1box before voc23_verdant).
    prefix = max(
        (p for p in _BOX_SUPERVISED_METADATA if name.startswith(p)),
        key=len,
    )
    return _BOX_SUPERVISED_METADATA[prefix]


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
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BOX_SUPERVISED.items():
        register_coco_instances(
            key,
            _metadata_for(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()
