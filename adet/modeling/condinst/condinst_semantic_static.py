from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .condinst_semantic import CondInstSemantic
from .semantic_static_head import build_semantic_seg_head
from .semantic_aggregation import union_bitmasks_by_class

__all__ = ["BoxInstSemanticStatic"]


@META_ARCH_REGISTRY.register()
class BoxInstSemanticStatic(CondInstSemantic):
    """
    Static (non-conditional) counterpart of CondInstSemantic.

    Same FCOS detector and BoxInst box-supervised losses, and the same dense
    (num_classes, H, W) semantic output + postprocessing (inherited from
    CondInstSemantic). The only difference is the mask head: instead of a
    controller dynamically generating the filters of a per-(image, class)
    dynamic conv, this uses an ordinary static conv stack (SemanticSegHead)
    that maps the mask branch features straight to per-class logits. There is
    therefore no controller and no dynamically-generated filters.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # The static segmentation head generates no filters, so the base
        # class's controller is unused. Drop it (and skip feeding it to FCOS)
        # so no dynamic conditioning path exists at all.
        self.controller = None

    def _build_mask_head(self, cfg):
        return build_semantic_seg_head(cfg)

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        num_classes = self.mask_head.num_classes
        gt_class_bitmasks = [
            union_bitmasks_by_class(g.gt_bitmasks, g.gt_classes, num_classes)
            for g in gt_instances
        ]
        # `image_color_similarity` is per-image data, merely replicated once
        # per GT instance for indexing convenience by the (unchanged)
        # add_bitmasks_from_boxes -- any row is the same, so take the first.
        image_color_similarity = [g.image_color_similarity[0:1] for g in gt_instances]

        return self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            gt_class_bitmasks=gt_class_bitmasks, image_color_similarity=image_color_similarity,
        )

    def _forward_mask_heads_test(self, proposals, mask_feats):
        return self.mask_head(mask_feats, self.mask_branch.out_stride)
