import torch

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures.instances import Instances

from .condinst import CondInst
from .semantic_dynamic_mask_head import build_semantic_dynamic_mask_head
from .semantic_aggregation import union_bitmasks_by_class
from adet.utils.comm import aligned_bilinear

__all__ = ["CondInstSemantic"]


@META_ARCH_REGISTRY.register()
class CondInstSemantic(CondInst):
    """
    Semantic-segmentation sibling of CondInst: reuses the FCOS detector and
    the controller/dynamic-conv mechanism unchanged, but pools per-instance
    controller vectors into one vector per (image, class) slot and outputs a
    dense (num_classes, H, W) map instead of per-instance masks.

    Output channels are independent sigmoid probabilities, not a softmax
    partition -- overlapping GT boxes of different classes are structurally
    possible.
    """

    def _build_mask_head(self, cfg):
        return build_semantic_dynamic_mask_head(cfg)

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        pred_instances = proposals["instances"]
        pred_instances.mask_head_params = pred_instances.top_feats

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
            mask_feats, self.mask_branch.out_stride, pred_instances, gt_instances,
            gt_class_bitmasks=gt_class_bitmasks, image_color_similarity=image_color_similarity,
        )

    def _forward_mask_heads_test(self, proposals, mask_feats):
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        return self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances, num_images=len(proposals)
        )

    def _postprocess_inference(self, sem_seg_probs, batched_inputs, images_norm):
        padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
        mask_h, mask_w = sem_seg_probs.size()[-2:]
        factor_h = padded_im_h // mask_h
        factor_w = padded_im_w // mask_w
        assert factor_h == factor_w
        full_res_probs = aligned_bilinear(sem_seg_probs, factor_h)

        processed_results = []
        for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(full_res_probs[im_id], image_size, height, width)
            processed_results.append({"sem_seg": sem_seg_r})

        return processed_results
