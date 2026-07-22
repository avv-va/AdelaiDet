import torch
from torch import nn

from .dynamic_mask_head import (
    dice_coefficient, compute_max_labeling, compute_pairwise_loss,
)
from adet.utils.comm import aligned_bilinear


def build_semantic_seg_head(cfg):
    return SemanticSegHead(cfg)


class SemanticSegHead(nn.Module):
    """
    Static (non-conditional) per-class segmentation head -- the drop-in
    replacement for SemanticDynamicMaskHead that removes the conditional
    convolution mechanism entirely.

    Instead of a controller dynamically generating the filters of a per-slot
    conv stack, this head owns an ordinary, image-independent 1x1 conv stack
    whose last layer has num_classes output channels. It maps the mask branch
    features F_mask (N, C_mask, H, W) straight to per-class logits
    (N, num_classes, H, W). There is no controller, no relative coordinates,
    no prior embedding and no slot pooling.

    The BoxInst box-projection (max-labeling) + pairwise losses -- or the
    fully-supervised dice loss -- are applied per class exactly as in the
    dynamic version: the (N, num_classes, H, W) logits are flattened to
    (N * num_classes, 1, H, W) so the shared loss helpers see one binary map
    per (image, class), matching the num_slots layout of the dynamic head.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        # boxinst configs (same generic weak-supervision hyperparameters used
        # by the instance and dynamic-semantic heads)
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.pairwise_loss_type = cfg.MODEL.BOXINST.PAIRWISE.LOSS_TYPE
        self.projection_inflation = cfg.MODEL.BOXINST.PROJECTION_INFLATION
        self.register_buffer("_iter", torch.zeros([1]))

        # This head generates no dynamic parameters. The attribute is read by
        # CondInst.__init__ to size its controller; CondInstSemantic discards
        # that controller, so 0 simply means "no conditional filters".
        self.num_gen_params = 0

        # Static 1x1 conv stack, mirroring the dynamic mask head's geometry
        # (same depth/width, ReLU between layers) but with fixed, shared
        # weights and num_classes output channels instead of 1.
        layers = []
        for l in range(self.num_layers):
            in_ch = self.in_channels if l == 0 else self.channels
            out_ch = self.num_classes if l == self.num_layers - 1 else self.channels
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0))
            if l < self.num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*layers)

    def _mask_logits(self, mask_feats, mask_feat_stride):
        """(N, C_mask, H, W) -> per-class logits at mask_out_stride resolution."""
        logits = self.tower(mask_feats)  # (N, num_classes, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        logits = aligned_bilinear(logits, int(mask_feat_stride / self.mask_out_stride))
        return logits

    def _forward_train(self, mask_feats, mask_feat_stride, gt_class_bitmasks, image_color_similarity):
        self._iter += 1
        num_images = mask_feats.size(0)
        num_classes = self.num_classes
        num_slots = num_images * num_classes

        logits = self._mask_logits(mask_feats, mask_feat_stride)  # (N, num_classes, Hm, Wm)
        Hm, Wm = logits.shape[-2:]

        # Flatten to one binary map per (image, class) so the shared BoxInst /
        # dice loss helpers operate exactly as in the dynamic-semantic head.
        mask_logits = logits.reshape(num_slots, 1, Hm, Wm)

        gt_bitmasks = torch.stack(gt_class_bitmasks, dim=0).reshape(num_slots, 1, Hm, Wm)
        gt_bitmasks = gt_bitmasks.to(dtype=mask_feats.dtype)
        mask_scores = mask_logits.sigmoid()

        losses = {}
        if self.boxinst_enabled:
            color_sim = torch.cat(image_color_similarity, dim=0)  # (num_images, K-1, Hm, Wm)
            color_sim = color_sim.repeat_interleave(num_classes, dim=0).to(dtype=mask_feats.dtype)

            loss_prj_term = compute_max_labeling(
                mask_logits, gt_bitmasks,
                inflate=self.projection_inflation == "inflation",
            )

            weights = (color_sim >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
            loss_pairwise = compute_pairwise_loss(
                mask_logits, weights, self.pairwise_loss_type,
                self.pairwise_size, self.pairwise_dilation,
                self._warmup_iters, self._iter.item(),
            )

            losses.update({"loss_prj": loss_prj_term, "loss_pairwise": loss_pairwise})
        else:
            losses["loss_mask"] = dice_coefficient(mask_scores, gt_bitmasks).mean()

        return losses

    def _forward_test(self, mask_feats, mask_feat_stride):
        logits = self._mask_logits(mask_feats, mask_feat_stride)  # (N, num_classes, Hm, Wm)
        return logits.sigmoid()

    def forward(self, mask_feats, mask_feat_stride, gt_class_bitmasks=None, image_color_similarity=None):
        if self.training:
            return self._forward_train(
                mask_feats, mask_feat_stride, gt_class_bitmasks, image_color_similarity,
            )
        return self._forward_test(mask_feats, mask_feat_stride)
