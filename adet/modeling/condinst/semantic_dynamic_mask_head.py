import torch
from torch import nn

from .dynamic_mask_head import (
    parse_dynamic_params, mask_heads_forward, dice_coefficient,
    compute_max_labeling, compute_pairwise_loss,
)
from .semantic_aggregation import (
    aggregate_by_index_mean, aggregate_controller_params, PriorControllerEmbedding,
    group_rows_by_index, compute_nearest_center_offsets,
)
from adet.utils.comm import compute_locations, aligned_bilinear


def build_semantic_dynamic_mask_head(cfg):
    return SemanticDynamicMaskHead(cfg)


def _weight_bias_nums(num_layers, channels, in_channels, coord_channels):
    weight_nums, bias_nums = [], []
    for l in range(num_layers):
        if l == 0:
            weight_nums.append((in_channels + coord_channels) * channels)
            bias_nums.append(channels)
        elif l == num_layers - 1:
            weight_nums.append(channels * 1)
            bias_nums.append(1)
        else:
            weight_nums.append(channels * channels)
            bias_nums.append(channels)
    return weight_nums, bias_nums


class SemanticDynamicMaskHead(nn.Module):
    """
    Per-class counterpart of DynamicMaskHead: instead of running one dynamic
    conv "instance" per detected object, controller vectors of all detected
    instances of the same class in an image are pooled into one vector per
    (image, class) slot, and one dynamic conv "instance" is run per slot.
    The final layer still outputs exactly 1 channel per slot -- num_classes
    output channels per image fall out of there being num_classes slots per
    image, not from widening the last layer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        self.aggregation_mode = cfg.MODEL.CONDINST_SEM.AGGREGATION
        self.distance_mode = cfg.MODEL.CONDINST_SEM.DISTANCE_MODE
        self.empty_slot_fill = cfg.MODEL.CONDINST_SEM.EMPTY_SLOT_DISTANCE_FILL

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]).float())

        # boxinst configs (reused as-is -- these are generic weak-supervision
        # hyperparameters, not instance-vs-semantic-specific)
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.pairwise_loss_type = cfg.MODEL.BOXINST.PAIRWISE.LOSS_TYPE
        self.register_buffer("_iter", torch.zeros([1]))

        if self.disable_rel_coords:
            self.coord_channels = 0
        elif self.distance_mode == "scalar":
            self.coord_channels = 1
        else:
            self.coord_channels = 2

        self.weight_nums, self.bias_nums = _weight_bias_nums(
            self.num_layers, self.channels, self.in_channels, self.coord_channels
        )
        self.num_gen_params = sum(self.weight_nums) + sum(self.bias_nums)

        self.prior_embedding = PriorControllerEmbedding(
            self.num_classes, self.num_gen_params,
            cfg.MODEL.CONDINST_SEM.PRIOR_EMBEDDING_INIT_STD,
        )

    def _compute_slot_soi(self, fpn_levels, slot_inds, num_slots):
        soi_per_row = self.sizes_of_interest[fpn_levels].unsqueeze(1)
        pooled, counts = aggregate_by_index_mean(soi_per_row, slot_inds, num_slots)
        pooled = pooled.squeeze(1)
        fallback = self.sizes_of_interest[-1]
        return torch.where(counts > 0, pooled, fallback.expand_as(pooled))

    def _aggregate_slots(self, mask_feats, pred_instances, num_images, classes_field, aggregation_mode, use_gt_inds):
        num_classes = self.num_classes
        num_slots = num_images * num_classes
        device = mask_feats.device

        slot_im_ids = torch.arange(num_images, device=device).repeat_interleave(num_classes)
        class_ids_of_slot = torch.arange(num_classes, device=device).repeat(num_images)
        prior = self.prior_embedding(class_ids_of_slot)

        if len(pred_instances) > 0:
            classes = getattr(pred_instances, classes_field)
            slot_inds = pred_instances.im_inds * num_classes + classes
            gt_inds = pred_instances.gt_inds if use_gt_inds else None

            pooled_params, _ = aggregate_controller_params(
                pred_instances.mask_head_params, slot_inds, num_slots,
                gt_inds=gt_inds, mode=aggregation_mode,
            )
            slot_soi = self._compute_slot_soi(pred_instances.fpn_levels, slot_inds, num_slots)
            slot_centers = group_rows_by_index(pred_instances.locations, slot_inds, num_slots)
        else:
            pooled_params = prior.new_zeros(num_slots, self.num_gen_params)
            slot_soi = self.sizes_of_interest[-1].expand(num_slots).clone()
            slot_centers = [prior.new_zeros((0, 2)) for _ in range(num_slots)]

        slot_params = pooled_params + prior
        return slot_im_ids, slot_params, slot_soi, slot_centers

    def _run_dynamic_conv(self, mask_feats, mask_feat_stride, slot_im_ids, slot_params, slot_centers, slot_soi):
        N, _, H, W = mask_feats.size()
        num_slots = slot_params.size(0)

        feat_part = mask_feats[slot_im_ids].reshape(num_slots, self.in_channels, H * W)

        if self.coord_channels > 0:
            locations = compute_locations(H, W, stride=mask_feat_stride, device=mask_feats.device)
            coords = compute_nearest_center_offsets(
                slot_centers, locations, slot_soi.reshape(-1, 1, 1),
                mode=self.distance_mode, empty_fill=self.empty_slot_fill,
            ).to(dtype=mask_feats.dtype)
            mask_head_inputs = torch.cat([coords, feat_part], dim=1)
        else:
            mask_head_inputs = feat_part

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(slot_params, self.channels, self.weight_nums, self.bias_nums)
        mask_logits = mask_heads_forward(mask_head_inputs, weights, biases, num_slots)
        mask_logits = mask_logits.reshape(num_slots, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        return mask_logits

    def _forward_train(self, mask_feats, mask_feat_stride, pred_instances, gt_instances,
                        gt_class_bitmasks, image_color_similarity):
        self._iter += 1
        num_images = len(gt_instances)
        num_classes = self.num_classes
        num_slots = num_images * num_classes

        slot_im_ids, slot_params, slot_soi, slot_centers = self._aggregate_slots(
            mask_feats, pred_instances, num_images, "labels", self.aggregation_mode, use_gt_inds=True,
        )

        mask_logits = self._run_dynamic_conv(
            mask_feats, mask_feat_stride, slot_im_ids, slot_params, slot_centers, slot_soi
        )  # (num_slots, 1, Hm, Wm)
        Hm, Wm = mask_logits.shape[-2:]

        gt_bitmasks = torch.stack(gt_class_bitmasks, dim=0).reshape(num_slots, 1, Hm, Wm)
        gt_bitmasks = gt_bitmasks.to(dtype=mask_feats.dtype)
        mask_scores = mask_logits.sigmoid()

        losses = {}
        if self.boxinst_enabled:
            color_sim = torch.cat(image_color_similarity, dim=0)  # (num_images, K-1, Hm, Wm)
            color_sim = color_sim.repeat_interleave(num_classes, dim=0).to(dtype=mask_feats.dtype)

            loss_prj_term = compute_max_labeling(mask_logits, gt_bitmasks)

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

    def _forward_test(self, mask_feats, mask_feat_stride, pred_instances, num_images):
        num_classes = self.num_classes

        slot_im_ids, slot_params, slot_soi, slot_centers = self._aggregate_slots(
            mask_feats, pred_instances, num_images, "pred_classes", "per_location", use_gt_inds=False,
        )

        mask_logits = self._run_dynamic_conv(
            mask_feats, mask_feat_stride, slot_im_ids, slot_params, slot_centers, slot_soi
        )
        Hm, Wm = mask_logits.shape[-2:]
        return mask_logits.sigmoid().reshape(num_images, num_classes, Hm, Wm)

    def forward(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None,
                gt_class_bitmasks=None, image_color_similarity=None, num_images=None):
        if self.training:
            return self._forward_train(
                mask_feats, mask_feat_stride, pred_instances, gt_instances,
                gt_class_bitmasks, image_color_similarity,
            )
        return self._forward_test(mask_feats, mask_feat_stride, pred_instances, num_images)
