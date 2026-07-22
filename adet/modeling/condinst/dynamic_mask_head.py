import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear


def _add_best_neighbor_positives(positives, foreground, cls_mask):
    from adet.modeling.condinst.condinst import unfold_wo_center

    dilation = 1
    kernel_size = 3
    neighbor_vals = unfold_wo_center(foreground, kernel_size=kernel_size, dilation=dilation)  # B, C, K*K-1, H, W
    neighbor_in_box = unfold_wo_center(cls_mask, kernel_size=kernel_size, dilation=dilation) > 0  # B, C, K*K-1, H, W

    eligible = neighbor_in_box  # rank only in-box neighbours
    neighbor_vals = neighbor_vals.masked_fill(~eligible, float('-inf'))  # rank only eligible neighbours
    best = neighbor_vals.argmax(dim=2)  # B, C, H, W index of highest eligible neighbour in 0..K*K-2
    sel = F.one_hot(best, num_classes=neighbor_vals.shape[2]).permute(0, 1, 4, 2, 3).to(positives.dtype)
    sel = sel * positives.unsqueeze(2) * eligible.to(positives.dtype)

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    size = kernel_size ** 2
    b, c, _, h, w = sel.shape
    sel_full = torch.zeros(b, c, size, h, w, device=sel.device, dtype=sel.dtype)
    sel_full[:, :, :size // 2] = sel[:, :, :size // 2]
    sel_full[:, :, size // 2 + 1:] = sel[:, :, size // 2:]
    folded = F.fold(
        sel_full.reshape(b, c * size, h * w), output_size=(h, w),
        kernel_size=kernel_size, padding=padding, dilation=dilation,
    )
    neighbor_positives = (folded > 0).to(positives.dtype)
    return (positives + neighbor_positives > 0).to(positives.dtype)


def compute_project_term(mask_scores, gt_bitmasks, inflate=False):
    if not inflate:
        mask_losses_y = dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean()

    # Inflate version
    cls_mask = (gt_bitmasks > 0).float()
    foreground = mask_scores.detach() * cls_mask
    col_max = foreground.amax(dim=2, keepdim=True)
    row_max = foreground.amax(dim=3, keepdim=True)
    normalizer = torch.minimum(col_max, row_max) * cls_mask
    positives = (foreground > (0.95 * normalizer)).float() * cls_mask
    positives = _add_best_neighbor_positives(positives, foreground, cls_mask)

    weights = torch.maximum(positives, 1.0 - cls_mask)
    eps = 1
    numerator = 2 * (mask_scores * positives).sum(dim=(2, 3))
    denominator = (mask_scores * weights).pow(2).sum(dim=(2, 3)) + positives.sum(dim=(2, 3))
    loss = 1. - (numerator + eps) / (denominator + eps)
    return loss.mean()


def compute_max_labeling(mask_logits, gt_bitmasks, inflate=False):
    # batch_size = gt_bitmasks.shape[0]
    cls_mask = (gt_bitmasks > 0).float()
    foreground = mask_logits.detach().sigmoid() * cls_mask
    col_max = foreground.amax(dim=2, keepdim=True)
    row_max = foreground.amax(dim=3, keepdim=True)

    normalizer = torch.minimum(col_max, row_max) * cls_mask
    positives = (foreground > (0.95 * normalizer)).float() * cls_mask
    if inflate:
        positives = _add_best_neighbor_positives(positives, foreground, cls_mask)
    col_box_height = cls_mask.sum(dim=2, keepdim=True)
    col_pos_count = positives.sum(dim=2, keepdim=True)
    pos_weights = positives * (col_box_height / col_pos_count.clamp(min=1.0))

    weights = torch.maximum(pos_weights, 1.0 - cls_mask)
    loss = F.binary_cross_entropy_with_logits(mask_logits, cls_mask, reduction="none") * weights
    loss = loss.mean()
    return loss


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def compute_pairwise_l1_term(mask_logits, pairwise_size, pairwise_dilation):
    """L1 pairwise term: |sigmoid(logit_i) - sigmoid(logit_j)| per neighbour.

    Same (N, K, Hm, Wm) output shape as compute_pairwise_term, so the downstream
    color/box weighting is identical. Unlike the log-prob term it is stable from
    iteration 0 and therefore needs no warmup.
    """
    assert mask_logits.dim() == 4

    from adet.modeling.condinst.condinst import unfold_wo_center
    probs = mask_logits.sigmoid()
    probs_unfold = unfold_wo_center(
        probs, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    pairwise_diff = torch.abs(probs[:, :, None] - probs_unfold)

    return pairwise_diff[:, 0]


def compute_pairwise_loss(mask_logits, weights, loss_type,
                          pairwise_size, pairwise_dilation, warmup_iters, cur_iter):
    """Weighted, normalized BoxInst pairwise loss for the selected term.

    loss_type "l1"  -> L1 term, no warmup.
    otherwise       -> original log-prob term, warmed up over warmup_iters.
    `weights` is the caller-computed color-similarity * in-box mask (already
    accounts for the per-instance vs. per-class layout differences between heads).
    """
    if loss_type == "l1":
        pairwise_losses = compute_pairwise_l1_term(
            mask_logits, pairwise_size, pairwise_dilation
        )
    else:
        pairwise_losses = compute_pairwise_term(
            mask_logits, pairwise_size, pairwise_dilation
        )

    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

    if loss_type == "l1":
        return loss_pairwise

    warmup_factor = min(cur_iter / float(warmup_iters), 1.0)
    return loss_pairwise * warmup_factor


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


def mask_heads_forward(features, weights, biases, num_insts):
    '''
    :param features
    :param weights: [w0, w1, ...]
    :param bias: [b0, b1, ...]
    :return:
    '''
    assert features.dim() == 4
    n_layers = len(weights)
    x = features
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = F.conv2d(
            x, w, bias=b,
            stride=1, padding=0,
            groups=num_insts
        )
        if i < n_layers - 1:
            x = F.relu(x)
    return x


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.pairwise_loss_type = cfg.MODEL.BOXINST.PAIRWISE.LOSS_TYPE
        self.projection_inflation = cfg.MODEL.BOXINST.PROJECTION_INFLATION

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        return mask_heads_forward(features, weights, biases, num_insts)

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(
                        mask_scores, gt_bitmasks,
                        inflate=self.projection_inflation == "inflation",
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = compute_pairwise_loss(
                        mask_logits, weights, self.pairwise_loss_type,
                        self.pairwise_size, self.pairwise_dilation,
                        self._warmup_iters, self._iter.item(),
                    )

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances
