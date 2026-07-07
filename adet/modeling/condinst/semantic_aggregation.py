import torch
from torch import nn


def aggregate_by_index_mean(values, index, num_bins):
    """
    values: (n, D) float tensor
    index:  (n,) long tensor with entries in [0, num_bins)
    Returns (pooled: (num_bins, D), counts: (num_bins,)) -- counts are float,
    row counts contributing to each bin. Bins with count 0 come back as 0.
    """
    D = values.size(1)
    pooled = values.new_zeros(num_bins, D)
    pooled.index_add_(0, index, values)
    counts = values.new_zeros(num_bins)
    counts.index_add_(0, index, values.new_ones(values.size(0)))
    pooled = pooled / counts.clamp(min=1.0).unsqueeze(1)
    return pooled, counts


def aggregate_controller_params(mask_head_params, slot_inds, num_slots, gt_inds=None, mode="per_instance"):
    """
    mask_head_params: (n, num_gen_params)
    slot_inds:        (n,) long, slot id = im_ind * num_classes + class_id, one per row
    gt_inds:          (n,) long, GT-instance id per row -- required for mode="per_instance"

    mode="per_instance" does a two-stage uniform mean (rows -> unique GT
    instance, then instance -> slot), so a GT instance matched by many FCOS
    locations doesn't dominate the pooled vector for its slot.
    mode="per_location" does a single-stage mean directly over rows -- correct
    when rows are already one-per-instance (e.g. post-NMS test-time detections).

    Returns (pooled: (num_slots, num_gen_params), instance_counts: (num_slots,)).
    """
    if mode == "per_location" or gt_inds is None:
        return aggregate_by_index_mean(mask_head_params, slot_inds, num_slots)

    unique_gt_inds, inverse = torch.unique(gt_inds, return_inverse=True)
    per_instance_params, _ = aggregate_by_index_mean(mask_head_params, inverse, unique_gt_inds.numel())

    per_instance_slot = slot_inds.new_zeros(unique_gt_inds.numel())
    per_instance_slot[inverse] = slot_inds  # all rows sharing a gt_ind share one slot, by construction

    return aggregate_by_index_mean(per_instance_params, per_instance_slot, num_slots)


class PriorControllerEmbedding(nn.Module):
    """
    Learnable per-class fallback controller vector. Added to every slot's
    pooled controller vector -- since pooling over zero rows yields exactly
    0, a slot with no matched/detected instances of its class collapses to
    this prior alone, giving it a sensible, trainable default.
    """

    def __init__(self, num_classes, num_gen_params, init_std=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, num_gen_params))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, class_ids):
        return self.weight[class_ids]


def group_rows_by_index(values, index, num_bins):
    """
    values: (n, D) tensor
    index:  (n,) long tensor with entries in [0, num_bins)
    Returns a list of length num_bins, each a (k_i, D) tensor of the rows
    whose index equals that bin (k_i may be 0).
    """
    groups = [values.new_zeros((0, values.size(1))) for _ in range(num_bins)]
    for b in range(num_bins):
        sel = index == b
        if sel.any():
            groups[b] = values[sel]
    return groups


def compute_nearest_center_offsets(slot_centers, locations, distance_norm, mode="offset", empty_fill=10000.0):
    """
    slot_centers:  list[Tensor(k_i, 2)], length num_slots (k_i may be 0)
    locations:     (HW, 2) pixel grid, shared across slots
    distance_norm: scalar or (num_slots, 1, 1) broadcastable tensor
    mode:          "offset" -> 2-channel (dx, dy) to the nearest center
                   "scalar" -> 1-channel Euclidean distance to the nearest center
    empty_fill:    pre-normalization fill value used for slots with zero centers
                   (a large constant, not 0 -- see EMPTY_SLOT_DISTANCE_FILL doc)

    Returns (num_slots, n_ch, HW), already divided by distance_norm.
    """
    num_slots = len(slot_centers)
    HW = locations.size(0)
    n_ch = 2 if mode == "offset" else 1
    out = locations.new_full((num_slots, n_ch, HW), float(empty_fill))

    for i, centers in enumerate(slot_centers):
        if centers.numel() == 0:
            continue
        diffs = centers.reshape(-1, 1, 2) - locations.reshape(1, HW, 2)  # (k, HW, 2)
        dists = diffs.norm(dim=2)  # (k, HW)
        nearest = dists.argmin(dim=0)  # (HW,)
        if mode == "offset":
            chosen = diffs[nearest, torch.arange(HW, device=diffs.device)]  # (HW, 2)
            out[i] = chosen.permute(1, 0)
        else:
            out[i, 0] = dists.min(dim=0)[0]

    return out / distance_norm


def union_bitmasks_by_class(gt_bitmasks, gt_classes, num_classes):
    """
    gt_bitmasks: (k, Hm, Wm) per-instance bitmasks for one image
    gt_classes:  (k,) class id per instance
    Returns (num_classes, Hm, Wm): per-class union (OR) of instance bitmasks.
    """
    Hm, Wm = gt_bitmasks.shape[-2:]
    target = gt_bitmasks.new_zeros(num_classes, Hm, Wm)
    for c in range(num_classes):
        m = gt_classes == c
        if m.any():
            target[c] = gt_bitmasks[m].amax(dim=0)
    return target
