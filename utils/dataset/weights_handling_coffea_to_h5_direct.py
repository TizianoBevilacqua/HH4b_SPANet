import numpy as np
from collections import defaultdict
import pyarrow


def compute_weight_mask(w, region, args):
    """Return (mask, applied, threshold) for the high-weight filter, given the current CLI args.
    Shared between the class-balancing pre-pass and the main writing loop so both agree
    on exactly which events are considered 'kept'.
    """
    N = len(w)
    apply_filter = args.remove_high_weights and (
        "post" in region or args.all_cat_weight_filter
    )
    if not apply_filter:
        return np.ones(N, dtype=bool), False, None

    threshold = (
        args.weight_threshold
        if args.weight_threshold is not None
        else args.weight_threshold_factor * np.median(np.abs(w))
    )

    return np.abs(w) < threshold, True, threshold


def compute_sample_weight_sums(cols, regions, class_labels, weight_name, args, dataset_to_class_index_fn):
    """Pre-pass over all datasets to compute sum(|weight|) per individual (skey, dataset)
    sample, after applying the same high-weight filtering that will be used when actually
    writing the h5 file. Used by --balance-weights to equalize weights across classes and/or
    samples instead of relying on the (potentially unreliable) sum_genweights metadata.

    Returns
    -------
    sample_abs_sum : dict[(skey, dataset)] -> sum(|weight|)
    class_idx_of_sample : dict[(skey, dataset)] -> class_idx
    """
    sample_abs_sum = {}
    class_idx_of_sample = {}

    for skey in cols:
        class_idx = dataset_to_class_index_fn(skey, class_labels)
        region = regions[class_idx]

        for dataset in cols[skey]:
            if region not in cols[skey][dataset]:
                raise ValueError(
                    f"Region '{region}' not found for dataset '{skey}' dataset '{dataset}'"
                )

            if args.novars:
                payload = cols[skey][dataset][region]
            else:
                payload = cols[skey][dataset][region]["nominal"]

            if type(payload) == pyarrow._dataset.FileSystemDataset:
                w = np.array(payload.to_table()[weight_name])
            else:
                w = payload[weight_name].value

            weight_mask, _, _ = compute_weight_mask(w, region, args)
            sample_abs_sum[(skey, dataset)] = float(np.sum(np.abs(w[weight_mask])))
            class_idx_of_sample[(skey, dataset)] = class_idx

    return sample_abs_sum, class_idx_of_sample


def compute_weight_norm_map(cols, regions, class_labels, weight_name, mode, sample_scope, neg_weight_treatment, args, dataset_to_class_index_fn):
    """Build the per-(skey, dataset) normalization divisor for --balance-weights.

    mode == "class": every class ends up with sum(|weight|) == 1, aggregated over all of
        its samples.
    mode == "sample", sample_scope == "global": every sample independently ends up with
        sum(|weight|) == 1, regardless of class.
    mode == "sample", sample_scope == "within-class": every sample is individually balanced
        to its siblings, and each class's aggregate additionally ends up with sum(|weight|)
        == 1 (split evenly across however many samples are in that class).
    """
    sample_abs_sum, class_idx_of_sample = compute_sample_weight_sums(
        cols, regions, class_labels, weight_name, args, dataset_to_class_index_fn
    )

    class_abs_sum = defaultdict(float)
    n_samples_per_class = defaultdict(int)
    for key, total in sample_abs_sum.items():
        class_idx = class_idx_of_sample[key]
        class_abs_sum[class_idx] += total
        n_samples_per_class[class_idx] += 1

    norm_map = {}
    for key, sample_total in sample_abs_sum.items():
        class_idx = class_idx_of_sample[key]
        if mode == "class":
            norm_map[key] = class_abs_sum[class_idx]
        elif sample_scope == "global":
            norm_map[key] = sample_total
        elif sample_scope == "custom":
            if "ZZ" in key[0] or "GluGlutoHHto4B_kl-1p00_kt-1p00" in key[0]:
                print(f"Custom sample scope: doubling normalization for sample {key} (class {class_idx})")
                norm_map[key] = sample_total / 2.0
            else:
                norm_map[key] = sample_total
        else:
            norm_map[key] = sample_total * n_samples_per_class[class_idx]

    return norm_map, class_abs_sum, n_samples_per_class


def process_weights(w, region, payload, sum_genweights, dataset, weight_norm_map, skey, class_idx, args):
    """Apply high-weight filter, negative-weight treatment, and weight normalization/balancing.

    Returns (w, weight_mask, apply_weight_filter).
    """
    N = len(w)
    weight_mask, apply_weight_filter, threshold = compute_weight_mask(w, region, args)
    if apply_weight_filter:
        print()
        print("-"*80)
        print(f"Applying weight filter for dataset '{dataset}' in region '{region}'")
        if args.weight_threshold is not None:
            print(f"Weight threshold (fixed): {threshold:.4f}")
        else:
            print(f"Weight threshold (dynamic, {args.weight_threshold_factor}x median): {threshold:.4f}")
        n_removed = np.sum(~weight_mask)
        print(f"Removed {n_removed}/{N} events ({100*n_removed/N:.2f}%) with |weight| >= threshold")
        print(f"Max |weight| after cut: {np.max(np.abs(w[weight_mask])):.4f}")
        print(f"Max |weight| before cut: {np.max(np.abs(w)):.4f}")

    if args.neg_weight_treatment == "zero":
        weight_mask = weight_mask & (w > 0)
    elif args.neg_weight_treatment == "abs":
        w = np.abs(w)

    if args.norm_weights:
        print()
        print("-"*80)
        print(f"Normalizing weights for dataset '{dataset}' in region '{region}'")
        norm = sum_genweights[dataset]
        if apply_weight_filter and "events_genWeight" in payload:
            gen_w = payload["events_genWeight"]
            excluded_genw_sum = np.sum(gen_w[~weight_mask])
            norm = sum_genweights[dataset] - excluded_genw_sum
            print(
                f"sum_genweights (metadata) = {sum_genweights[dataset]:.3f}, "
                f"excluded genWeight sum = {excluded_genw_sum:.3f}, "
                f"effective norm = {norm:.3f}"
            )
            print(f"Dividing by sum_genweights = {norm:.3f}")
        elif apply_weight_filter:
            print("WARNING: 'genWeight' column not found in payload; falling back to sum_genweights from metadata")
            print(f"Dividing by sum_genweights = {norm:.3f}")
        else:
            print(f"Dividing by sum_genweights = {norm:.3f}")
        print(f"weights before norm, sum = {np.sum(w):.3f}, mean = {np.mean(w):.3f}, std = {np.std(w):.3f}")
        print(f"actual weight sample: {list(w[:10])}")
        w = w / norm
        print(f"weights after norm, sum = {np.sum(w):.8f}, mean = {np.mean(w):.20f}, std = {np.std(w):.20f}")
        print(f"weights after norm selected, sum = {np.sum(w[weight_mask]):.8f}, mean = {np.mean(w[weight_mask]):.20f}, std = {np.std(w[weight_mask]):.20f}")
        print(f"actual weight sample: {list(w[:10])}")
        print()
        print("number of events", N)
    elif args.balance_weights != "none":
        print()
        print("-"*80)
        print(f"Balancing weights for dataset '{dataset}' in region '{region}' (class {class_idx}, mode={args.balance_weights})")
        norm = weight_norm_map[(skey, dataset)]
        print(f"normalization divisor for this sample = {norm:.6f}")
        print(f"weights before norm, sum = {np.sum(w):.3f}, mean = {np.mean(w):.3f}, std = {np.std(w):.3f}")
        print(f"actual weight sample: {list(w[:10])}")
        w = w / norm
        print(f"weights after norm, sum = {np.sum(w):.8f}, mean = {np.mean(w):.20f}, std = {np.std(w):.20f}")
        print(f"weights after norm selected, sum = {np.sum(w[weight_mask]):.8f}, mean = {np.mean(w[weight_mask]):.20f}, std = {np.std(w[weight_mask]):.20f}")
        print(f"actual weight sample: {list(w[:10])}")
        print()
        print("number of events", N)
    else:
        print()
        print("-"*80)
        print("number of events", N)
        print(f"weights, sum = {np.sum(w):.3f}, mean = {np.mean(w):.3f}, std = {np.std(w):.3f}")
        print(f"weights selected, sum = {np.sum(w[weight_mask]):.3f}, mean = {np.mean(w[weight_mask]):.3f}, std = {np.std(w[weight_mask]):.3f}")
        print(f"actual weight sample: {list(w[:10])}")

    return w, weight_mask, apply_weight_filter
