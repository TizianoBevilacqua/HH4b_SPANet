#!/usr/bin/env python3
"""Prepare a stage-1 (pretrain) checkpoint for stage-2 (fine-tune) loading.

Builds the real stage-2 SPANet model (from --event_file/--training_file/--options_file,
exactly as spanet.train does) purely to read off each layer's expected shape, then filters
the stage-1 checkpoint's state dict down to only the keys whose shape matches. Keys that
exist in both but with a different shape (e.g. the classifier head after growing from 2 to
4 classes, or a branch's first embedding layer after adding an input variable) are dropped
so they fall back to fresh random init instead of crashing `load_state_dict`.

Must be run in your SPANet training environment (needs torch, pytorch_lightning, spanet
importable) -- not the lightweight plotting venv.

Usage:
    python prepare_finetune_checkpoint.py \
        --checkpoint stage1_run/version_0/checkpoints/best.ckpt \
        --event_file event_files/hh4b_5class.yaml \
        --training_file stage2_train.h5 \
        --validation_file stage2_val.h5 \
        --options_file options_files/HH4b/classification/.../my_options.json \
        --output stage1_filtered_for_stage2.pth

Then launch stage 2 with:
    python -m spanet.train --options_file <same options_file> \
        --event_file <same event_file> \
        --training_file stage2_train.h5 --validation_file stage2_val.h5 \
        --state_dict stage1_filtered_for_stage2.pth \
        -n <run_name> --log_dir <run_name> --gpus <n>
    # add --freeze_state_dict if you want to freeze the transferred layers at first


-----------------------------------

Prepare a checkpoint for a class-count change that SPLITS an existing class into several
new ones, rather than just adding brand-new classes (see prepare_finetune_checkpoint.py for
the plain "add a new class" case). Activated by passing a class mapping argument with --class-mapping.

Plain shape-matching (what prepare_finetune_checkpoint.py does) drops the *entire* classification
head as soon as its shape changes -- fine when you're only adding new classes, but wasteful when
you're splitting an existing, already-learned class (e.g. HH -> ggF + VBF): it would throw away
the whole HH-vs-everything-else boundary, not just the missing ggF/VBF distinction.

This script instead does row-level surgery on the classification head: each new class's
output-layer row and bias is copied from whichever old class it maps from (via --class-mapping).
Classes that split from the same old class start out agreeing with that old class's decision
boundary, with a small amount of noise added only to rows involved in a genuine split (more than
one new class mapping to the same old class), to break symmetry between them. Everything else in
the checkpoint (the shared backbone) is transferred via plain shape matching, same as
prepare_finetune_checkpoint.py.

Must be run in your SPANet training environment (needs torch, pytorch_lightning, spanet
importable) -- not the lightweight plotting venv.

Usage:
    python prepare_finetune_checkpoint_split.py \
        --checkpoint stage_4class/version_0/checkpoints/best.ckpt \
        --event_file event_files/hh4b_5class.yaml \
        --training_file stage_5class_train.h5 \
        --validation_file "" \
        --options_file options_files/.../step_3.json \
        --classification-key EVENT/class \
        --class-mapping 0=0 1=0 2=1 3=2 4=3 \
        --output stage_4class_split_for_5class.pth

--class-mapping entries are "new_index=old_index". Example above: old class 0 (HH) maps to both
new 0 (ggF) and new 1 (VBF) -- that's the split, and gets noise added. New 2 (QCD) <- old 1,
new 3 (ZH) <- old 2, new 4 (ZZ) <- old 3 are one-to-one carryovers, copied verbatim, no noise.
Any new index not listed in --class-mapping is left at random init (useful if you're also adding
a genuinely brand-new class in the same step).

Then launch training with:
    python -m spanet.train ... --state_dict stage_4class_split_for_5class.pth
    # add --freeze_state_dict if you want to freeze the transferred layers at first
"""
import argparse
import json
import sys
from collections import defaultdict

import torch


def build_reference_model(event_file, training_file, validation_file, options_file):
    from spanet import JetReconstructionModel, Options

    options = Options(event_file, training_file, validation_file)
    if options_file is not None:
        with open(options_file, "r") as f:
            options.update_options(json.load(f))

    # Building the model loads the full training dataset to size every layer
    # (num classes, num input features per branch, etc.) -- same cost as spanet.train pays.
    print("Building reference model (this loads the full training dataset)...")
    model = JetReconstructionModel(options)
    return model


def filter_checkpoint(reference_state, stage1_state, skip_keys):

    kept, dropped, new_layers = {}, [], []

    for key, tensor in stage1_state.items():
        if key in skip_keys:
            continue
        if key not in reference_state:
            dropped.append((key, tuple(tensor.shape), None))
            continue
        if tuple(tensor.shape) != tuple(reference_state[key].shape):
            dropped.append((key, tuple(tensor.shape), tuple(reference_state[key].shape)))
            continue
        kept[key] = tensor

    for key in reference_state:
        if key not in stage1_state and key not in skip_keys:
            new_layers.append((key, tuple(reference_state[key].shape)))

    return kept, dropped, new_layers


def parse_class_mapping(pairs):
    mapping = {}
    for pair in pairs:
        new_idx, old_idx = pair.split("=")
        mapping[int(new_idx)] = int(old_idx)
    return mapping


def split_classification_head(old_state, reference_state, weight_key, bias_key, class_mapping, noise_std):
    old_weight = old_state[weight_key]
    old_bias = old_state[bias_key]
    new_weight = reference_state[weight_key].clone()
    new_bias = reference_state[bias_key].clone()

    num_new_classes = new_weight.shape[0]
    num_old_classes = old_weight.shape[0]

    # Old indices targeted by more than one new index are genuine splits -> get noise.
    old_to_new = defaultdict(list)
    for new_idx, old_idx in class_mapping.items():
        old_to_new[old_idx].append(new_idx)

    filled = set()
    for new_idx, old_idx in class_mapping.items():
        if not (0 <= new_idx < num_new_classes):
            raise ValueError(f"new index {new_idx} out of range for a {num_new_classes}-class head")
        if not (0 <= old_idx < num_old_classes):
            raise ValueError(f"old index {old_idx} out of range for a {num_old_classes}-class head")

        row = old_weight[old_idx].clone()
        b = old_bias[old_idx].clone()

        if len(old_to_new[old_idx]) > 1:
            row = row + noise_std * row.std() * torch.randn_like(row)
            b = b + noise_std * old_bias.std() * torch.randn_like(b)

        new_weight[new_idx] = row
        new_bias[new_idx] = b
        filled.add(new_idx)

    unfilled = sorted(set(range(num_new_classes)) - filled)
    if unfilled:
        print(f"New class indices {unfilled} not in --class-mapping, left at random init "
              f"(fine if these are genuinely brand-new classes, not part of a split).")

    return new_weight, new_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Stage-1 (pretrain) .ckpt file")
    parser.add_argument("--event_file", required=True, help="Stage-2 event.yaml")
    parser.add_argument("--training_file", required=True, help="Stage-2 training .h5")
    parser.add_argument("--validation_file", required=True, help="Stage-2 validation .h5")
    parser.add_argument("--options_file", default=None, help="options.json (same one used for both stages)")
    parser.add_argument("--output", required=True, help="Where to write the filtered state dict (.pth)")
    parser.add_argument("--spanet-dir", default=None, help="Path to SPANet repo, if not pip-installed")
    parser.add_argument("--classification-key", default="EVENT/class",
                         help="Classification target name, matching CLASSIFICATIONS/<key> in the h5 (default: EVENT/class)")
    parser.add_argument("--class-mapping", nargs="+", default=None,
                         help="new_index=old_index pairs, e.g. 0=0 1=0 2=1 3=2 4=3. "
                              "Old indices used by more than one new index are treated as a split.")
    parser.add_argument("--noise-std", type=float, default=0.01,
                         help="Relative noise scale added to rows involved in a split, to break symmetry (default 0.01)")
    args = parser.parse_args()

    if args.spanet_dir:
        sys.path.insert(0, args.spanet_dir)

    print("Building stage-2 reference model (this loads the full stage-2 training dataset)...")
    model = build_reference_model(
        args.event_file, args.training_file, args.validation_file, args.options_file
    )
    reference_state = model.state_dict()

    raw = torch.load(args.checkpoint, map_location="cpu")
    old_state = raw["state_dict"] if "state_dict" in raw else raw

    if args.class_mapping == None:
        print("")
        print(" Running extension of classification head for stage 2 spanet models...")
        print("")
        print(f"Filtering stage-1 checkpoint: {args.checkpoint}")

        skip_keys = set()
        kept, dropped, new_layers = filter_checkpoint(reference_state, old_state, skip_keys)

        print(f"\nTransferred {len(kept)}/{len(reference_state)} tensors from stage 1.\n")

        if dropped:
            print(f"Dropped {len(dropped)} tensors (shape changed -> will be randomly re-initialized):")
            for key, old_shape, new_shape in dropped:
                if new_shape is None:
                    print(f"  {key}: was {old_shape}, no longer exists in stage-2 model")
                else:
                    print(f"  {key}: {old_shape} -> {new_shape}")

        if new_layers:
            print(f"\n{len(new_layers)} tensors exist only in the stage-2 model (new layers/branches, randomly initialized):")
            for key, shape in new_layers:
                print(f"  {key}: {shape}")

        torch.save({"state_dict": kept}, args.output)
        print(f"\nSaved filtered checkpoint to {args.output}")
        print("Use it with: python -m spanet.train ... --state_dict " + args.output)
    
    else:
        print("")
        print(" Running split of classification head for stage 2 spanet models...")
        print("")
        class_mapping = parse_class_mapping(args.class_mapping)

        weight_key = f"classification_decoder.networks.{args.classification_key}.output_layer.weight"
        bias_key = f"classification_decoder.networks.{args.classification_key}.output_layer.bias"
        if weight_key not in old_state or weight_key not in reference_state:
            raise KeyError(f"{weight_key} not found in checkpoint or reference model - check --classification-key")

        print(f"Splitting classification head '{args.classification_key}' "
              f"({old_state[weight_key].shape[0]} -> {reference_state[weight_key].shape[0]} classes)")
        new_weight, new_bias = split_classification_head(
            old_state, reference_state, weight_key, bias_key, class_mapping, args.noise_std
        )

        # Plain shape-matching transfer for everything else (the shared backbone), same approach
        # as prepare_finetune_checkpoint.py -- skip the head keys, already handled above.
        skip_keys = {weight_key, bias_key}
        kept, dropped, new_layers = filter_checkpoint(reference_state, old_state, skip_keys)

        kept[weight_key] = new_weight
        kept[bias_key] = new_bias

        print(f"\nTransferred {len(kept)}/{len(reference_state)} tensors "
              f"(including the split+copied classification head).\n")

        if dropped:
            print(f"Dropped {len(dropped)} tensors (shape changed, not the classification head -- random re-init):")
            for key, old_shape, new_shape in dropped:
                if new_shape is None:
                    print(f"  {key}: was {old_shape}, no longer exists in the new model")
                else:
                    print(f"  {key}: {old_shape} -> {new_shape}")

        if new_layers:
            print(f"\n{len(new_layers)} tensors exist only in the new model (new layers/branches, random init):")
            for key, shape in new_layers:
                print(f"  {key}: {shape}")

        torch.save({"state_dict": kept}, args.output)
        print(f"\nSaved split checkpoint to {args.output}")
        print("Use it with: python -m spanet.train ... --state_dict " + args.output)


if __name__ == "__main__":
    main()