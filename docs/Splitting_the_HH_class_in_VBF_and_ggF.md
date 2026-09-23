# SPANet classifier: splitting HH into ggF/VBF (4-class -> 5-class)

Follow-up to `notes_spanet_classifier_evaluation_and_finetuning.md`. Covers the third staged fine-tune: taking the already-trained 4-class model (HH/QCD/ZH/ZZ) and splitting the merged "HH" class into separate ggF and VBF production modes (5-class: ggF/VBF/QCD/ZH/ZZ).

## 1. Why this isn't the same as the earlier "add a new class" step

Growing 2 classes -> 4 classes (the earlier stage) meant the classifier head just needed 2 more _brand-new_ output rows -- the existing HH/QCD rows didn't need to change meaning. Splitting HH -> ggF + VBF is different: there's no new information being _added_, an already-well-learned class is being _divided_. The plain shape-matching approach (`prepare_finetune_checkpoint.py`) drops the **entire** classification head as soon as its shape changes, since PyTorch's `state_dict` loading only tolerates missing keys, not same-key-different-shape ones -- so applied naively here it would throw away the whole "HH vs everything else" decision boundary, not just the missing ggF/VBF distinction, forcing the network to relearn signal-vs-background from noise at the same time as it learns a much subtler cue (ggF vs VBF differ mainly in production topology, both are genuine HH->4b signal -- a finer distinction than signal-vs-background).

## 2. The fix: row-level surgery on the classification head

Instead of discarding the head, **duplicate** the old HH row into both new ggF and VBF rows (with a small amount of noise added only to break symmetry between the two), while one-to-one carryover classes (QCD, ZH, ZZ -- unaffected, just shifted to new index positions) are copied **exactly**, no noise. Both new classes start out agreeing with the old "this is HH" boundary, and training only has to learn the actual ggF-vs-VBF split from there, not relearn signal-vs-background from scratch. The shared backbone (embeddings, attention layers) still transfers via the same plain shape-matching as before -- this only changes how the classification head specifically is handled.

## 3. Why the existing (merged) training file can't just be relabeled

Checked: the current 4-class training `.h5` retains no per-event field that could disambiguate which events were originally ggF vs VBF (only `era` and `kl`, neither of which does this). That's because label merging happens upstream, at the coffea-to-h5 conversion step (`HH4b_SPANet/utils/dataset/coffea_to_h5_direct.py`'s `dataset_to_class_index()`), which assigns each event's class by matching the **coffea sample key** (dataset name) against whatever `--class-labels` substrings were passed at conversion time -- once ggF and VBF sample keys both match the same label and get merged into one integer, the original per-event sample identity isn't carried into the `.h5` at all.

**Fix:** regenerate the `.h5` from the original per-sample coffea-level output, re-running `coffea_to_h5_direct.py` with a `--class-labels` list that keeps ggF and VBF sample keys separate this time, rather than trying to post-process the already-merged file. (This is what was actually done this session -- new 5-class file built directly, not relabeled from the old one.)

## 4. Scripts

- `prepare_finetune_checkpoint_split.py` -- standalone script implementing the row-surgery approach. Takes `--class-mapping` as `new_index=old_index` pairs (e.g. `0=0 1=0 2=1 3=2 4=3`); any old index claimed by more than one new index is auto-detected as a genuine split and gets the symmetry-breaking noise, one-to-one mappings are copied verbatim.
- `prepare_finetune_checkpoint_integrated.py` -- user-authored merge of this split-mode logic with the earlier plain "add new classes" script (`prepare_finetune_checkpoint.py`) into one file, dispatching on whether `--class-mapping` was passed. Reviewed and both code paths functionally tested (synthetic state-dict tests: plain add-classes mode, split mode, and the shared `filter_checkpoint` helper under both). Found and the user fixed three cosmetic issues:
    - `filter_checkpoint`'s first parameter (`checkpoint_path`) was dead/unused -- removed.
    - `skip_keys = {}` was an empty **dict**, not an empty set (`{}` is always a dict in Python) -- happened to work by coincidence since empty-dict and empty-set membership tests both always return `False`, but was fixed to `skip_keys = set()` for clarity.
    - Both branches printed the same log message ("Running extension of classification head...") -- now differentiated ("...extension..." vs "...split...") so job logs show which mode ran.

Usage (split mode):

```
python prepare_finetune_checkpoint_split.py \  --checkpoint <old 4-class>.ckpt \  --event_file <event.yaml, same as before -- class count isn't defined there, it's inferred                from the data, so it doesn't need to change just because the label count did> \  --training_file <new 5-class training .h5> \  --validation_file "" \  --options_file <5-class options.json> \  --classification-key EVENT/class \  --class-mapping 0=0 1=0 2=1 3=2 4=3 \  --output stage_split_for_5class.pth
```

Then launch training with `--state_dict stage_split_for_5class.pth` (optionally `--freeze_state_dict` for an initial frozen burn-in, same freeze/unfreeze pattern as the earlier 2-class -> 4-class stage).

## 5. This session's real run

- New 5-class training file: `input_spanet_training/classification/v2/ggF_VBF_HZ_ZZ_DATA/ SM_ggF_VBF_HZ_ZZ_2bDATA_..._train.h5`
- New options file: `HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/ ggF_VBF_DATA_classification_only_v2_step_3.json`
- Event file: reused `HH_ZH_ZZ_DATA_classification_only_v2.yaml` unchanged (correct -- no new input variables, only the classification label count changed)
- Old checkpoint: `out_HH_DATA_classification_only_v2_step2_phaseB_continued_2/ .../checkpoints/796-154286.812.ckpt` (a further continuation beyond what's in the earlier notes file, reaching epoch 796)

**Verified the `--class-mapping 0=0 1=0 2=1 3=2 4=3` against the real data** before running anything, rather than trusting the arithmetic on faith. New file's label counts: `{0: 204679, 1: 74999, 2: 571221, 3: 387389, 4: 160126}`. Cross-checked against the old 4-class counts:

- `204679 + 74999 = 279678` exactly matches the old merged-HH count, and the ggF:VBF ratio (~2.73:1) matches the original per-process split -> confirms **0=ggF, 1=VBF**.
- `160126` exactly matches the old ZZ count -> **4=ZZ**.
- `387389` closely matches the old ZH count (387343) -> **3=ZH**.
- `571221` (the remainder) -> **2=QCD/DATA** (doesn't need to match the old QCD count exactly, since this background sample was regenerated independently with a different downscale factor).

So the real encoding is `0=ggF, 1=VBF, 2=QCD, 3=ZH, 4=ZZ`, split from the old `0=HH, 1=QCD, 2=ZH, 3=ZZ` -- exactly what the mapping encodes. Also cross-checked that the new options file's own embedded `training_file`/`event_info_file` match what's passed on the command line (no silent-override surprise this time, unlike the `_only`/no-`_only` yaml mismatch caught in an earlier session).


# Recently used commands:

## recreate the .h5 input

```bash
micromamba activate coffea_old

python /work/bevila_t/PostDoc/HH4b/HH4b_SPANet/utils/dataset/coffea_to_h5_direct.py  --input /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_data_parkingHH_vbf_spanet_bkg_rwg_2023_postBPix/other_samples/output_total.coffea  --output /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_data_parkingHH_spanet_bkg_rwg_2023_postBPix/SM_ggF_VBF_HZ_ZZ_2bDATA_filter_abs_neg_weights_norm_by_sample_downsc_fact_0p05_  --regions 4b_region 4b_region 2b_region_postW 4b_region 4b_region --class-labels GluGlu VBF DATA ZH ZZ --max-jets 7 7 7 7 7 -j JetTotalSPANetPadded --resonance-list h1 h2 --remove-high-weights --all-cat-weight-filter -bw sample --balance-sample-scope custom -nwt abs --downscale_training DATA 0.05
```

## split the HH class (lxplus)

```bash
spanet_singularity

cd postdoc
source spanet_setup.sh; y

cd HH4b_SPANet

python scripts/prepare_finetune_checkpoint_integrated.py   \
--checkpoint /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2_step2_phaseB_continued_2/out_seed_trainings_100/version_0/checkpoints/796-154286.812.ckpt \
  --event_file /afs/cern.ch/user/t/tbevilac/postdoc/HH4b_SPANet/event_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_only_v2.yaml \
  --training_file /eos/user/t/tbevilac/PostDoc/spanet_input/classification/v2/input_ggF_vbf_ZZ_ZH_data_parkingHH_vbf_spanet_bkg_rwg_2023_postBPix/ggF_VBF_HZ_ZZ_DATA/SM_ggF_VBF_HZ_ZZ_2bDATA_filter_abs_neg_weights_norm_by_sample_downsc_fact_0p05_JetTotalSPANetPadded_train.h5 \
  --validation_file "" \
  --options_file /afs/cern.ch/user/t/tbevilac/postdoc/HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/ggF_VBF_DATA_classification_only_v2_step_3.json \
  --classification-key EVENT/class \
  --class-mapping 0=0 1=0 2=1 3=2 4=3 \
  --output /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2_step2_phaseB_continued_2/out_seed_trainings_100/version_0/checkpoints/stage_split_for_5class.pth
```

## training

`-cf` maps to `spanet.train --checkpoint`, which does a **full PyTorch Lightning resume** (`trainer.fit(model, ckpt_path=...)`) — it expects a complete Lightning checkpoint dict with `epoch`, `global_step`, `optimizer_states`, `lr_schedulers`, etc. But `stage_split_for_5class.pth` was saved by our script as just `{"state_dict": kept}` — nothing else. Lightning's resume path will try to read those missing keys and fail. That file was built specifically for `--state_dict` (`-sf` on `train.py`), which does the lenient, weights-only load (`torch.load(path)["state_dict"]` + `strict=False`) — exactly the mechanism we used for the earlier 2-class→4-class jump too.

```bash
python3 jobs/submit_jobs_seed.py   -o options_files/HH4b/classification/HH_ZH_ZZ/ggF_VBF_DATA_classification_only_v2_step_3.json   -c jobs/config/training_1gpu_3d.yaml -s 100:100  -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260914/ -a "--state_dict /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2_step2_phaseB_continued_2/out_seed_trainings_100/version_0/checkpoints/stage_split_for_5class.pth --epochs 200"
```