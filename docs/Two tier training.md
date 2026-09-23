
# SPANet multiclass classifier: evaluation + staged fine-tuning notes

Session notes for: reading SPANet `.h5` classifier outputs, plotting a confusion matrix,
and doing a staged pretrain -> fine-tune (2-class HH/QCD -> 4-class HH/QCD/ZH/ZZ).

## 1. File / environment setup

Two local venvs were created under `PostDoc/`:
- `.venv_h5plot` — lightweight: `h5py`, `numpy`, `matplotlib`, `scikit-learn`. For reading `.h5` and plotting only.
- `.venv_spanet` — real modeling stack: `torch`, `pytorch-lightning`, `spanet` (editable-installed from `SPANet/`). Needed for anything that has to build the actual model (checkpoint surgery, training).

`spanet.train` needs `torch`/`pytorch_lightning`/`spanet` importable — plotting-only tasks don't.

## 2. Reading the h5 files / confusion matrix

Truth and prediction live in **separate** `.h5` files, both under the key `CLASSIFICATIONS/EVENT/class`:
- Truth file: `CLASSIFICATIONS/EVENT/class` = 1D int array of class labels.
- Prediction file: `CLASSIFICATIONS/EVENT/class` = `(N, num_classes)` float array of softmax probabilities (rows sum to 1). Predicted class = `argmax` over axis 1.
- `WEIGHTS/weight` = per-event weight (already normalized per-sample/class — roughly equal weighted yield per class, so don't use weighted yields to infer class identity).

Original 5-class encoding (confirmed from filename `SM_ggF_VBF_ZZ_ZH...`): **0=ggF, 1=VBF, 2=ZH, 3=ZZ, 4=QCD**.

Script: `plot_confusion_matrix.py` — loads both files, argmaxes predictions, plots a row-normalized (weighted) and a raw-count confusion matrix via `sklearn.metrics.confusion_matrix`. Weighted accuracy on the original 5-class test set was ~72.3%.

## 3. Why pretrain then fine-tune

Idea: train the network first on an easier 2-class problem (HH vs QCD), then grow the classifier head to include ZH and ZZ, transferring the already-learned shared backbone instead of starting from scratch.

*Class encoding: **0=HH (ggF+VBF merged), 1=QCD, 2=ZH, 3=ZZ**.*

In this actual analysis, the labeled 2-class and 4-class `.h5` files were **already produced upstream**  found in `input_spanet_training/classification/`:
- Stage 1 (2-class): `input_ggF_data_parkingHH_spanet_bkg_rwg_2023_postBPix/.../SM_HH_DATA_..._train.h5` — labels `{0: HH, 1: QCD}`.
- Stage 2 (4-class): `input_HH_ZZ_ZH_data_parkingHH_spanet_bkg_rwg_2023_postBPix/SM_HH_DATA_..._train.h5` — labels `{0: HH, 1: QCD, 2: ZH, 3: ZZ}`.

`make_pretrain_subset.py` builds this 2-class subset generically from a
5-class file, for cases where you *do* need to build it yourself (filters to ggF/VBF/QCD events, merges ggF+VBF -> label 0, QCD -> label 1). Wasn't actually needed for this real run since the data already existed, but keep it for future similar splits.

## 4. Checkpoint-loading mechanics (why this isn't just "load the checkpoint")

A checkpoint's `state_dict` is a dict: layer name (string key) -> weight tensor. `spanet.train`'s
`--state_dict` flag loads with `strict=False`:
- Key missing from the checkpoint -> silently skipped, layer stays at random init. This is what `strict=False` actually tolerates.
- Key present in both but **shape differs** -> hard crash. `strict=False` does NOT forgive this case.

Growing 2 classes -> 4 classes changes the classifier head's output shape, so a naive load crashes.
(Adding a new *input variable* to a branch would similarly resize only that branch's first embedding layer — `EmbeddingStack`'s later layers and all untouched branches keep identical shapes and are unaffected, per `SPANet/spanet/network/layers/embedding_stack.py`. Not relevant this round since stage 1 and stage 2 use the same input variables.)

**Fix:** build the real stage-2 model (so its layer shapes are known), then filter the stage-1
checkpoint's `state_dict` to drop any key whose shape doesn't match the stage-2 model's own shape for that key, before loading. Everything with a matching shape transfers; everything else re-initializes.

Script: `prepare_finetune_checkpoint.py` — mirrors `spanet.train`'s own model-construction path
(`Options(event_file, training_file, validation_file)` -> `options.update_options(json.load(options_file))` -> `JetReconstructionModel(options)`), then does the shape-filtered dict copy and saves it in the `{"state_dict": ...}` format `--state_dict` expects.

**Real run result** (stage-1 checkpoint `46-98426.922.ckpt`, the best of 10 saved by the `mode='max'`metric callback, -> stage-2 model): **1103 of 1106 tensors transferred**; only 3 dropped, all in the
classification head:
```
classification_weights.EVENT/class                                (2,)    -> (4,)
classification_decoder.networks.EVENT/class.output_layer.weight    (2, 32) -> (4, 32)
classification_decoder.networks.EVENT/class.output_layer.bias      (2,)    -> (4,)
```
Confirms the whole shared backbone (embeddings, attention layers) transfers untouched.

Usage:
```bash
python prepare_finetune_checkpoint.py \
  --checkpoint <stage1>.ckpt \
  --event_file <stage2_event.yaml> \
  --training_file <stage2_train.h5> \
  --validation_file "" \
  --options_file <stage2_options.json> \
  --output stage1_filtered_for_stage2.pth
```
Note: `${SPANET_MAIN_DIR}`-style vars inside options JSON files get expanded via `os.path.expandvars` (`SPANet/spanet/options.py:323`) — export that env var first if running this standalone outside `training.sh`.

## 5. Two-phase fine-tuning (freeze -> unfreeze)

Rationale: the new classifier head starts as random noise; training everything at once lets its noisy early gradients degrade the already-good pretrained backbone. Freezing the backbone for a short burn-in lets the head catch up first.

- **`--freeze_state_dict`**: only applies in the `--state_dict` load path (`SPANet/spanet/train.py:153`)
  — freezes every tensor that *was* successfully transferred, leaving only the reinitialized layers
  (the classifier head) trainable.
- **`--checkpoint`**: full resume (weights + optimizer + trainer state via `ckpt_path=`). Crucially, this path does **not** re-run the freeze logic at all (that block only executes when loading via
  `--state_dict`), so resuming via `--checkpoint` naturally unfreezes everything — this is the mechanism used to move from phase A to phase B.

**Phase A** (frozen backbone, short burn-in, e.g. ~20 epochs — it's a tiny 32-wide head, converges fast):
```bash
./HH4b_SPANet/jobs/training.sh \
  -o HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json \
  -n out_HH_ZH_ZZ_step2_phaseA_frozen \
  -s 100 -g 1 -m <SPANET_MAIN_DIR> -e <SPANET_ENV_DIR> -H <HOME_DIR> \
  -- --state_dict stage1_filtered_for_stage2.pth --freeze_state_dict --epochs 20
```

**Phase B** (unfreeze, resume from phase A, continue to full epoch target):
```bash
./HH4b_SPANet/jobs/training.sh -o HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json -n out_HH_ZH_ZZ_step2_phaseB_unfrozen -s 100 -g 1 -m ${SPANET_MAIN_DIR} -e ${SPANET_ENV_DIR} -H ./ -- --checkpoint /eos/user/t/tbevilac/PostDoc/spanet_outputs_260819/out_spanet_outputs/out_HH_ZH_ZZ_DATA_classification_step_2_phaseA_frozen/out_seed_trainings_100/version_0/checkpoints/last.ckpt --epochs 200
```
`--epochs` here is the **total** target epoch count (matches the options file's own value), not an
increment — Lightning resumes from phase A's saved `current_epoch`.

## 6. Condor submission (`jobs/submit_jobs_seed.py`)

Chain: `submit_jobs_seed.py` -> `submit_to_condor.py` -> `training.sh` -> `python -m spanet.train`.

- `-cf/--checkpoint` on `submit_jobs_seed.py` is first-class and flows straight to `spanet.train --checkpoint` (phase B).
- `-a/--add_args` is a generic passthrough string, landing verbatim after `--` in the `training.sh` call — used for phase A's `--state_dict`/`--freeze_state_dict`/`--epochs` (not first-class flags on the submit scripts).
- **Watch out:** output dir name is derived from the options filename (`submit_jobs_seed.py:70`). Since both phases reuse the same `..._step_2.json`, use `--suffix` to keep phase A/B output directories from colliding.

```bash
# Phase A
python3 jobs/submit_jobs_seed.py \
  -o options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json \
  -c jobs/config/training_1gpu_1d.yaml -s 100:100 \
  -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260819/ \
  --suffix _phaseA_frozen \
  -a "--state_dict /eos/.../stage1_filtered_for_stage2.pth --freeze_state_dict --epochs 20"

# Phase B (fill in the real last.ckpt path phase A produced)
python3 jobs/submit_jobs_seed.py \
  -o options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json \
  -c jobs/config/training_1gpu_1d.yaml -s 100:100 \
  -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260819/ \
  --suffix _phaseB_unfrozen \
  -cf /eos/user/t/tbevilac/PostDoc/spanet_outputs_260819/out_spanet_outputs/out_HH_ZH_ZZ_DATA_classification_step_2_phaseA_frozen/out_seed_trainings_100/version_0/checkpoints/last.ckpt \
  -a "--epochs 200"
```

## 7. File inventory (in `PostDoc/`)

| File                                         | Purpose                                                | Status                                               |
| -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| `plot_confusion_matrix.py`                   | Reads truth+pred h5, plots confusion matrices          | Ready, run on real 5-class test data                 |
| `make_pretrain_subset.py`                    | Builds a 2-class HH/QCD subset from a 5-class h5       | Ready; not needed this round (data pre-existed)      |
| `prepare_finetune_checkpoint.py`             | Shape-filters a checkpoint for the next training stage | Ready, verified on real data (1103/1106 transferred) |
| `.venv_h5plot/`, `.venv_spanet/`             | Local venvs for plotting vs real SPANet/torch work     | —                                                    |
| `finetune_local/`, `test_pretrain_HH_QCD.h5` | Local dry-run scratch artifacts from this session      | Safe to delete once done referencing                 |

## 8. Real paths used this session

- Stage 1 checkpoint used: `spanet_output/out_HH_DATA_classification_only/out_seed_trainings_100/version_0/checkpoints/46-98426.922.ckpt` (best of 10 saved, by the `mode='max'` metric callback)
- Stage 1 options: `HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_DATA_classification_only.json`
- Stage 2 options: `HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json`
- Stage 2 event yaml: `HH_ZH_ZZ_DATA_classification.yaml` (differs from stage 1's `..._only.yaml` only in `EVENT` target-assignment syntax, not `INPUTS` — irrelevant here since `assignment_loss_scale: 0.0`)


# RECENTLY USED COMMANDS

### Fine tuning
```bash
python HH4b_SPANet/scripts/prepare_finetune_checkpoint.py --checkpoint /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only/out_seed_trainings_100/version_0/checkpoints/46-98426.922.ckpt --event_file "${SPANET_MAIN_DIR}/HH4b_SPANet/event_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification.yaml" --training_file /eos/user/t/tbevilac/PostDoc/spanet_input/classification/input_HH_ZZ_ZH_data_parkingHH_spanet_bkg_rwg_2023_postBPix/SM_HH_DATA_JetGoodFromHiggsOrdered_filter_abs_neg_weights_norm_by_sample_JetGoodFromHiggsOrdered_train.h5 --validation_file "" --options_file HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_step_2.json --output stage1_filtered_for_stage2.pth



# fine tuning v2
python HH4b_SPANet/scripts/prepare_finetune_checkpoint.py --checkpoint /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2/out_seed_trainings_100/version_0/checkpoints/126-192289.953.ckpt --event_file "${SPANET_MAIN_DIR}/HH4b_SPANet/event_files/HH4b/classification/HH_ZH_ZZ/HH_ZH_ZZ_DATA_classification_only_v2.yaml" --training_file /eos/user/t/tbevilac/PostDoc/spanet_input/classification/v2/input_ggF_vbf_ZZ_ZH_data_parkingHH_vbf_spanet_bkg_rwg_2023_postBPix/HH_HZ_ZZ_DATA/SM_HH_HZ_ZZ_DATA_filter_abs_neg_weights_norm_by_sample_JetTotalSPANetPadded_train.h5 --validation_file "" --options_file HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/HH_DATA_classification_only_v2_step2.json --output stage1_filtered_for_stage2.pth
```

### step 2 frozen

```bash

python3 jobs/submit_jobs_seed.py -o options_files/HH4b/classification/HH_ZH_ZZ/HH_DATA_classification_only_v2_step2.json -c jobs/config/training_1gpu_1d.yaml -s 100:100 -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/ --suffix _phaseA_frozen -a "--state_dict /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2/out_seed_trainings_100/version_0/checkpoints/stage1_filtered_for_stage2.pth --freeze_state_dict --epochs 40"
```

## step 2 unfrozen:

```bash
python3 jobs/submit_jobs_seed.py -o /afs/cern.ch/user/t/tbevilac/postdoc/HH4b_SPANet/options_files/HH4b/classification/HH_ZH_ZZ/ggF_VBF_DATA_classification_only_v2_step_3.json -c jobs/config/training_1gpu_3d.yaml -s 100:100 -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260914/ -cf /eos/user/t/tbevilac/PostDoc/spanet_outputs_260820/out_spanet_outputs/out_HH_DATA_classification_only_v2_step2_phaseB_continued_2/out_seed_trainings_100/version_0/checkpoints/796-154286.812.ckpt -a "--epochs 800"
```


