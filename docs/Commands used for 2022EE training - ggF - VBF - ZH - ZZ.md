
# SPANET inputs

## PocketCoffea
```bash
run_pocket_coffea spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap_spanet_2022EE VBF_HH4b_ZZ_ZH_config.py ../VBF_HH4b/params/t3_run_options_spanet_predict_20Gb_short.yaml /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE

```
## merging
```bash
sbatch merge_output_slurm.sh /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE
```
## conversion to .h5
```bash
# real training file
 python /work/bevila_t/PostDoc/HH4b/HH4b_SPANet/utils/dataset/coffea_to_h5_direct.py  --input /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE/output_total.coffea  --output /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE/AllKlambda_HiggsPairing_ggF_VBF_HZ_ZZ_filter_abs_neg_weights_norm_by_sample_  --regions 4b_region 4b_region 4b_region 4b_region --class-labels GluGlu VBF ZH ZZ --max-jets 5 5 5 5 -j JetTotalSPANetPtFlattenPadded --resonance-list h1 h2 --remove-high-weights --all-cat-weight-filter -bw sample --balance-sample-scope custom -nwt abs
 
 # test file without padding
 python /work/bevila_t/PostDoc/HH4b/HH4b_SPANet/utils/dataset/coffea_to_h5_direct.py  --input /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE/output_total.coffea  --output /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_vbf_spanet_2022EE/AllKlambda_HiggsPairing_ggF_VBF_HZ_ZZ_filter_abs_neg_weights_norm_by_sample_  --regions 4b_region 4b_region 4b_region 4b_region --class-labels GluGlu VBF ZH ZZ --max-jets 5 5 5 5 -j JetTotalSPANetPadded --resonance-list h1 h2 --remove-high-weights --all-cat-weight-filter -bw sample --balance-sample-scope custom -nwt abs
```
## training

```bash
 python3 jobs/submit_jobs_seed.py -o options_files/HH4b/hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE.json -c jobs/config/training_1gpu_1d.yaml -s 100:100  -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/
 
 # to continue
 python3 jobs/submit_jobs_seed.py -o options_files/HH4b/hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE.json -c jobs/config/training_1gpu_1d.yaml -s 100:100 -out /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/ --suffix _continue_400ep -cf /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE/out_seed_trainings_100/version_0/checkpoints/last.ckpt -a "--epochs 400"
```

## prediction

```bash
python -m spanet.predict -g -tf /eos/user/t/tbevilac/PostDoc/spanet_input/260918/input_ggF_vbf_ZZ_ZH_vbf_spanet_2022EE/AllKlambda_HiggsPairing_ggF_VBF_HZ_ZZ_filter_abs_neg_weights_norm_by_sample_JetTotalSPANetPtFlattenPadded_test.h5 -ckpt /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE_continue_400ep/out_seed_trainings_100/version_0/checkpoints/374-0.949.ckpt /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE_continue_400ep/out_seed_trainings_100/version_0/ /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE_continue_400ep/out_seed_trainings_100/version_0/AllKlambda_HiggsPairing_ggF_VBF_HZ_ZZ_filter_abs_neg_weights_norm_by_sample_JetTotalSPANetPtFlattenPadded_testoutput.h5
```

## efficiency plots
```bash

```

## onnx conversion

```bash
python -m spanet.export /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE_continue_400ep/out_seed_trainings_100/version_0 /eos/user/t/tbevilac/PostDoc/spanet_outputs_260919/out_spanet_outputs/out_hh4b_pairing_higgs_ptFlatten_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE_continue_400ep/out_seed_trainings_100/version_0/spanet_hh4b_5jets_ptvary_btag_5wp_4b_region_all_Klambda_normalised_filtered_ggF_VBF_ZH_ZZ_2022EE.onnx --gpu
```

## pocket coffea run for bkg reweighting
```bash
 run_pocket_coffea spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap_spanet_bkg_2022EE VBF_HH4b_ZZ_ZH_config.py params/t3_run_options_spanet_predict_20Gb_short.yaml /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_data_JetMetspanet_bkg_rwg_2022EE
```
## training of bkg model
```bash
salloc --account gpu_gres --job-name "InteractiveJob" --cpus-per-task 4 --mem-per-cpu 3000 --time 01:00:00  -p gpu --gres=gpu:1

micromamba activate ML_pytorch

# train

sbatch run_20_trainings_in_4_parallel.sh DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_btag5WP_SpanetJets_2022EE /work/bevila_t/PostDoc/HH4b/Output/ML_trainings/SPANetJets_5WP_2022EE_ZZ_ZH_HH_260922 --overwrite

# merge output models
ml_onnx -i best_models -o best_models -ar -v bkg_morphing_spanet_dnn_input_variables
```
## pocket coffea with bkg reweighting
```bash
run_pocket_coffea spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap_spanet_bkg_2022EE VBF_HH4b_ZZ_ZH_config.py params/t3_run_options_spanet_predict_20Gb_short.yaml /work/bevila_t/PostDoc/HH4b/Output/PocketCoffea/ggF_vbf_ZZ_ZH_data_JetMetspanet_bkg_rwg_SvB_2022EE

# plots

```

## output merging
```bash

```
