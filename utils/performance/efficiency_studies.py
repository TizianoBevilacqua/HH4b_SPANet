import argparse
import logging
import os
import awkward as ak
import h5py
import numpy as np
import vector
import importlib.util
import sys
from pathlib import Path

from efficiency_functions import (
    best_reco_higgs,
    calculate_diff_efficiencies,
    calculate_efficiencies,
    run2_algorithm,
    load_jets_and_pairing,
    plot_diff_eff,
    plot_diff_eff_klambda,
    plot_histos_1d,
    plot_mhh,
    separate_klambda,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import helpers

vector.register_numba()
vector.register_awkward()

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
parser.add_argument(
    "-pd",
    "--plot-dir",
    default="plots",
    help="Directory to save the plots",
)
parser.add_argument(
    "-n",
    "--num-events",
    default=None,
    type=int,
    help="Consider only a limited number of events",
)
parser.add_argument(
    "-k",
    "--klambda",
    default=False,
    action="store_true",
    help="evaluate on different klambda values",
)
parser.add_argument(
    "-d",
    "--data",
    default=False,
    action="store_true",
    help="evaluate on data",
)
parser.add_argument(
    "-r",
    "--region",
    default="inclusive",
    help="define evaluation region. If 'inclusive' no cuts are applied",
)
parser.add_argument(
    "-v",
    "--vbf",
    default=False,
    action="store_true",
    help="Compute efficiency also for vbf jets",
)
parser.add_argument(
    "-vl",
    "--vbf-labels",
    default=False,
    action="store_true",
    help="Use vbf labels for k-values",
)
parser.add_argument(
    "-ih",
    "--ignore-higgs",
    default=False,
    action="store_true",
    help="Compute efficiency excluding the jets from Higgs",
)
parser.add_argument(
    "-hm",
    "--histo-mass",
    default=False,
    action="store_true",
    help="Plot the Higgs mass histograms",
)
parser.add_argument(
    "-c", "--class-label", default=None, help="Consider only the class specified"
)
parser.add_argument(
    "-conf",
    "--configuration",
    default=None,
    help="Configuration with the models to consider",
)
parser.add_argument(
    "-cms",
    "--cmstext",
    type=str,
    help="Text shown on CMS plot",
    default="Private",
)
args = parser.parse_args()

if not args.vbf and args.ignore_higgs:
    raise ValueError("Efficiency must be computed at least for one resonance!")

# the key is the value saved in the h5 file under `kl` but it is
# actually C2V (not k_lambda) because it is unique to each dataset
cv_c2v_kl_values_dict = {
    "1.37": "$\\kappa_{V}$=1.74 \n$\\kappa_{2V}$=1.37 \n$\\kappa_{\\lambda}$=14.4",
    "0.03": "$\\kappa_{V}$=-0.01 \n$\\kappa_{2V}$=0.03 \n$\\kappa_{\\lambda}$=10.2",
    "1.44": "$\\kappa_{V}$=-0.76 \n$\\kappa_{2V}$=1.44 \n$\\kappa_{\\lambda}$=-19.3",
    "0.96": "$\\kappa_{V}$=-0.96 \n$\\kappa_{2V}$=0.96 \n$\\kappa_{\\lambda}$=-1.43",
    "1.94": "$\\kappa_{V}$=-1.21 \n$\\kappa_{2V}$=1.94 \n$\\kappa_{\\lambda}$=-0.94",
    "2.72": "$\\kappa_{V}$=-1.6 \n$\\kappa_{2V}$=2.72 \n$\\kappa_{\\lambda}$=-1.36",
    "3.57": "$\\kappa_{V}$=-1.83 \n$\\kappa_{2V}$=3.57 \n$\\kappa_{\\lambda}$=-3.39",
    "3.87": "$\\kappa_{V}$=-2.12 \n$\\kappa_{2V}$=3.87 \n$\\kappa_{\\lambda}$=-5.96",
    "0.0": "$\\kappa_{V}$=1\n$\\kappa_{2V}$=0\n$\\kappa_{\\lambda}$=1",
    "1.0": "$\\kappa_{V}$=1\n$\\kappa_{2V}$=1 \n$\\kappa_{\\lambda}$=1",
}


os.makedirs(args.plot_dir, exist_ok=True)
helpers.setup_logging(args.plot_dir)
logger = logging.getLogger(__name__)

if args.configuration:
    config = helpers.import_module_from_path(args.configuration)
    logger.info(f"Imported configuration: {config}")
    run2_dataset_DATA = config.run2_dataset_DATA
    run2_dataset_MC = config.run2_dataset_MC
    spanet_dict = config.spanet_dict
    true_dict = config.true_dict
else:
    # from efficiency_configuration_cutscomparison import run2_dataset_DATA, run2_dataset_MC, spanet_dict, true_dict
    from efficiency_configuration import (
        run2_dataset_DATA,
        run2_dataset_MC,
        spanet_dict,
        true_dict,
    )

if args.data:
    # remove non data samples
    run2_dataset = run2_dataset_DATA
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" in k}
else:
    run2_dataset = run2_dataset_MC
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" not in k}

# mh_bins = np.linspace(0, 300, 150)
mh_bins = np.linspace(0, 300, 61)
mh_bins_peak = np.linspace(100, 140, 20)
mh_bins_2d = (
    [np.linspace(50, 200, 80) for _ in range(3)]
    + [np.linspace(50, 200, 40) for _ in range(6)]
    + [np.linspace(0, 500, 50) for _ in range(2)]
    + [np.linspace(0, 500, 100) for _ in range(5)]
)

# For differential pairing efficiency by hh mass. (where the pairing efficiency is somehow added cumulative).
mhh_bins = np.linspace(250, 700, 10)

# Create plotting directory
plot_dir = args.plot_dir
os.makedirs(plot_dir, exist_ok=True)


def main():
    # We are now filling a dictionary with one entry for each file.
    # Then we extract all the information from the files, that we did before, but now targeted at the specific entries.
    df_collection = {}
    for model_name, file_dict in spanet_dict.items():
        spanetfile = h5py.File(file_dict["file"], "r")
        truefile = h5py.File(true_dict[file_dict["true"]]["name"], "r")
        truefile_klambda = true_dict[file_dict["true"]]["klambda"]

        logger.info(
            f"Executing model {model_name} with spanetfile {file_dict['file']} and truefile {true_dict[file_dict['true']]['name']}"
        )

        do_vbf_pairing = (
            True
            if (args.vbf and "vbf" in file_dict.keys() and file_dict["vbf"])
            else False
        )

        true_entry = true_dict[file_dict["true"]]
        n_higgs_jets = true_entry.get("n_higgs_jets", 4)
        n_higgs_jets_pred = file_dict.get("n_higgs_jets", n_higgs_jets)
        jet_coll_higgs = true_entry.get("jet_coll_higgs", "Jet")
        jet_coll_vbf = true_entry.get("jet_coll_vbf", None)
        offset_jet_idx_higgs = file_dict.get(
            "offset_jet_idx_higgs", file_dict.get("offset_jet_idx", 0)
        )
        offset_jet_idx_vbf = file_dict.get("offset_jet_idx_vbf", 0)

        # define region mask
        mask_region_spanet = helpers.get_region_mask(
            args.region,
            spanetfile,
            do_vbf_pairing,
            jet_coll_higgs=jet_coll_higgs,
            jet_coll_vbf=jet_coll_vbf,
            n_higgs_jets=n_higgs_jets_pred,
        )
        mask_region_true = helpers.get_region_mask(
            args.region,
            truefile,
            do_vbf_pairing,
            jet_coll_higgs=jet_coll_higgs,
            jet_coll_vbf=jet_coll_vbf,
            n_higgs_jets=n_higgs_jets,
        )
        assert all(mask_region_spanet == mask_region_true)

        # define the class mask
        # take the one for the true file because
        # the prediction file might have the predicted class
        # and not the original one
        mask_class_true = helpers.get_class_mask(
            args.class_label, truefile, jet_coll=jet_coll_higgs
        )

        if args.num_events:
            mask_num_events = helpers.get_region_mask(
                f"test_{args.num_events}",
                truefile,
                False,
                jet_coll_higgs=jet_coll_higgs,
                jet_coll_vbf=jet_coll_vbf,
                n_higgs_jets=n_higgs_jets,
            )
            logger.info(f"max num events {ak.sum(mask_num_events)}")
        else:
            mask_num_events = ak.ones_like(truefile["INPUTS"][jet_coll_higgs]["MASK"][:, 0])

        mask_spanet = mask_region_spanet & mask_class_true & mask_num_events
        mask_true = mask_region_true & mask_class_true & mask_num_events

        logger.info(f"Number of events after the masks : {ak.sum(mask_true)}")

        if jet_coll_vbf is not None:
            jet_higgs = helpers.get_jet_4vec(truefile, mask_true, jet_coll=jet_coll_higgs)
            jet_vbf = helpers.get_jet_4vec(truefile, mask_true, jet_coll=jet_coll_vbf)
            n_higgs_slots = truefile["INPUTS"][jet_coll_higgs]["MASK"].shape[1]
            jet_vbf = ak.with_field(jet_vbf, jet_vbf.index + n_higgs_slots, "index")
            jet = ak.concatenate([jet_higgs, jet_vbf], axis=1)
        else:
            jet = helpers.get_jet_4vec(truefile, mask_true, jet_coll=jet_coll_higgs)

        resonances = file_dict.get("resonances")
        idx_true = load_jets_and_pairing(
            truefile,
            "true",
            higgs=not args.ignore_higgs,
            vbf=do_vbf_pairing,
            resonances=resonances,
        )[mask_true]
        idx_spanet_pred = load_jets_and_pairing(
            spanetfile,
            "spanet",
            higgs=not args.ignore_higgs,
            vbf=do_vbf_pairing,
            resonances=resonances,
        )[mask_spanet]

        # These lists are to be expanded. Didn't think of a better way than to copy them here already
        # if not klambda, the two lists stay equal.
        alltrue_idx = [idx_true]
        allspanet_idx = [idx_spanet_pred]
        alljet = [jet]
        all_name_list = [model_name]

        if args.klambda and truefile_klambda != "none":
            logger.info("Separating the klambdas")
            # separate the different klambdas
            (
                true_kl_idx_list,
                spanet_kl_idx_list,
                kl_values,
                jet_separate_klambda,
            ) = separate_klambda(
                jet,
                truefile,
                spanetfile,
                idx_true,
                idx_spanet_pred,
                mask_region=mask_true,
            )

            # I got now indexes for true and spanet and different kl.
            # For this, I will add them to a list. If no kl is there, I will just make it a list of one element
            alltrue_idx.extend(true_kl_idx_list)
            allspanet_idx.extend(spanet_kl_idx_list)
            alljet.extend(jet_separate_klambda)
            all_name_list.extend(kl_values)
        else:
            kl_values = np.array([])

        # Fully matched events
        mask_fully_matched = [
            ak.all(ak.all(idx >= 0, axis=-1), axis=-1) for idx in alltrue_idx
        ]
        alltrue_idx_fully_matched = [
            idx[mask] for idx, mask in zip(alltrue_idx, mask_fully_matched)
        ]
        allspanet_idx_fully_matched = [
            idx[mask] for idx, mask in zip(allspanet_idx, mask_fully_matched)
        ]

        logger.info(f"Model name: {model_name}")
        logger.info(file_dict["label"])
        logger.info(
            f"alltrue_idx_fully_matched {[len(idx) for idx in alltrue_idx_fully_matched]}"
        )
        logger.info(
            f"allspanet_idx_fully_matched {[len(idx) for idx in allspanet_idx_fully_matched]}"
        )
        logger.info(f"allspanet_idx {[len(pred) for pred in allspanet_idx]}")
        logger.info(f"mask_fully_matched {[len(mask) for mask in mask_fully_matched]}")

        if not args.data:
            # Performing the matching
            (
                frac_fully_matched,
                efficiencies_fully_matched,
                total_efficiencies_fully_matched,
                unc_eff_fully_matched,
                unc_total_eff_fully_matched,
                matching_eval_spanet,
            ) = calculate_efficiencies(
                alltrue_idx_fully_matched,
                allspanet_idx_fully_matched,
                mask_fully_matched,
                file_dict["true"],
                kl_values,
                all_name_list,
                "fully matched",
                higgs=not args.ignore_higgs,
                vbf=do_vbf_pairing,
                offset_jet_idx_higgs=offset_jet_idx_higgs,
                offset_jet_idx_vbf=offset_jet_idx_vbf,
            )

            # Omitting calculation of partially matched for now
            # # do the same for partially matched events (only one higgs is matched)
            # mask_1h = [
            #     ak.sum(ak.any(idx == -1, axis=-1) == 1, axis=-1) == 1 for idx in alltrue_idx
            # ]
            # idx_true_partially_matched_1h = [idx[mask] for idx, mask in zip(alltrue_idx, mask_1h)]
            # idx_spanet_partially_matched_1h = [idx[mask] for idx, mask in zip(allspanet_idx, mask_1h)]

            # frac_partially_matched_1h, efficiencies_partially_matched_spanet, total_efficiencies_partially_matched_spanet, unc_eff_partially_matched, unc_total_eff_partially_matched, matching_eval_spanet_partial = calculate_efficiencies(idx_true_partially_matched_1h, idx_spanet_partially_matched_1h, mask_1h, file_dict["true"], kl_values, all_name_list, "partially matched", higgs=not args.ignore_higgs, vbf=do_vbf_pairing)

            # # compute number of events with 0 higgs matched
            # mask_0h = [ak.sum(ak.any(idx == -1, axis=-1), axis=-1) == 2 for idx in alltrue_idx]

            # idx_true_unmatched = [idx[mask] for idx, mask in zip(alltrue_idx, mask_0h)]
            # frac_unmatched = [ak.sum(mask) / len(mask) for mask in mask_0h]
            # logger.info("\n")
            # for label, frac in zip([file_dict["true"]] + kl_values.tolist(), frac_unmatched):
            #     logger.info(f"Fraction of unmatched events for {label}: {frac:.3f}")

        # Reconstruct the Higgs boson candidates
        # of the jets considering the true pairings, the spanet pairings
        jet_fully_matched = [j[m] for j, m in zip(alljet, mask_fully_matched)]
        # Reconstruction of the Higgs boson candidates with the predicted/true pairings
        spanet_higgs_fully_matched = [
            best_reco_higgs(j, spanet_idx, higgs=not args.ignore_higgs)
            for j, spanet_idx in zip(jet_fully_matched, allspanet_idx_fully_matched)
        ]
        if not args.data:
            true_higgs_fully_matched = [
                best_reco_higgs(j, idx, higgs=not args.ignore_higgs)
                for j, idx in zip(jet_fully_matched, alltrue_idx_fully_matched)
            ]
            true_hh_fully_matched = [
                (
                    true_h_matched[:, 0] + true_h_matched[:, 1]
                    if true_h_matched is not None
                    else None
                )
                for true_h_matched in true_higgs_fully_matched
            ]

        if not args.data:
            # Differential efficiency
            eff_dict = {
                "diff_eff_spanet": [],
                "unc_diff_eff_spanet": [],
                "total_diff_eff_spanet": [],
                "total_unc_diff_eff_spanet": [],
            }
            if not args.ignore_higgs:
                for true_hh, matched_spanet, mask_matched in zip(
                    true_hh_fully_matched, matching_eval_spanet, mask_fully_matched
                ):
                    temp = {k: [] for k in eff_dict.keys()}
                    for i in range(1, len(mhh_bins)):
                        if true_hh is not None:
                            mask = (true_hh.mass > mhh_bins[i - 1]) & (
                                true_hh.mass < mhh_bins[i]
                            )

                            (
                                eff_spanet,
                                unc_eff_spanet,
                                total_eff_spanet,
                                unc_total_eff_spanet,
                            ) = calculate_diff_efficiencies(
                                matched_spanet, mask, mask_matched
                            )
                        else:
                            (
                                eff_spanet,
                                unc_eff_spanet,
                                total_eff_spanet,
                                unc_total_eff_spanet,
                            ) = (
                                None,
                                None,
                                None,
                                None,
                            )

                        temp["diff_eff_spanet"].append(eff_spanet)
                        temp["unc_diff_eff_spanet"].append(unc_eff_spanet)
                        temp["total_diff_eff_spanet"].append(total_eff_spanet)
                        temp["total_unc_diff_eff_spanet"].append(unc_total_eff_spanet)
                        # for key, val in zip(temp.keys(), [eff_spanet, unc_eff_spanet, total_eff_spanet, unc_total_eff_spanet]):
                        #     temp[key] = val
                    # Remember, we iterate here through: [inclusive, *[single kls])
                    for key in eff_dict.keys():
                        eff_dict[key].append(temp[key])

        # Iteration over the different datasets over ####
        df_collection[model_name] = {
            "file_dict": file_dict,
            "spanet_higgs_fully_matched": spanet_higgs_fully_matched,
        }
        df_collection[model_name] = df_collection[model_name] | {
            "kl_values": kl_values,
        }
        if not args.data:
            df_collection[model_name] = df_collection[model_name] | {
                "efficiencies_fully_matched": efficiencies_fully_matched,
                "unc_efficiencies_fully_matched": unc_eff_fully_matched,
                "total_efficiencies_fully_matched": total_efficiencies_fully_matched,
                "unc_total_efficiencies_fully_matched": unc_total_eff_fully_matched,
                "diff_eff_spanet": eff_dict["diff_eff_spanet"],
                "unc_diff_eff_spanet": eff_dict["unc_diff_eff_spanet"],
                "total_diff_eff_spanet": eff_dict["total_diff_eff_spanet"],
                "total_unc_diff_eff_spanet": eff_dict["total_unc_diff_eff_spanet"],
                "true_hh_fully_matched": true_hh_fully_matched,
                "true_higgs_fully_matched": true_higgs_fully_matched,
            }

    # -- Loading Run2 model
    truefile = h5py.File(true_dict[run2_dataset]["name"], "r")
    truefile_klambda = true_dict[run2_dataset]["klambda"]

    do_vbf_pairing = (
        True
        if (
            args.vbf
            and "vbf" in true_dict[run2_dataset].keys()
            and true_dict[run2_dataset]["vbf"]
        )
        else False
    )

    run2_true_entry = true_dict[run2_dataset]
    n_higgs_jets = run2_true_entry.get("n_higgs_jets", 4)
    jet_coll_higgs = run2_true_entry.get("jet_coll_higgs", "Jet")
    jet_coll_vbf = run2_true_entry.get("jet_coll_vbf", None)

    # define region mask
    mask_region_true = helpers.get_region_mask(
        args.region,
        truefile,
        do_vbf_pairing,
        jet_coll_higgs=jet_coll_higgs,
        jet_coll_vbf=jet_coll_vbf,
        n_higgs_jets=n_higgs_jets,
    )
    # assert all(mask_region_spanet == mask_region_true)

    # define the class mask
    # take the one for the true file because
    # the prediction file might have the predicted class
    # and not the original one
    mask_class_true = helpers.get_class_mask(
        args.class_label, truefile, jet_coll=jet_coll_higgs
    )
    mask_true = mask_region_true & mask_class_true

    if jet_coll_vbf is not None:
        jet_higgs_for_idx = helpers.get_jet_4vec(truefile, ak.ones_like(mask_true), jet_coll=jet_coll_higgs)
        jet_vbf_only_for_idx = helpers.get_jet_4vec(truefile, ak.ones_like(mask_true), jet_coll=jet_coll_vbf)
        n_higgs_slots = truefile["INPUTS"][jet_coll_higgs]["MASK"].shape[1]
        jet_vbf_global = ak.with_field(jet_vbf_only_for_idx, jet_vbf_only_for_idx.index + n_higgs_slots, "index")
        jet_for_idx = [ak.concatenate([jet_higgs_for_idx, jet_vbf_global], axis=1)]
        jet_vbf_for_idx = [jet_vbf_only_for_idx]
    else:
        jet_for_idx = [
            helpers.get_jet_4vec(truefile, ak.ones_like(mask_true), jet_coll=jet_coll_higgs)
        ]
        jet_vbf_for_idx = [j[:, n_higgs_jets:] for j in jet_for_idx]

    if do_vbf_pairing:
        # get the idx of the 2 leading mjj jets (excluding the 4 jets leading in btag from higgs)
        allowed_idx_vbf_run2 = helpers.get_lead_mjj_jet_idx(jet_vbf_for_idx)[0]
    else:
        allowed_idx_vbf_run2 = []

    # Run 2 method cannot use the 5th jet, so we have to compare to the 4jet model
    # Also the VBF jets are only the 2 leading in mjj
    idx_true = load_jets_and_pairing(
        truefile,
        "true_run2",
        allowed_idx_higgs=[0, 1, 2, 3],
        allowed_idx_vbf=allowed_idx_vbf_run2,
        higgs=not args.ignore_higgs,
        vbf=do_vbf_pairing,
        resonances=run2_true_entry.get("resonances"),
    )[mask_true]

    # keep only the correct jets
    jet = jet_for_idx[0][mask_true]

    # These lists are to be expanded. Didn't think of a better way than to copy them here already
    # if not klambda, the two lists stay equal.
    alltrue_idx = [idx_true]
    alljet = [jet]
    all_name_list = [run2_dataset]

    if args.klambda and truefile_klambda != "none":
        logger.info("Separating the klambdas")
        # separate the different klambdas
        (
            true_kl_idx_list,
            _,
            kl_values,
            jet_separate_klambda,
        ) = separate_klambda(
            jet, truefile, None, idx_true, None, mask_region=mask_true
        )  # Needs option for None in Spanet
        # I got now indexes for true and spanet and different kl.
        # For this, I will add them to a list. If no kl is there, I will just make it a list of one element
        alltrue_idx.extend(true_kl_idx_list)
        alljet.extend(jet_separate_klambda)
        all_name_list.extend(kl_values)

    # Fully matched events
    mask_fully_matched = [
        ak.all(ak.all(idx >= 0, axis=-1), axis=-1) for idx in alltrue_idx
    ]
    alltrue_idx_fully_matched = [
        idx[mask] for idx, mask in zip(alltrue_idx, mask_fully_matched)
    ]

    logger.info(f"Run2 dataset name: {run2_dataset}")
    logger.info(
        f"alltrue_idx_fully_matched {[len(idx) for idx in alltrue_idx_fully_matched]}"
    )
    logger.info(f"len mask_fully_matched {[len(mask) for mask in mask_fully_matched]}")
    logger.info(
        f"sum mask_fully_matched {[ak.sum(mask) for mask in mask_fully_matched]}"
    )

    allrun2_idx_fully_matched = run2_algorithm(
        alljet,
        mask_fully_matched,
        higgs=not args.ignore_higgs,
        vbf=do_vbf_pairing,
        n_higgs_jets=n_higgs_jets,
    )

    if not args.data:
        # compute efficiencies for fully matched events for Run 2 pairing
        (
            frac_fully_matched,
            efficiencies_run2,
            total_efficiencies_run2,
            unc_efficiencies_run2,
            unc_total_efficiencies_run2,
            matching_eval_run2,
        ) = calculate_efficiencies(
            alltrue_idx_fully_matched,
            allrun2_idx_fully_matched,
            mask_fully_matched,
            run2_dataset,
            kl_values,
            all_name_list,
            "run2",
            higgs=not args.ignore_higgs,
            vbf=do_vbf_pairing,
        )

    # Reconstruct the Higgs boson candidates with the efficiency_fully_matched_run2 = (
    # of the jets considering the true pairings, the spanet pairings
    # and the run2 pairings
    jet_fully_matched = [j[m] for j, m in zip(alljet, mask_fully_matched)]
    run2_higgs_fully_matched = [
        best_reco_higgs(j, idx, higgs=not args.ignore_higgs)
        for j, idx in zip(jet_fully_matched, allrun2_idx_fully_matched)
    ]
    if not args.data:
        true_higgs_fully_matched = [
            best_reco_higgs(j, idx, higgs=not args.ignore_higgs)
            for j, idx in zip(jet_fully_matched, alltrue_idx_fully_matched)
        ]
        true_hh_fully_matched = [
            (
                true_h_matched[:, 0] + true_h_matched[:, 1]
                if true_h_matched is not None
                else None
            )
            for true_h_matched in true_higgs_fully_matched
        ]

    if not args.data:
        # Differential efficiency
        eff_dict = {
            "diff_eff_run2": [],
            "unc_diff_eff_run2": [],
            "total_diff_eff_run2": [],
            "total_unc_diff_eff_run2": [],
        }
        if not args.ignore_higgs:
            for true_hh, matched_run2, mask_matched in zip(
                true_hh_fully_matched, matching_eval_run2, mask_fully_matched
            ):
                temp = {k: [] for k in eff_dict.keys()}
                for i in range(1, len(mhh_bins)):
                    if true_hh is not None:
                        mask = (true_hh.mass > mhh_bins[i - 1]) & (
                            true_hh.mass < mhh_bins[i]
                        )

                        eff_run2, unc_eff_run2, total_eff_run2, unc_total_eff_run2 = (
                            calculate_diff_efficiencies(
                                matched_run2, mask, mask_matched
                            )
                        )
                    else:
                        (
                            eff_spanet,
                            unc_eff_spanet,
                            total_eff_spanet,
                            unc_total_eff_spanet,
                        ) = (
                            None,
                            None,
                            None,
                            None,
                        )
                    temp["diff_eff_run2"].append(eff_run2)
                    temp["unc_diff_eff_run2"].append(unc_eff_run2)
                    temp["total_diff_eff_run2"].append(total_eff_run2)
                    temp["total_unc_diff_eff_run2"].append(unc_total_eff_run2)
                for key in eff_dict.keys():
                    eff_dict[key].append(temp[key])

    r2_model = {
        "file_dict": file_dict,
        "run2_higgs_fully_matched": run2_higgs_fully_matched,
    }
    r2_model = r2_model | {
        "kl_values": kl_values,
    }
    if not args.data:
        r2_model = r2_model | {
            "efficiencies_fully_matched_run2": efficiencies_run2,
            "unc_efficiencies_fully_matched_run2": unc_efficiencies_run2,
            "total_efficiencies_fully_matched_run2": total_efficiencies_run2,
            "unc_total_efficiencies_fully_matched_run2": unc_total_efficiencies_run2,
            # Parameters for the diff_eff plots
            "diff_eff_run2": eff_dict["diff_eff_run2"],
            "unc_diff_eff_run2": eff_dict["unc_diff_eff_run2"],
            "total_diff_eff_run2": eff_dict["total_diff_eff_run2"],
            "total_unc_diff_eff_run2": eff_dict["total_unc_diff_eff_run2"],
            "true_hh_fully_matched": true_hh_fully_matched,
            "true_higgs_fully_matched": true_higgs_fully_matched,
        }

    # Plotting begins here
    if not args.data:
        if args.klambda:
            logger.info("\n")
            logger.info("Plotting efficiencies fully matched for all klambda values")
            # We are adding the run2 of the chosen element (defined in efficiency_calibrations) as last element
            logger.info(f"All datasets: {df_collection.keys()}")
            logger.info(f"Run2 set: {run2_dataset}")
            plot_diff_eff_klambda(
                [
                    model["efficiencies_fully_matched"][1:]
                    for model in df_collection.values()
                ]
                + [r2_model["efficiencies_fully_matched_run2"][1:]],
                [
                    model["unc_efficiencies_fully_matched"][1:]
                    for model in df_collection.values()
                ]
                + [r2_model["unc_efficiencies_fully_matched_run2"][1:]],
                [model["kl_values"] for model in df_collection.values()]
                + [r2_model["kl_values"]],
                [model["file_dict"]["label"] for model in df_collection.values()]
                + [r"Leading $m_{jj}$" if (args.vbf and args.ignore_higgs) else r"$D_{HH}$-method + Leading $m_{jj}$" if args.vbf else r"$D_{HH}$-method"],
                [model["file_dict"]["color"] for model in df_collection.values()]
                + ["yellowgreen"],
                "eff_fully_matched_allklambda",
                plot_dir,
                xlabels=(
                    cv_c2v_kl_values_dict if (args.vbf or args.vbf_labels) else None
                ),
                cmstext=args.cmstext,
                region=args.region,
            )
            plot_diff_eff_klambda(
                [
                    model["total_efficiencies_fully_matched"][1:]
                    for model in df_collection.values()
                ]
                + [r2_model["total_efficiencies_fully_matched_run2"][1:]],
                [
                    model["unc_total_efficiencies_fully_matched"][1:]
                    for model in df_collection.values()
                ]
                + [r2_model["unc_total_efficiencies_fully_matched_run2"][1:]],
                [model["kl_values"] for model in df_collection.values()]
                + [r2_model["kl_values"]],
                [model["file_dict"]["label"] for model in df_collection.values()]
                + [r"Leading $m_{jj}$" if (args.vbf and args.ignore_higgs) else r"$D_{HH}$-method + Leading $m_{jj}$" if args.vbf else r"$D_{HH}$-method"],
                [model["file_dict"]["color"] for model in df_collection.values()]
                + ["yellowgreen"],
                "tot_eff_fully_matched_allklambda",
                plot_dir,
                xlabels=(
                    cv_c2v_kl_values_dict if (args.vbf or args.vbf_labels) else None
                ),
                cmstext=args.cmstext,
                region=args.region,
            )
        if not args.ignore_higgs:
            logger.info("Plotting differential efficiencies")
            plot_diff_eff(
                mhh_bins,
                [model["diff_eff_spanet"][0] for model in df_collection.values()]
                + [r2_model["diff_eff_run2"][0]],
                [model["unc_diff_eff_spanet"][0] for model in df_collection.values()]
                + [r2_model["unc_diff_eff_run2"][0]],
                [model["file_dict"]["label"] for model in df_collection.values()]
                + [r"Leading $m_{jj}$" if (args.vbf and args.ignore_higgs) else r"$D_{HH}$-method + Leading $m_{jj}$" if args.vbf else r"$D_{HH}$-method"],
                [model["file_dict"]["color"] for model in df_collection.values()]
                + ["yellowgreen"],
                plot_dir,
                "diff_eff_spanet",
                cmstext=args.cmstext,
                region=args.region,
            )
            plot_diff_eff(
                mhh_bins,
                [model["total_diff_eff_spanet"][0] for model in df_collection.values()]
                + [r2_model["total_diff_eff_run2"][0]],
                [
                    model["total_unc_diff_eff_spanet"][0]
                    for model in df_collection.values()
                ]
                + [r2_model["total_unc_diff_eff_run2"][0]],
                [model["file_dict"]["label"] for model in df_collection.values()]
                + [r"Leading $m_{jj}$" if (args.vbf and args.ignore_higgs) else r"$D_{HH}$-method + Leading $m_{jj}$" if args.vbf else r"$D_{HH}$-method"],
                [model["file_dict"]["color"] for model in df_collection.values()]
                + ["yellowgreen"],
                plot_dir,
                "total_diff_eff_spanet",
                cmstext=args.cmstext,
                region=args.region,
            )

    if not args.ignore_higgs and args.histo_mass:
        if not args.data:
            logger.info("Plotting mhh")
            plot_mhh(
                mhh_bins,
                list(df_collection.values())[0]["true_hh_fully_matched"][0].mass,
                plot_dir,
                "mhh_fully_matched",
                cmstext=args.cmstext,
                region=args.region,
            )

        logger.info("Plotting higgs 1d all events")
        for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
            if args.data:
                kl_values = np.array(
                    []
                )  # This is needed to make sure, that we are only producing one plot.
            for number in [1, 2]:
                for kl_variation, kl_name in zip(
                    range(len(kl_values) + 1), ["all"] + kl_values.tolist()
                ):
                    os.makedirs(f"{plot_dir}/kl_{kl_name}_massplot", exist_ok=True)
                    plot_histos_1d(
                        bins,
                        [
                            model["spanet_higgs_fully_matched"][kl_variation][
                                :, number - 1
                            ].mass
                            for model in df_collection.values()
                        ],  # spanet values
                        r2_model["run2_higgs_fully_matched"][kl_variation][
                            :, number - 1
                        ].mass,  # run2 values
                        (
                            r2_model["true_higgs_fully_matched"][kl_variation][
                                :, number - 1
                            ].mass
                            if not args.data
                            else None
                        ),  # true values
                        [
                            model["file_dict"]["label"]
                            for model in df_collection.values()
                        ],
                        [
                            model["file_dict"]["color"]
                            for model in df_collection.values()
                        ],
                        number,
                        name=f"{name}_true_run2_kl_eval_{kl_name}",
                        plot_dir=f"{plot_dir}/kl_{kl_name}_massplot",
                        cmstext=args.cmstext,
                        region=args.region,
                    )
                    # for true_model in df_collection.values():
                    #     plot_histos_1d(
                    #         bins,
                    #         [model["spanet_higgs_fully_matched"][kl_variation][:, number - 1].mass for model in df_collection.values()],  # spanet values
                    #         r2_model["run2_higgs_fully_matched"][kl_variation][:, number - 1].mass,  # run2 values
                    #         true_model["true_higgs_fully_matched"][kl_variation][:, number - 1].mass if not args.data else None,  # true values
                    #         [model["file_dict"]["label"] for model in df_collection.values()],
                    #         [model["file_dict"]["color"] for model in df_collection.values()],
                    #         number,
                    #         name=f"kl_{kl_name}_massplots/{name}_true_{true_model['file_dict']['label']}_kl_eval_{kl_name}",
                    #         plot_dir=plot_dir,
                    #     )


if __name__ == "__main__":
    main()
    logger.info(f"Plots saved in {args.plot_dir}")
