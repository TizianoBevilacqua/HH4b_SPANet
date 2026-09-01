import logging
from math import sqrt
import copy
import awkward as ak
import numpy as np
import vector
import sys
import os
from hist import Hist
from utils_configs.plot.HEPPlotter import HEPPlotter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import helpers

logger = logging.getLogger(__name__)

vector.register_awkward()
vector.register_numba()

RESONANCES_DICT = {
    "OLD_RESONANCES": {
        "h1": (1, ("b1", "b2")),
        "h2": (2, ("b3", "b4")),
        "vbf": (3, ("q1", "q2")),
    },
    "DEFAULT_RESONANCES": {
        "h1": (1, ("b1", "b2")),
        "h2": (2, ("b1", "b2")),
        "vbf": (3, ("q1", "q2")),
    }
}


def check_double_assignment_vectorized(full_idx_list, label):
    # Stack indices for each event: shape (N_events, 4)
    idx_all = np.stack(full_idx_list, axis=1)
    # Mask for valid indices
    valid_mask = idx_all >= 0
    # Set invalid indices to a large negative number so they don't interfere
    idx_all_valid = np.where(valid_mask, idx_all, -9999)
    # For each event, sort the indices
    idx_sorted = np.sort(idx_all_valid, axis=1)
    # Compare adjacent indices for equality (double assignment)
    double_assigned = np.any(
        (idx_sorted[:, 1:] == idx_sorted[:, :-1]) & (idx_sorted[:, 1:] >= 0), axis=1
    )
    n_double = np.sum(double_assigned)
    if n_double > 0:
        rows = np.where(double_assigned)[0]
        logger.info(f"WARNING: {n_double} events in {label} have double-assigned jets!")
        logger.info(f"Rows with double assignments: {rows}")
    else:
        logger.info(f"OK: No double-assigned jets found in {label}.")


def load_jets_and_pairing(
    samplefile,
    label,
    allowed_idx_higgs=None,
    allowed_idx_vbf=None,
    higgs=True,
    vbf=False,
    resonances=None,
):
    if resonances is None:
        resonances = RESONANCES_DICT["OLD_RESONANCES"]
    else:
        resonances = RESONANCES_DICT[resonances]
        
    sorted_resonances = sorted(resonances.items(), key=lambda x: x[1][0])
    higgs_resonances = sorted_resonances[:-1]
    vbf_res_name, (_, (vbf_q1_name, vbf_q2_name)) = sorted_resonances[-1]

    full_idx_list = []
    if higgs:
        for res_name, (_, (q1_name, q2_name)) in higgs_resonances:
            idx_q1 = samplefile["TARGETS"][res_name][q1_name][()]
            idx_q2 = samplefile["TARGETS"][res_name][q2_name][()]

            if allowed_idx_higgs is not None:
                # keep up to max_jets for the pairing
                idx_q1 = ak.where(
                    ak.any(idx_q1[..., None] == allowed_idx_higgs, axis=-1), idx_q1, -1
                )
                idx_q2 = ak.where(
                    ak.any(idx_q2[..., None] == allowed_idx_higgs, axis=-1), idx_q2, -1
                )

            idx_res = ak.concatenate(
                (
                    ak.unflatten(idx_q1, ak.ones_like(idx_q1)),
                    ak.unflatten(idx_q2, ak.ones_like(idx_q2)),
                ),
                axis=1,
            )
            full_idx_list.append(ak.unflatten(idx_res, ak.ones_like(idx_res[:, 0])))

    if vbf and vbf_res_name in samplefile["TARGETS"].keys():
        # Include the VBF matching
        idx_q1 = samplefile["TARGETS"][vbf_res_name][vbf_q1_name][()]
        idx_q2 = samplefile["TARGETS"][vbf_res_name][vbf_q2_name][()]

        if allowed_idx_vbf is not None:
            # keep up to max_jets for the pairing
            idx_q1 = ak.where(
                ak.any(idx_q1[..., None] == allowed_idx_vbf, axis=-1), idx_q1, -1
            )
            idx_q2 = ak.where(
                ak.any(idx_q2[..., None] == allowed_idx_vbf, axis=-1), idx_q2, -1
            )

        idx_vbf = ak.concatenate(
            (
                ak.unflatten(idx_q1, ak.ones_like(idx_q1)),
                ak.unflatten(idx_q2, ak.ones_like(idx_q2)),
            ),
            axis=1,
        )
        full_idx_list.append(ak.unflatten(idx_vbf, ak.ones_like(idx_vbf[:, 0])))

    check_double_assignment_vectorized(full_idx_list, label)
    idx = ak.concatenate(
        tuple(full_idx_list),
        axis=1,
    )

    return idx


def calculate_efficiencies(
    true_idx,
    prediction_idx,
    mask,
    truename,
    kl_values,
    all_names,
    label,
    higgs=True,
    vbf=False,
    offset_jet_idx_higgs=0,
    offset_jet_idx_vbf=0,
):
    matching_eval_model = [ak.ones_like(true[:, 0, 0]) for true in true_idx]

    if higgs:
        # higgs 1 and higgs 2
        matching_eval_model_higgs = [
            (
                ak.all(true[:, 0] == prediction[:, 0] + offset_jet_idx_higgs, axis=1)
                | ak.all(true[:, 0] == prediction[:, 1] + offset_jet_idx_higgs, axis=1)
            )
            & (
                ak.all(true[:, 1] == prediction[:, 0] + offset_jet_idx_higgs, axis=1)
                | ak.all(true[:, 1] == prediction[:, 1] + offset_jet_idx_higgs, axis=1)
            )
            for true, prediction in zip(true_idx, prediction_idx)
        ]
        matching_eval_model = [
            matching_eval_model[i] & matching_eval_model_higgs[i]
            for i in range(len(matching_eval_model))
        ]

    if vbf:
        # if there are also the higgs idx, then the vbf is the third idx
        # otherwise is the first idx
        if higgs:
            idx_vbf = 2
        else:
            idx_vbf = 0

        # vbf
        matching_eval_model_vbf = [
            (
                ak.all(true[:, idx_vbf] == prediction[:, idx_vbf] + offset_jet_idx_vbf, axis=1)
                # check also if the idx are swapped (altought this shouldn't happen)
                | ak.all(true[:, idx_vbf, ::-1] == prediction[:, idx_vbf] + offset_jet_idx_vbf, axis=1)
            )
            for true, prediction in zip(true_idx, prediction_idx)
        ]
        matching_eval_model = [
            matching_eval_model[i] & matching_eval_model_vbf[i]
            for i in range(len(matching_eval_model))
        ]

    # compute efficiencies for events
    model_eff = [
        ak.sum(matching_eval_ds) / len(matching_eval_ds)
        for matching_eval_ds in matching_eval_model
    ]
    logger.info(f"matching_{label}_model: {[len(c) for c in matching_eval_model]}")
    fraction = [ak.sum(m) / len(m) for m in mask]
    total_model_eff = [eff * frac for frac, eff in zip(fraction, model_eff)]
    unc_model_eff = [
        sqrt(eff * (1 - eff) / len(matching_eval_ds))
        for eff, matching_eval_ds in zip(model_eff, matching_eval_model)
    ]
    unc_total_model_eff = [
        sqrt((total_eff * (1 - total_eff)) / len(matching_eval_ds))
        for total_eff, matching_eval_ds in zip(total_model_eff, matching_eval_model)
    ]
    # Printing the values
    logger.info("\n")
    for lab, frac in zip([truename] + kl_values.tolist(), fraction):
        logger.info(f"Fraction of {label} events for {lab}: {frac:.3f}")

    logger.info("\n")
    for lab, eff in zip(all_names, model_eff):
        logger.info(f"Efficiency {label} for {lab}: {eff:.3f}")
    # logger.info("\n")
    # for name, toteff in zip(all_names, unc_model_eff):
    #     logger.info(f"Efficiency uncertainty {label} for {name}: {toteff:.3f}")

    logger.info("\n")
    for lab, eff in zip(all_names, total_model_eff):
        logger.info(f"Total efficiency {label} for {lab}: {eff:.3f}")
    # logger.info("\n")
    # for name, toteff in zip(all_names, unc_total_model_eff):
    #     logger.info(f"Total efficiency uncertainty {label} for {name}: {toteff:.3f}")

    return (
        fraction,
        model_eff,
        total_model_eff,
        unc_model_eff,
        unc_total_model_eff,
        matching_eval_model,
    )


def distance_pt_func(higgs_pair, k):
    if len(higgs_pair[0, 0]) == 0:
        return np.array([])
    higgs1 = higgs_pair[:, :, 0]
    higgs2 = higgs_pair[:, :, 1]
    dist = abs(higgs1.mass - higgs2.mass * k) / sqrt(1 + k**2)
    max_pt = np.maximum(higgs1.pt, higgs2.pt)
    return dist, max_pt


def reco_higgs(jet_collection, idx_collection):
    if len(jet_collection) == 0:
        higgs_candidates_unflatten_order = ak.Array([[[], []], [[], []], [[], []]])
        return higgs_candidates_unflatten_order

    higgs_01 = ak.unflatten(
        jet_collection[:, idx_collection[0][0][0]]
        + jet_collection[:, idx_collection[0][0][1]],
        1,
    )
    higgs_23 = ak.unflatten(
        jet_collection[:, idx_collection[0][1][0]]
        + jet_collection[:, idx_collection[0][1][1]],
        1,
    )

    higgs_02 = ak.unflatten(
        jet_collection[:, idx_collection[1][0][0]]
        + jet_collection[:, idx_collection[1][0][1]],
        1,
    )
    higgs_13 = ak.unflatten(
        jet_collection[:, idx_collection[1][1][0]]
        + jet_collection[:, idx_collection[1][1][1]],
        1,
    )

    higgs_03 = ak.unflatten(
        jet_collection[:, idx_collection[2][0][0]]
        + jet_collection[:, idx_collection[2][0][1]],
        1,
    )
    higgs_12 = ak.unflatten(
        jet_collection[:, idx_collection[2][1][0]]
        + jet_collection[:, idx_collection[2][1][1]],
        1,
    )

    higgs_pair_0 = ak.concatenate([higgs_01, higgs_23], axis=1)
    higgs_pair_1 = ak.concatenate([higgs_02, higgs_13], axis=1)
    higgs_pair_2 = ak.concatenate([higgs_03, higgs_12], axis=1)

    higgs_candidates = ak.concatenate(
        [higgs_pair_0, higgs_pair_1, higgs_pair_2], axis=1
    )
    higgs_candidates_unflatten = ak.unflatten(higgs_candidates, 2, axis=1)

    # order the higgs candidates by pt
    higgs_candidates_unflatten_order_idx = ak.argsort(
        higgs_candidates_unflatten.pt, axis=2, ascending=False
    )
    higgs_candidates_unflatten_order = higgs_candidates_unflatten[
        higgs_candidates_unflatten_order_idx
    ]
    return higgs_candidates_unflatten_order


def best_reco_higgs(jet_collection, idx_collection, higgs=True):
    idx_collection_noNone = np.asarray(
        ak.fill_none(copy.copy(idx_collection), -1), dtype=np.int64
    )
    if len(jet_collection) > 0:
        if higgs:
            higgs_1 = ak.unflatten(
                jet_collection[
                    np.arange(len(idx_collection_noNone)),
                    idx_collection_noNone[:, 0, 0],
                ]
                + jet_collection[
                    np.arange(len(idx_collection_noNone)),
                    idx_collection_noNone[:, 0, 1],
                ],
                1,
            )
            higgs_2 = ak.unflatten(
                jet_collection[
                    np.arange(len(idx_collection_noNone)),
                    idx_collection_noNone[:, 1, 0],
                ]
                + jet_collection[
                    np.arange(len(idx_collection_noNone)),
                    idx_collection_noNone[:, 1, 1],
                ],
                1,
            )

            higgs_pair = ak.concatenate([higgs_1, higgs_2], axis=1)

            # order the higgs candidates by pt
            higgs_candidates_unflatten_order_idx = ak.argsort(
                higgs_pair.pt, axis=1, ascending=False
            )
            higgs_candidates_unflatten_order = higgs_pair[
                higgs_candidates_unflatten_order_idx
            ]
        else:
            # if we don't reconstruct the higgs, put a dummy output
            higgs_candidates_unflatten_order = ak.ones_like(
                ak.concatenate(
                    [
                        ak.unflatten(idx_collection_noNone[:, 0, 0], 1),
                        ak.unflatten(idx_collection_noNone[:, 0, 0], 1),
                    ],
                    axis=1,
                )
            )
    else:
        higgs_candidates_unflatten_order = None

    return higgs_candidates_unflatten_order


def run2_algorithm(jet, mask_fully_matched, higgs=True, vbf=False, n_higgs_jets=4):
    logger.info("Running Run 2 algorithm...")
    comb_idx_min_list = []
    if higgs:
        # implement the Run 2 pairing algorithm
        comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

        higgs_candidates_unflatten_order = [reco_higgs(j, comb_idx) for j in jet]
        distance = [
            distance_pt_func(higgs, 1.04)[0]
            for higgs in higgs_candidates_unflatten_order
        ]
        max_pt = [
            distance_pt_func(higgs, 1.04)[1]
            for higgs in higgs_candidates_unflatten_order
        ]

        dist_order_idx = [ak.argsort(d, axis=1, ascending=True) for d in distance]
        dist_order = [ak.sort(d, axis=1, ascending=True) for d in distance]

        pt_order_idx = [ak.argsort(pt, axis=1, ascending=False) for pt in max_pt]
        # if the distance between the two best candidates is less than 30, we do not consider the event
        min_idx = [
            ak.where(d[:, 1] - d[:, 0] > 30, d_idx[:, 0], pt_idx[:, 0])
            for d, d_idx, pt_idx in zip(dist_order, dist_order_idx, pt_order_idx)
        ]

        comb_idx = [np.tile(comb_idx, (len(m), 1, 1, 1)) for m in min_idx]
        # given the min_idx, select the correct combination corresponding to the index
        comb_idx_min_higgs = [
            comb[np.arange(len(m)), m] for comb, m in zip(comb_idx, min_idx)
        ]
        comb_idx_min_list.append(comb_idx_min_higgs)

    if vbf:
        # choose higgs jets as the two jets with the highest mjj that are not from higgs decay
        jet_vbf = [j[:, n_higgs_jets:] for j in jet]
        comb_idx_min_vbf = helpers.get_lead_mjj_jet_idx(jet_vbf)
        comb_idx_min_vbf = [np.expand_dims(ci, 1) for ci in comb_idx_min_vbf]

        comb_idx_min_list.append(comb_idx_min_vbf)

    # combine the idx
    comb_idx_min_conc = ak.concatenate(comb_idx_min_list, axis=2)
    allrun2_idx_fully_matched = [
        ak.Array(comb)[m] for comb, m in zip(comb_idx_min_conc, mask_fully_matched)
    ]

    logger.info("Finished running Run 2 algorithm!")

    return allrun2_idx_fully_matched


def calculate_diff_efficiencies(matched, mask, mask_matched):
    """This function is designed to determine the differencial efficiencies by mhh_bins.
    Used to avoid repetition between run2 and spanet.
    :param matched: the events passing a mask for being fully matched
    :param mask: the mask determining the current bin
    :param mask_matched: masking array for an event being fully matched (not very efficient...)

    :return: efficiency, uncertainty of efficiency, total efficiency, uncertainty of total efficiency
    """
    eff = ak.sum(matched[mask]) / ak.count(matched[mask])
    unc_eff = sqrt(eff * (1 - eff) / ak.count(matched[mask]))
    frac_fully_matched = ak.sum(mask_matched[mask]) / len(mask_matched[mask])
    total_eff = eff * frac_fully_matched
    unc_total_eff = sqrt((total_eff * (1 - total_eff)) / len(mask_matched[mask]))
    return eff, unc_eff, total_eff, unc_total_eff


def separate_klambda(
    jet, df_true, df_spanet_pred, idx_true, idx_spanet_pred, mask_region
):
    logger.info(f"jet {len(jet)}, {len(jet[0])}")
    try:
        kl_array_true = df_true["INPUTS"]["Event"]["kl"][()][mask_region]
    except KeyError:
        logger.info("Did not find Event/kl in kl_array_true, will try EVENT/kl")
        kl_array_true = df_true["INPUTS"]["EVENT"]["kl"][()][mask_region]

    logger.info(f"kl_arrays {kl_array_true}")

    # for each kl_array, separate the array based on the kl value
    # and create a list of arrays with the same kl value
    # true_separate_klambda = []
    # spanet_separate_klambda = []

    kl_unique_true = np.unique(kl_array_true)

    # Filling two dictionaries with the events of the respective kl values
    # One for the true one and one for the spanet one
    jet_separate_klambda = []

    true_kl_idx_list = []
    kl_values = []
    for kl in kl_unique_true:
        mask = kl_array_true == kl
        # logger.info(f"mask {mask}")
        logger.info(f"kl_array_true[mask] {kl_array_true[mask]}")
        logger.info(f"kl {kl}")
        true_kl_idx_list.append(idx_true[mask])
        kl_values.append(kl)
        jet_separate_klambda.append(jet[mask])

    if df_spanet_pred is not None and idx_spanet_pred is not None:
        try:
            kl_array_spanet = df_spanet_pred["INPUTS"]["Event"]["kl"][()][mask_region]
        except KeyError:
            logger.info("Did not find Event/kl in kl_array_spanet, will try EVENT/kl")
            kl_array_spanet = df_spanet_pred["INPUTS"]["EVENT"]["kl"][()][mask_region]

        kl_unique_spanet = np.unique(kl_array_spanet)
        spanet_kl_idx_list = []
        for kl in kl_unique_spanet:
            mask = kl_array_spanet == kl
            # logger.info(f"mask {mask}")
            logger.info(f"kl_array_spanet[mask] {kl_array_spanet[mask]}")
            logger.info(f"kl {kl}")
            spanet_kl_idx_list.append(idx_spanet_pred[mask])
            assert kl in kl_values
    else:
        spanet_kl_idx_list = None

    # keep only two decimal in the kl_values
    kl_values = np.round(kl_values, 2)

    return (
        true_kl_idx_list,
        spanet_kl_idx_list,
        kl_values,
        jet_separate_klambda,
    )


##################
# PLOT FUNCTIONS #
##################
def plot_histos_1d(
    bins,
    spanet,
    run2=None,
    true=None,
    labels=None,
    colors=None,
    num=1,
    name="",
    plot_dir="plots",
    cmstext="Private",
    region=None,
):

    xlabel = r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"

    output_base = f"{plot_dir}/higgs_mass_{num}{name}"

    # ---------------------------------------------------------
    # Determine reference histogram
    # priority: true > run2 > first spanet
    # ---------------------------------------------------------

    reference_source = None
    reference_label = None

    if isinstance(true, ak.Array):
        reference_source = true
        reference_label = "True pairing"
        ratio_label = "Predicted / True"

    elif isinstance(run2, ak.Array):
        reference_source = run2
        reference_label = r"$D_{HH}$-method"
        ratio_label = r"Predicted / $D_{HH}$"

    else:
        reference_source = spanet[0]
        reference_label = labels[0]
        ratio_label = "Ratio"

    # ---------------------------------------------------------
    # helper to convert array -> hist.Hist normalized
    # ---------------------------------------------------------

    def make_hist(array):
        h = Hist.new.Variable(bins, name="mass", flow=False).Weight()
        weights = np.ones(len(array)) / len(array)
        array_numpy = ak.to_numpy(array, allow_missing=True)
        h.fill(array_numpy, weight=weights)
        return h

    # ---------------------------------------------------------
    # Build series_dict
    # ---------------------------------------------------------

    series_dict = {}

    # reference first
    series_dict[reference_label] = {
        "data": make_hist(reference_source),
        "style": {
            "color": "black",
            "histtype": "step",
            "is_reference": True,
        },
    }

    # spanet entries
    for arr, label, color in zip(spanet, labels, colors):

        # avoid duplicating reference
        if label == reference_label:
            continue

        series_dict[label] = {
            "data": make_hist(arr),
            "style": {
                "color": color,
                "histtype": "step",
            },
        }

    # add run2 if not reference
    if isinstance(run2, ak.Array) and reference_source is not run2:

        series_dict[r"$D_{HH}$-method"] = {
            "data": make_hist(run2),
            "style": {
                "color": "yellowgreen",
                "histtype": "step",
            },
        }

    # add true if not reference
    if isinstance(true, ak.Array) and reference_source is not true:

        series_dict["True pairing"] = {
            "data": make_hist(true),
            "style": {
                "color": "black",
                "histtype": "step",
            },
        }

    # ---------------------------------------------------------
    # Axis limits
    # ---------------------------------------------------------

    xlim = (None, None) if "peak" in name else (50, 300)

    # ---------------------------------------------------------
    # Run plot
    # ---------------------------------------------------------

    lumitext_prefix=r"$\kappa_{\lambda}$: " + name.split("_")[-1]
    (
        HEPPlotter("CMS")
        .set_plot_config(
            figsize=(10, 10),
            lumitext=f"{lumitext_prefix}      (13.6 TeV)",
        )
        .set_output(output_base)
        .set_labels(
            xlabel=xlabel,
            ylabel="Normalized events",
            ratio_label=ratio_label,
        )
        .set_data(series_dict, plot_type="1d")
        .set_options(
            legend_loc="upper right",
            legend_font_size=23,
            grid=True,
            set_xlim=True,
            set_ylim_ratio=1,
            xlim_left_value=xlim[0],
            xlim_right_value=xlim[1],
            ylim_top_factor=1.6 if "peak" in name else 1.2,
        )
        .set_plot_config(
            lumitext="(13.6 TeV)", cmstext=cmstext,
            figsize=[13, 13],
        )
        .add_annotation(0.05, 0.92, region or "inclusive", fontsize=20, ha="left", va="top")
        .run()
    )


def plot_mhh(bins, mhh, plot_dir="plots", name="mhh", cmstext="Private", region=None):

    # ---------------------------------------------------------
    # Create normalized histogram
    # ---------------------------------------------------------

    h = Hist.new.Variable(bins, name="mhh", flow=False).Weight()

    # density=True equivalent
    weights = np.ones(len(mhh)) / len(mhh)
    h.fill(mhh, weight=weights)

    series_dict = {
        "mhh": {
            "data": h,
            "style": {
                "histtype": "step",
                "linewidth": 1,
            },
        }
    }

    # ---------------------------------------------------------
    # Run plot
    # ---------------------------------------------------------

    (
        HEPPlotter("CMS")
        .set_output(f"{plot_dir}/{name}")
        .set_labels(
            xlabel=r"$m_{HH}$ [GeV]",
            ylabel="Normalized events",
        )
        .set_data(series_dict, plot_type="1d")
        .set_options(
            grid=True,
            legend=False,  # matches your original (only 1 entry)
        )
        .set_plot_config(
            lumitext="(13.6 TeV)", cmstext=cmstext,
            figsize=[13, 13],
        )
        .add_annotation(0.95, 0.92, region or "inclusive", fontsize=20, ha="right", va="top")
        .run()
    )


def plot_histos_2d(mh_bins, higgs, label, name, plot_dir="plots", cmstext="Private"):
    # NOTE: not tested

    # ---------------------------------------------------------
    # Extract masses
    # ---------------------------------------------------------

    m_lead = np.array(higgs[:, 0].mass)
    m_sub = np.array(higgs[:, 1].mass)

    # ---------------------------------------------------------
    # Create normalized 2D histogram
    # ---------------------------------------------------------

    h2 = (
        Hist.new.Variable(mh_bins, name="m_lead")
        .Variable(mh_bins, name="m_sub")
        .Weight()
    )

    weights = np.ones(len(m_lead)) / len(m_lead)
    h2.fill(m_lead, m_sub, weight=weights)

    series_dict = {
        f"{name} {label}": {
            "data": h2,
            "style": {
                "cmap": "GnBu",
            },
        }
    }

    # ---------------------------------------------------------
    # Run plot
    # ---------------------------------------------------------

    (
        HEPPlotter("CMS")
        .set_output(f"{plot_dir}/higgs_mass_2d_{name}_{label}")
        .set_labels(
            xlabel=r"Leading $m_{H}$ [GeV]",
            ylabel=r"Subleading $m_{H}$ [GeV]",
            cbar_label="Normalized counts",
        )
        .set_data(series_dict, plot_type="2d")
        .set_options(
            grid=True,
        )
        # 125 GeV cross lines
        .add_line("v", x=125, color="black", linewidth=1)
        .add_line("h", y=120, color="black", linewidth=1)
        # # circle around Higgs mass region
        # .add_circle(
        #     x=125,
        #     y=120,
        #     radius=30,
        #     edgecolor="black",
        #     facecolor="none",
        #     linewidth=1,
        # )
        .set_plot_config(
            lumitext="(13.6 TeV)", cmstext=cmstext,
            figsize=[13, 13],
        )
        .run()
    )


def plot_diff_eff(
    mhh_bins,
    efficiency,
    unc_efficiency,
    labels,
    colors,
    plot_dir,
    file_name,
    cmstext="Private",
    region=None,
):

    # ---------------------------------------------------------
    # Bin centers
    # ---------------------------------------------------------

    x = 0.5 * (mhh_bins[:-1] + mhh_bins[1:])

    # ---------------------------------------------------------
    # Build series_dict for graph plot
    # ---------------------------------------------------------

    series_dict = {}

    for eff, unc_eff, label, col in zip(efficiency, unc_efficiency, labels, colors):

        series_dict[label] = {
            "data": {
                "x": [x, None],
                "y": [eff, unc_eff],
            },
            "style": {
                "marker": ".",
                "linestyle": "-",
                "color": col,
            },
        }

    # ---------------------------------------------------------
    # Run plot
    # ---------------------------------------------------------
    (
        HEPPlotter("CMS")
        .set_output(f"{plot_dir}/{file_name}")
        .set_labels(
            xlabel=r"$m_{HH}$ [GeV]",
            ylabel="$\\varepsilon^{tot}$" if "tot" in file_name else "$\\varepsilon$",
        )
        .set_data(series_dict, plot_type="graph")
        .set_options(
            legend=True,
            legend_loc="lower right",
            grid=True,
            set_ylim=False,
        )
        .set_plot_config(
            lumitext="(13.6 TeV)", cmstext=cmstext,
            figsize=[13, 13],
        )
        .add_annotation(0.05, 0.92, region or "inclusive", fontsize=20, ha="left", va="top")
        .run()
    )


def plot_diff_eff_klambda(
    effs,
    unc_effs,
    kl_values,
    labels,
    colors,
    name,
    plot_dir="plots",
    xlabels=None,  # dict: {kl_value: "label"}
    cmstext="Private",
    region=None,
):
    """
    Parameters
    ----------
    xlabels : dict or None
        Dictionary mapping kl_values -> string labels.
        Example: { -1.0: "SM", 0.0: "BSM 1", 2.0: "BSM 2" }
    """

    # ---------------------------------------------------------
    # Build x-axis mapping if categorical mode
    # ---------------------------------------------------------

    if xlabels is not None:

        # preserve order of first appearance
        all_kls = []
        for kls in kl_values:
            for kl in kls:
                if kl not in all_kls:
                    all_kls.append(kl)

        kl_to_pos = {kl: i for i, kl in enumerate(all_kls)}
        positions = np.arange(len(all_kls))

        # final tick labels
        tick_labels = [xlabels.get(str(kl), str(kl)) for kl in all_kls]

        xlabel = ""  # categorical → no numeric xlabel

    else:
        xlabel = r"$\kappa_{\lambda}$"
        tick_labels = None
        positions = None

    # ---------------------------------------------------------
    # Build series_dict for graph plotting
    # ---------------------------------------------------------

    series_dict = {}

    for label, kls, eff, unc_eff, col in zip(labels, kl_values, effs, unc_effs, colors):

        if xlabels is None:
            x = kls
        else:
            x = [kl_to_pos[kl] for kl in kls]

        series_dict[label] = {
            "data": {
                "x": [np.array(x), None],
                "y": [np.array(eff), np.array(unc_eff)],
            },
            "style": {
                "linestyle": "-",
                "marker": ".",
                "color": col,
            },
        }

    # ---------------------------------------------------------
    # Create plotter
    # ---------------------------------------------------------

    plotter = (
        HEPPlotter("CMS")
        .set_output(f"{plot_dir}/{name}")
        .set_labels(
            xlabel=xlabel,
            ylabel="$\\varepsilon^{tot}$" if "tot" in name else "$\\varepsilon$",
            xticklabels=tick_labels,
            label_pos=positions,
            xtick_fontsize=11,
        )
        .set_data(series_dict, plot_type="graph")
        .set_options(
            legend=True,
            legend_font_size=20,
            legend_loc="lower left",
            grid=True,
            set_ylim=False,
        )
        .set_plot_config(
            lumitext="(13.6 TeV)", cmstext=cmstext,
            figsize=[13, 13],
        )
        .add_annotation(0.95, 0.92, region or "inclusive", fontsize=20, ha="right", va="top")
    )

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    plotter.run()
