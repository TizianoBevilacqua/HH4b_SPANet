import awkward as ak
import numpy as np
import logging
import importlib.util
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

H5_PADDING_VALUE = 9999.0


def import_module_from_path(module_path):
    module_path = Path(module_path).resolve()

    module_name = module_path.stem  # filename without .py

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def setup_logging(logpath):
    """Create logger for nicer logger.infoing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%d-%b-%y %H-%M-%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{logpath}/logger_output.log", mode="a", encoding="utf-8"
            ),
        ],
    )

# TODO: Currently we only have the values for postEE!!!!
def get_region_mask(region, column_file, do_vbf_pairing, jet_coll_higgs="Jet", jet_coll_vbf=None, n_higgs_jets=4):
    if region == "inclusive":
        return ak.ones_like(column_file["INPUTS"][jet_coll_higgs]["MASK"][:, 0])

    if "test" in region:
        # keep only the first N events
        num_events = int(region.split("test_")[-1])
        logger.info(f"Keeping only the first {num_events} events!")
        mask = ak.concatenate(
            (
                ak.ones_like(column_file["INPUTS"][jet_coll_higgs]["MASK"][:, 0])[:num_events],
                ak.zeros_like(column_file["INPUTS"][jet_coll_higgs]["MASK"][:, 0])[num_events:],
            ),
        )
        return mask

    jet_btag = column_file["INPUTS"][jet_coll_higgs]["btagPNetB"]
    if region == "4b" or region == "4M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] > 0.2605)
            & (jet_btag[:, 3] > 0.2605)
        )
    elif region == "3M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] > 0.2605)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "2M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] < 0.2605)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "3T1M":
        mask = (
            (jet_btag[:, 0] > 0.6915)
            & (jet_btag[:, 1] > 0.6915)
            & (jet_btag[:, 2] > 0.6915)
            & (jet_btag[:, 3] > 0.2605)
        )
    elif region == "3T1L":
        mask = (
            (jet_btag[:, 0] > 0.6915)
            & (jet_btag[:, 1] > 0.6915)
            & (jet_btag[:, 2] > 0.6915)
            & (jet_btag[:, 3] > 0.0499)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "vbf_presel" and do_vbf_pairing:
        mask = get_mask_vbf_region(column_file, 400, 3.5, jet_coll=jet_coll_vbf if jet_coll_vbf else jet_coll_higgs, n_higgs_jets=n_higgs_jets)
    elif region == "vbf_no_kin_cuts" and do_vbf_pairing:
        mask = get_mask_vbf_region(column_file, 0, 0, jet_coll=jet_coll_vbf if jet_coll_vbf else jet_coll_higgs, n_higgs_jets=n_higgs_jets)
    else:
        raise ValueError(f"Undefined region {region}!")
    return mask


def get_mask_vbf_region(column_file, mjj_cut, delta_eta_cut, jet_coll="Jet", n_higgs_jets=4):
    jet_list = [
        get_jet_4vec(
            column_file, ak.ones_like(column_file["INPUTS"][jet_coll]["MASK"][:, 0]), jet_coll=jet_coll
        )
    ]
    jet_vbf = [j[:, n_higgs_jets:] for j in jet_list]
    idx_vbf_lead_mjj = get_lead_mjj_jet_idx(jet_vbf)[0]
    jet = jet_list[0]
    vbf_jets_max_mjj_0 = ak.unflatten(
        jet[
            ak.local_index(jet, axis=0),
            idx_vbf_lead_mjj[:, 0],
        ],
        1,
    )
    vbf_jets_max_mjj_1 = ak.unflatten(
        jet[
            ak.local_index(jet, axis=0),
            idx_vbf_lead_mjj[:, 1],
        ],
        1,
    )
    delta_eta = abs(vbf_jets_max_mjj_0.eta - vbf_jets_max_mjj_1.eta)
    mjj = (vbf_jets_max_mjj_0 + vbf_jets_max_mjj_1).mass

    mask = ak.flatten((mjj > mjj_cut) & (delta_eta > delta_eta_cut))
    mask = ak.where(ak.is_none(mask), False, mask)
    
    return mask


def get_class_mask(class_label, column_file, jet_coll="Jet"):
    if class_label:
        try:
            class_array = column_file["CLASSIFICATIONS"]["EVENT"]["class"][()].astype(
                np.int64
            )
            mask = class_array == int(class_label)
        except KeyError:
            try:
                logger.info(
                    'The file doesn\'t contain an "EVENT" array, so trying with "Event" instead'
                )
                class_array = column_file["CLASSIFICATIONS"]["Event"]["class"][
                    ()
                ].astype(np.int64)
                mask = class_array == int(class_label)
            except KeyError:
                logger.warning(
                    "The file doesn't contain a class array. Setting the mask for the class to True ..."
                )
                mask = ak.ones_like(column_file["INPUTS"][jet_coll]["MASK"][:, 0])
    else:
        mask = ak.ones_like(column_file["INPUTS"][jet_coll]["MASK"][:, 0])

    return mask



def get_jet_4vec(truefile, mask_true, jet_coll="Jet"):
    # load jet information
    try:
        jet_pt = truefile["INPUTS"][jet_coll]["ptPnetRegNeutrino"][()][mask_true]
    except KeyError:
        logger.warning("Did not find ptPnetRegNeutrino, will try to load pt normal")
        jet_pt = truefile["INPUTS"][jet_coll]["pt"][()][mask_true]

    jet_eta = truefile["INPUTS"][jet_coll]["eta"][()][mask_true]
    jet_phi = truefile["INPUTS"][jet_coll]["phi"][()][mask_true]
    jet_mass = truefile["INPUTS"][jet_coll]["mass"][()][mask_true]

    jet_infos = [jet_pt, jet_eta, jet_phi, jet_mass]
    for i in range(len(jet_infos)):
        jet_infos[i] = ak.mask(jet_infos[i], jet_infos[i] != H5_PADDING_VALUE)

    # create a LorentzVector for the jets
    jet = ak.zip(
        {
            "pt": jet_infos[0],
            "eta": jet_infos[1],
            "phi": jet_infos[2],
            "mass": jet_infos[3],
            "index": ak.local_index(jet_infos[0], axis=1),
        },
        with_name="Momentum4D",
    )
    return jet


def get_lead_mjj_jet_idx(jet):
    # choose vbf jets as the two jets with the highest mjj that are not from higgs decay

    jet_combinations = [ak.combinations(j, 2) for j in jet]
    jet_combinations_mass = [(jc["0"] + jc["1"]).mass for jc in jet_combinations]
    jet_combinations_mass_max_idx_ak = [
        ak.firsts(ak.argsort(jcm, axis=1, ascending=False))
        for jcm in jet_combinations_mass
    ]
    assert all([ak.all(~ak.is_none(jcm)) for jcm in jet_combinations_mass_max_idx_ak])
    jet_combinations_mass_max_idx = [
        ak.to_numpy(ak.fill_none(jcm, -1)) for jcm in jet_combinations_mass_max_idx_ak
    ]
    jets_max_mass = [
        jc[ak.local_index(jc, axis=0), jcm]
        for jc, jcm in zip(jet_combinations, jet_combinations_mass_max_idx)
    ]
    comb_idx_min_vbf = [
        ak.to_numpy(
            ak.concatenate(
                [
                    ak.unflatten(jmm["0"].index, 1),
                    ak.unflatten(jmm["1"].index, 1),
                ],
                axis=1,
            ),
        )
        for jmm in jets_max_mass
    ]

    return comb_idx_min_vbf

