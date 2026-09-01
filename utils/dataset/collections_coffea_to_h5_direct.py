KEEP_TOGETHER_COLLECTIONS = ["add_jet1pt"]

jet_collections_dict = {
    "JET_COLLECTIONS_SEPARATE_HIGGS_VBF": [
        {
            "JetGoodProvHiggsPtFlattenPadded": {
                "saved_name": "JetHiggs",
                "max_num_jets": 4,
                "resonances": ["h1", "h2"],
                "prov_key": "provenance",
                "min_num_jets": 4,
            },
            "JetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name": "JetVBF",
                "max_num_jets": 5,
                "resonances": ["vbf"],
                "prov_key": "provenance",
                "min_num_jets": 0,
            },
        },
        {
            "JetGoodProvHiggsPtFlattenPadded": {
                "saved_name": "JetHiggs",
                "max_num_jets": 4,
                "resonances": ["h1", "h2"],
                "prov_key": "provenance",
                "min_num_jets": 4,
            },
            "JetGoodVBFMergedProvVBFPadded": {
                "saved_name": "JetVBF",
                "max_num_jets": 5,
                "resonances": ["vbf"],
                "prov_key": "provenance",
                "min_num_jets": 0,
            },
        },
        {
            "JetGoodProvHiggsPadded": {
                "saved_name": "JetHiggs",
                "max_num_jets": 4,
                "resonances": ["h1", "h2"],
                "prov_key": "provenance",
                "min_num_jets": 4,
            },
            "JetGoodVBFMergedProvVBFPadded": {
                "saved_name": "JetVBF",
                "max_num_jets": 5,
                "resonances": ["vbf"],
                "prov_key": "provenance",
                "min_num_jets": 0,
            },
        },
    ],
    "JET_COLLECTIONS_SEPARATE_HIGGS_VBF_MERGED_COLL": [
        {
            "JetTotalSPANetSeparateProvHiggsVBFPtFlattenPadded": {
                "saved_name": "Jet",
                "max_num_jets": 9,
                "resonances": ["h1", "h2", "vbf"],
                "prov_key": "provenance",
            },
        },
        {
            "JetTotalSPANetSeparateProvHiggsVBFPtFlattenOnlyHiggsPadded": {
                "saved_name": "Jet",
                "max_num_jets": 9,
                "resonances": ["h1", "h2", "vbf"],
                "prov_key": "provenance",
            },
        },
        {
            "JetTotalSPANetSeparateProvHiggsVBFPadded": {
                "saved_name": "Jet",
                "max_num_jets": 9,
                "resonances": ["h1", "h2", "vbf"],
                "prov_key": "provenance",
            },
        },
    ],
    "JET_COLLECTIONS_SIG_BKG_CLASSIFIER": [
        {
            "JetGoodFromHiggsOrderedLeading": {
                "saved_name": "JetHiggsLeading",
                "max_num_jets": 2,
                "resonances": ["h1"],
                "prov_key": "reco_provenance",
                "min_num_jets": 2,
            },
            "JetGoodFromHiggsOrderedSubLeading": {
                "saved_name": "JetHiggsSubLeading",
                "max_num_jets": 2,
                "resonances": ["h2"],
                "prov_key": "reco_provenance",
                "min_num_jets": 2,
            },
            "add_jet1pt": {
                "saved_name": "Jet5th",
                "max_num_jets": 1,
                "resonances": ["add"],
                "prov_key": "reco_provenance",
                "min_num_jets": 0,
            },
        },
    ],
    "JET_COLLECTIONS_VBF_PAIRING_AFTER_HIGGS_PAIRING": [
        {
            "JetGoodVBFMergedProvVBFPadded": {
                "saved_name": "Jet",
                "max_num_jets": 5,
                "resonances": ["vbf"],
                "min_num_jets": 0,
                "prov_key": "provenance",
            },
        },
    ],
    "JET_COLLECTIONS_VBF_PAIRING_AFTER_HIGGS_PAIRING_TOTAL": [
        # {
        #     "JetGoodVBFMergedProvVBFPadded": {
        #         "saved_name": "Jet",
        #         "max_num_jets": 5,
        #         "resonances": ["vbf"],
        #         "min_num_jets": 0,
        #         "prov_key": "provenance_vbf",
        #     },
        # },
        # {
        #     "JetGoodVBFMergedProvVBFPadded": {
        #         "saved_name": "JetVBF",
        #         "max_num_jets": 5,
        #         "resonances": ["vbf"],
        #         "min_num_jets": 0,
        #         "prov_key": "provenance_vbf",
        #     },
        #     "JetGoodProvHiggsPadded": {
        #         "saved_name": "JetHiggs",
        #         "max_num_jets": 4,
        #         "resonances": [],
        #         "min_num_jets": 4,
        #         "prov_key": "provenance_higgs",
        #     },
        # },
        {
            "JetTotalSPANetPadded": {
                "saved_name": "Jet",
                "max_num_jets": 9,
                "resonances": ["vbf"],
                "min_num_jets": 4,
                "prov_key": "provenance_vbf",
            },
        },
    ],
    "JET_COLLECTION_VBF_BOOSTED": [
        {
            "JetGoodVBF": {
                "saved_name": "JetVBF",
                "resonances": ["vbf"],
                "min_num_jets": 0,
                "max_num_jets": 2,
                "prov_key": "provenance",
            },
            # "JetGoodVBFCandidates": {
            #     "saved_name": "JetVBFCandidates",
            #     "resources": ["vbf"],
            #     "min_num_jets": 0,
            #     "max_num_jets": 5,
            #     "prov_key": "provenance",
            # }
        }
    ]
}

global_collections_dict = {
    "GLOBAL_COLLECTIONS_VBF": [
        {
            "events_mjjJetTotalSPANetPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetTotalSPANetPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingRun2JetTotalSPANetPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingRun2VBF",
            },
            "events_centralityHiggsSubLeadingRun2JetTotalSPANetPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingRun2VBF",
            },
        },
        {
            "events_mjjJetTotalSPANetPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetTotalSPANetPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingRun2JetTotalSPANetPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingRun2VBF",
            },
            "events_centralityHiggsSubLeadingRun2JetTotalSPANetPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingRun2VBF",
            },
        },
    ],
    "GLOBAL_COLLECTIONS_SEPARATE_HIGGS_VBF": [
        {
            "events_mjjJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingVBF",
            },
            "events_centralityHiggsSubLeadingJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingVBF",
            },
            "__save_all_remaining__": True,
        },
        {
            "events_mjjJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingVBF",
            },
            "events_centralityHiggsSubLeadingJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingVBF",
            },
            "__save_all_remaining__": True,
        },
        {
            "events_mjjJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingVBF",
            },
            "events_centralityHiggsSubLeadingJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingVBF",
            },
            "__save_all_remaining__": True,
        },
    ],
    "GLOBAL_COLLECTIONS_VBF_MERGED_COLL": [
        {
            "events_mjjJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingRun2JetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingRun2VBF",
            },
            "events_centralityHiggsSubLeadingRun2JetGoodVBFMergedProvVBFPtFlattenPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingRun2VBF",
            },
        },
        {
            "events_mjjJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingRun2JetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingRun2VBF",
            },
            "events_centralityHiggsSubLeadingRun2JetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingRun2VBF",
            },
        },
        {
            "events_mjjJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "mjjVBF",
            },
            "events_detaJetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "detaVBF",
            },
            "events_centralityHiggsLeadingRun2JetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsLeadingRun2VBF",
            },
            "events_centralityHiggsSubLeadingRun2JetGoodVBFMergedProvVBFPadded": {
                "saved_name_coll": "Event",
                "saved_name_var": "centralityHiggsSubLeadingRun2VBF",
            },
        },
    ],
    
}
