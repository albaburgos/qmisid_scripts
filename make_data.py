from pathlib import Path

import pandas as pd
import uproot
import awkward as ak
import vector
from pandas import DataFrame
from tqdm import tqdm

from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder

def prep_truth_and_reco_data(fil):
    with uproot.open(fil) as f:
        truth_data = f['reco'].arrays([
            'truth_el_pt',
            'truth_el_eta',
            'truth_el_phi',
            'truth_el_charge',
            'truth_el_MCTC_isPrompt',
        ])
        reco_data = f['reco'].arrays([
            'el_charge',
            'el_pt_NOSYS',
            'el_eta',
            'el_phi',
        ])

    truth_data = truth_data[truth_data['truth_el_MCTC_isPrompt'] == 1]
    truth_vecs = vector.zip({
        'pt': truth_data['truth_el_pt'],
        'eta': truth_data['truth_el_eta'],
        'phi': truth_data['truth_el_phi'],
        'mass': 0,
    })
    reco_vecs = vector.zip({
        'pt': reco_data['el_pt_NOSYS'],
        'eta': reco_data['el_eta'],
        'phi': reco_data['el_phi'],
        'mass': 0,
    })
    deltaR = truth_vecs[:, :, None].deltaR(reco_vecs[:, None, :])
    is_matched = deltaR < 0.2
    min_dists = ak.argmin(deltaR, axis=-1)
    num_reco_matches = ak.sum(is_matched, axis=2)
    num_truth_matches = ak.sum(is_matched, axis=1)
    matched_reco = reco_data[min_dists]
    contains_double_match_mask = ak.max(num_truth_matches, axis=1) < 2
    print(f"Discarding {1 - ak.mean(contains_double_match_mask)}% events due to double truth match")
    truth_data = truth_data[num_reco_matches == 1][contains_double_match_mask]
    matched_reco = matched_reco[num_reco_matches == 1][contains_double_match_mask]
    return truth_data, matched_reco


def prep_df(fil):
    truth_data, matched_reco = prep_truth_and_reco_data(fil)
    truth_data['is_flipped'] = truth_data['truth_el_charge'] != matched_reco['el_charge']
    truth_data['truth_el_q_over_pt'] = truth_data['truth_el_charge'] / truth_data['truth_el_pt']

    dict_ = {k: ak.flatten(truth_data[k]) for k in truth_data.fields}
    dict_.update({k: ak.flatten(matched_reco[k]) for k in matched_reco.fields})
    return DataFrame(dict_)


def prep_reco_df(fil):
    with uproot.open(fil) as f:
        reco_data = f['reco'].arrays([
            'el_charge',
            'el_pt_NOSYS',
            'el_eta',
            'el_phi',
        ])
    reco_data = reco_data[ak.num(reco_data['el_pt_NOSYS']) == 2]
    reco_data['el_q_over_pt'] = reco_data['el_charge'] / reco_data['el_pt_NOSYS']
    df = pd.DataFrame()
    for i in range(2):
        for k in reco_data.fields:
            df[f'{k}_{i}'] = reco_data[k][:, i]

    df['label'] = df['el_charge_0'] == df['el_charge_1']
    return df


def prep_qmisid_single_df(fil):
    with uproot.open(fil) as f:
        branches = f['qmisid_cr'].arrays([
            'l1_pdg',
            'l2_pdg',
            'l1_pt',
            'l2_pt',
            'l1_eta',
            'l2_eta',
            'l1_isQMisID',
            'l2_isQMisID',
            'm_l1l2',
        ])

    branches['l1_q_over_pt'] = branches['l1_pdg'] / (11*branches['l1_pt'])
    branches['l2_q_over_pt'] = branches['l2_pdg'] / (11*branches['l2_pt'])
    mask = (branches["m_l1l2"] >= 81000) & (branches["m_l1l2"] <= 101000)
    df = pd.DataFrame()
    exclude = {"l1_pdg", "l2_pdg", "l1_pt", "l2_pt"}

    for k in ["l1_q_over_pt","l2_q_over_pt"]:
        df [f'Pt'] = branches[k][:][mask]

    for k in ["l1_eta","l2_eta"]:
        df [f'eta'] = branches[k][:][mask]

    for k in ["l1_isQMisID","l2_isQMisID"]:
        df [f'label'] = branches[k][:][mask]

    return df

def prep_qmisid_df(fil):
    with uproot.open(fil) as f:
        branches = f['qmisid_cr'].arrays([
            'same_charge',
            'opposite_charge',
            'l1_pdg',
            'l2_pdg',
            'l1_pt',
            'l2_pt',
            'l1_eta',
            'l2_eta',
            'l1_isQMisID',
            'l2_isQMisID',
            'm_l1l2',
        ])
    
    branches['l1_q_over_pt'] = branches['l1_pdg'] / (11*branches['l1_pt'])
    branches['l2_q_over_pt'] = branches['l2_pdg'] / (11*branches['l2_pt'])
    branches["qmisid"] = (branches["l1_isQMisID"] == 1) | (branches["l2_isQMisID"] == 1)

    mask = (branches["m_l1l2"] >= 81000) & (branches["m_l1l2"] <= 101000)
    df = pd.DataFrame()
    exclude = {"l1_pdg", "l2_pdg", "l1_pt", "l2_pt"}
    for k in branches.fields:
        if k in exclude: continue
        df[f'{k}'] = branches[k][:][mask]
        print(k)

    df['label'] = branches["opposite_charge"][mask]
    return df


def prep_shift(fil):
    with uproot.open(fil) as f:
        branches = f['qmisid_cr'].arrays([
            'same_charge',
            'opposite_charge',
            'l1_pdg',
            'l2_pdg',
            'l1_pt',
            'l2_pt',
            'l1_eta',
            'l2_eta',
            'l1_isQMisID',
            'l2_isQMisID',
            'm_l1l2',
        ])
    
    branches['l1_q_over_pt'] = branches['l1_pdg'] / (11*branches['l1_pt'])
    branches['l2_q_over_pt'] = branches['l2_pdg'] / (11*branches['l2_pt'])
    mask = (branches["m_l1l2"] >= 81000) & (branches["m_l1l2"] <= 101000)
    
    mask1 = branches["same_charge"] == 1
    
    mass_shift = ak.where(
    mask1,
    branches["m_l1l2"]*0.97,   
    branches["m_l1l2"]*1.00  
)
    
    mask_l1 = (branches["same_charge"] == 1) & (branches["l1_isQMisID"] == 1)
    mask_l2 = (branches["same_charge"] == 1) & (branches["l2_isQMisID"] == 1)

    pt_l1_shift = ak.where(
        mask_l1,
        branches["l1_q_over_pt"] * 0.97,
        branches["l1_q_over_pt"]
    )

    pt_l2_shift = ak.where(
        mask_l2,
        branches["l2_q_over_pt"] * 0.97,
        branches["l2_q_over_pt"]
    )
        
    df = pd.DataFrame()
    exclude = {"l1_pdg", "l2_pdg", "l1_pt", "l2_pt", "m_l1l2", 'l1_q_over_pt', 'l2_q_over_pt'}
    for k in branches.fields:
        if k in exclude: continue
        df[f'{k}'] = branches[k][:][mask]
        print(k)

    df['label'] = branches["opposite_charge"][mask]
    df['m_l1l2'] = mass_shift[mask]
    df['l2_q_over_pt'] = pt_l2_shift[mask]
    df['l1_q_over_pt']= pt_l1_shift[mask]

    return df

if __name__ == '__main__':

    root_dir = Path("/Users/albaburgosmondejar/Desktop/Input2")
    with PandasDirDataModuleBuilder(
        "/Users/albaburgosmondejar/Desktop/DatasetShift",
        force=True,
        file_size=int(5e6),
    ) as builder:
        for root_file in tqdm(list(root_dir.glob("*.root"))):
            df = prep_shift(root_file)
            builder.write(df)