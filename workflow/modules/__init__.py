
import itertools
import subprocess
import openbabel.pybel
import pandas as pd
import os, os.path
import numpy as np, scipy, scipy.stats, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, openbabel, openbabel.pybel
import sklearn.preprocessing
from pprint import pprint

DHODH_seq = 'MAWRHLKKRAQDAVIILGGGGLLFASYLMATGDERFYAEHLMPTLQGLLDPESAHRLAVRFTSLGLLPRARFQDSDMLEVRVLGHKFRNPVGIAAGFDKHGEAVDGLYKMGFGFVEIGSVTPKPQEGNPRPRVFRLPEDQAVINRYGFNSHGLSVVEHRLRARQQKQAKLTEDGLPLGVNLGKNKTSVDAAEDYAEGVRVLGPLADYLVVNVSSPNTAGLRSLQGKAELRRLLTKVLQERDGLRRVHRPAVLVKIAPDLTSQDKEDIASVVKELGIDGLIVTNTTVSRPAGLQGALRSETGGLSGKPLRDLSTQTIREMYALTQGRVPIIGVGGVSSGQDALEKIRAGASLVQLYTALTFWGPPVVGKVKRELEALLKEQGFGGVTDAIGADHRR'

library_id = [
    'dude_actives',
    'dude_decoys_a',
    'dude_decoys_b',
    'dude_decoys_c',
    'dude_decoys_d',
    'dude_decoys_e',
    'dude_decoys_f',
    'dude_decoys_g',
    'dude_decoys_h',
    'dude_decoys_i',
    'dude_decoys_j',
    'prestwick_a',
    'prestwick_b',
    'prestwick_c',
]

def read_gnina(fp):
    def get_(mol_, key_):
        return float(mol_.data[key_]) if key_ in mol_.data else float('NaN')
    cols_ = ['ligand_id', 'minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS']
    df_ = pd.DataFrame.from_records([[
        mol.title,
        #pose_id,
        get_(mol, 'minimizedAffinity'),
        get_(mol, 'CNNscore'),
        get_(mol, 'CNNaffinity'),
        get_(mol, 'CNN_VS'),
    ] for mol in openbabel.pybel.readfile('sdf', fp) ], columns=cols_)
    # Re-create pose_id to match output table shown during docking
    df_.insert(loc=1, column='mode_id', value=df_.groupby('ligand_id', as_index=False).cumcount() + 1)
    return df_

def read_ligands_charge(fp_):
    return pd.read_csv(fp_, sep='\t', names=['smiles', 'input_id', 'protonation_id', 'mwt', 'logp', 'rb', 'hba', 'hbd', 'charge'])

def read_dude():
    df_charge = read_ligands_charge('resources/DHODH_assay_decoys/dude-decoys/ligands.charge').replace('Tiratricol,_3,3_,5-triiodothyroacetic_acid', 'Tiratricol')
    df_charge['picked_file'] = [f'resources/DHODH_assay_decoys/dude-decoys/decoys/decoys.{r["protonation_id"]}.picked' for i, r in df_charge.iterrows() ]
    df_charge['compounds_id'] = [f'{r.input_id}_{r.protonation_id}' for i, r in df_charge.iterrows() ]
    return df_charge

def read_decoys(fp_):
    with open(fp_) as fh_:
        lig_ = fh_.read(7)
        assert lig_ == 'ligand\t'
        df_ = pd.read_csv(fh_, sep='\t', names=['smiles', 'name', 'protonation_id'])
        df_['compounds_id'] = [f'{r["name"]}_{r.protonation_id}' for i, r in df_.iterrows() ]
        return df_

def read_affinities(fp):
    # https://doi.org/10.1186/1752-153X-2-5
    return ( -float(mol.data['minimizedAffinity']) for mol in openbabel.pybel.readfile('sdf', fp) )

def read_affinities_dude(fp, label):#input_id, protonation_id):
    df_ = pd.DataFrame({'minimizedAffinity': read_affinities(fp)})
    df_['decoyNormalisedAffinity'] = scipy.stats.zscore(df_['minimizedAffinity'])
    df_.insert(loc=0, column='label', value=label)
    #df_.insert(loc=0, column='input_id', value=input_id)
    #df_.insert(loc=1, column='protonation_id', value=protonation_id)
    df_['is_decoy'] = True
    df_.at[0, 'is_decoy'] = False
    return df_

def any_check(x):
    # Check that all elements in iterator are identical & return any/first one
    assert len(set(iter(x))) == 1
    return next(iter(x))

def smi_largest_component(smiles):
    #https://github.com/dkoes/rdkit-scripts/blob/master/rdconf.py#L80
    if '.' in smiles:
        return max(smiles.split('.'), key=len)
    else:
        return smiles

