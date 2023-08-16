
import numpy as np, scipy, scipy.stats, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, openbabel, openbabel.pybel

def read_gnina(fp, agg=False):
    # agg: aggregate scores over all docking modes
    def get_(mol_, key_):
        return float(mol_.data[key_]) if key_ in mol_.data else float('NaN')

    cols_ = ['ligand_id', 'pose_id', 'minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS']
    df_ = pd.DataFrame.from_records([[
        mol.title,
        pose_id,
        -get_(mol, 'minimizedAffinity'),
        get_(mol, 'CNNscore'),
        get_(mol, 'CNNaffinity'),
        get_(mol, 'CNN_VS'),
    ] for (pose_id, mol) in enumerate(openbabel.pybel.readfile('sdf', fp), start=1) ], columns=cols_)

    if agg:
        return df_.groupby('ligand_id').agg(
            minimizedAffinity = ('minimizedAffinity', np.nanmax),
            CNNscore = ('CNNscore', np.nanmax),
            CNNaffinity = ('CNNaffinity', np.nanmax),
            CNN_VS = ('CNN_VS', np.nanmax),
        )
    else:
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
