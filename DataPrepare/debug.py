import pandas as pd
import selfies as sf
from tqdm import tqdm

save_path = '/data2/tianang/projects/mdlm/temp_data/small_molecules/strain_BAA-3170.txt'

SMILES_SM_raw = pd.read_csv('/data2/tianang/projects/Synergy/DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data.csv')['SMILES'].values

with open(save_path, 'w') as f:
    for SMILES in tqdm(SMILES_SM_raw, desc='processing'):
        f.write(sf.encoder(SMILES) + '\n')

print('done')