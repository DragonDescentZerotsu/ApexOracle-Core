import pandas as pd
from DataPrepare.smiles_to_peptide import smiles_to_pepseq
from tqdm import tqdm

df = pd.read_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles.csv')

original_columns = df.columns.tolist()
new_columns = original_columns + 'label'

data = df.values

labeled_data = []
for id, smiles in tqdm(data, desc='judging peptides'):
    _, pep_seq = smiles_to_pepseq(smiles)
    if pep_seq is None or 'X' in pep_seq:
        label = 0
    else:
        label = 1

    labeled_data.append([id, smiles, label])

df_labeled = pd.DataFrame(labeled_data, columns=new_columns)

print(f'saving')
df_labeled.to_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles_pep_SM_cls_v2.csv', index=False)