from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import copy

json_path = './Data/all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict = {}
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

huge_bact_names = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853", "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus", "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa", "Pseudomonas aeruginosa PA01", "Enterococcus faecalis ATCC 29212", "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228", "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603", "Staphylococcus aureus ATCC 43300", "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028", "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans", "Klebsiella pneumoniae"]

df = pd.read_csv('./Data/DBAASP_id_SMILES_merged.csv')

DBAASPid_SMILES = np.array(df[['DBAASP_id', 'SMILES']].values.tolist())
for AMP in tqdm(data):
    if str(AMP['id']) not in DBAASPid_SMILES[:, 0]:
        continue

    line_index = np.where(DBAASPid_SMILES[:, 0] == str(AMP['id']))[0][0]
    bact_measure_value_unit = {}
    if AMP['targetActivities'] is not None:
        for bact in AMP['targetActivities']:
            # if bact['unit'] is not None:
            #     if (bact['activityMeasureValue'], bact['unit']['name']) not in list(measure_unit.keys()):
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] = 1
            #     else:
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] += 1
            bact_name = bact['targetSpecies']['name']
            if bact_name in huge_bact_names:
                if bact_name not in list(bact_measure_value_unit.keys()):
                    bact_measure_value_unit[bact_name] = {}
                if bact['unit'] is not None:
                    if (bact['activityMeasureValue'], bact['unit']['name']) not in list(bact_measure_value_unit[bact_name].keys()):
                        if bact['activityMeasureValue'] == 'MIC' or 'inhibition' in bact['activityMeasureValue'] or 'inhibiton' in bact['activityMeasureValue']:
                            bact_measure_value_unit[bact_name][(bact['activityMeasureValue'], bact['unit']['name'])] = bact['concentration']
        # print(bact_measure_value_unit)
        for bact_name, value in list(bact_measure_value_unit.items()):
            if len(value) == 0:
                del bact_measure_value_unit[bact_name]
                continue

            if ('MIC', 'µM') in value.keys():
                concentration = value[('MIC', 'µM')]
                if ' ' in concentration.strip():
                    concentration = concentration.strip()
                    concentration = float(concentration.split(' ')[0]+concentration.split(' ')[-1])
                elif '>=' in concentration:
                    concentration = float(concentration.split('>=')[-1])*2
                elif '>' in concentration:
                    concentration = float(concentration.split('>')[-1]) + 156
                elif '-' in concentration:
                    if '<' in concentration:
                        concentration = concentration.split('<')[-1]
                    concentration = sum(float(comp) for comp in concentration.split('-')) / len(concentration.split('-'))
                elif '±' in concentration:
                    concentration = float(concentration.split('±')[0])
                elif '<=' in concentration:
                    concentration = float(concentration.split('<=')[-1])
                elif '<' in concentration:
                    if concentration[-1] == '<':
                        concentration = float(concentration.split('<')[0])
                    else:
                        concentration = float(concentration.split('<')[-1])
                elif ',' in concentration:
                    concentration = concentration.replace(',', '.')
                    concentration = float(concentration)
                else:
                    concentration = float(concentration)
                bact_measure_value_unit[bact_name] = {('MIC', 'µM'): concentration}

            elif ('MIC', 'µg/ml') in value.keys():
                concentration = value[('MIC', 'µg/ml')]

                mol = Chem.MolFromSmiles(DBAASPid_SMILES[line_index, 1])
                if mol is None:
                    print(AMP['id'])
                mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
                if '>=' in concentration:
                    if '±' in concentration:
                        concentration = concentration.split('±')[0]
                    concentration = float(concentration.split('>=')[-1]) * 1.5
                elif '>>' in concentration:
                    if '±' in concentration:
                        concentration = concentration.split('±')[0]
                    concentration = float(concentration.split('>>')[-1]) * 3
                elif '>' in concentration:
                    if '±' in concentration:
                        concentration = concentration.split('±')[0]
                    concentration = float(concentration.split('>')[-1]) * 2
                elif '<=' in concentration:
                    if '±' in concentration:
                        concentration = concentration.split('±')[0]
                    concentration = float(concentration.split('<=')[-1])
                elif '<' in concentration:
                    if '±' in concentration:
                        concentration = concentration.split('±')[0]
                        concentration = float(concentration.split('<')[-1])
                    if '-' in concentration:
                        # print(AMP['id'])
                        concentration = concentration.split('<')[-1]
                        concentration = sum(float(comp) for comp in concentration.split('-')) / len(concentration.split('-'))
                    # concentration = float(concentration.split('<')[-1])
                elif '-' in concentration:
                    if ' - =>' in concentration:
                        concentration = '12-12'
                    elif '<' in concentration:
                        concentration = concentration.split('<')[-1]
                    elif '>' in concentration:
                        concentration = concentration.split('>')[-1] * 2
                    print(AMP['id'])
                    concentration = sum(float(comp) for comp in concentration.split('-')) / len(concentration.split('-'))
                elif '±' in concentration:
                    concentration = concentration.split('±')[0]
                    if '>>' in concentration:
                        concentration = float(concentration.split('>>')[-1]) * 3
                    elif '<=' in concentration:
                        concentration = float(concentration.split('<=')[-1])
                    concentration = float(concentration)
                elif '>>' in concentration:
                    concentration = float(concentration.split('>>')[-1]) * 3
                else:
                    concentration = float(concentration)
                uM_concentration = float(concentration) / mol_wt * 1000
                bact_measure_value_unit[bact_name] = {('MIC', 'µM'): uM_concentration}

            # 那只能是 inhibition 或者 inhibiton 在里面了
            else:
                inhibition_percent_max = 0
                MIC_final = 0
                for (value_name, unit_name), concentration in list(value.items()):
                    inhibition_percent = value_name.split('%')[0]
                    if '±' in inhibition_percent:
                        inhibition_percent = float(inhibition_percent.split('±')[0])
                    elif '-' in inhibition_percent:
                        inhibition_percent = sum(float(comp) for comp in inhibition_percent.split('-')) / len(inhibition_percent.split('-'))
                    else:
                        inhibition_percent = float(inhibition_percent)
                    if inhibition_percent < 95:
                        continue
                    else:
                        if ' ' in concentration.strip():
                            concentration = concentration.strip()
                            concentration = float(concentration.split(' ')[0] + concentration.split(' ')[-1])
                        elif '>=' in concentration:
                            concentration = float(concentration.split('>=')[-1]) * 2
                        elif '>' in concentration:
                            concentration = float(concentration.split('>')[-1]) + 156
                        elif '-' in concentration:
                            concentration = sum(float(comp) for comp in concentration.split('-')) / len(
                                concentration.split('-'))
                        elif '±' in concentration:
                            concentration = float(concentration.split('±')[0])
                        elif '<=' in concentration:
                            concentration = float(concentration.split('<=')[-1])
                        elif '<' in concentration:
                            if concentration[-1] == '<':
                                concentration = float(concentration.split('<')[0])
                            else:
                                concentration = float(concentration.split('<')[-1])
                        else:
                            concentration = float(concentration)

                        # if unit_name == 'µM':
                        #     bact_measure_value_unit[bact_name] = {('MIC', 'µM'): concentration}
                        # elif unit_name == 'µg/ml':
                        if unit_name == 'µg/ml':
                            mol = Chem.MolFromSmiles(DBAASPid_SMILES[line_index, 1])
                            mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
                            concentration = float(concentration) / mol_wt * 1000
                            # bact_measure_value_unit[bact_name] = {('MIC', 'µM'): uM_concentration}

                        if inhibition_percent > inhibition_percent_max:
                            inhibition_percent_max = inhibition_percent
                            MIC_final = concentration

                if inhibition_percent_max >= 95:
                    bact_measure_value_unit[bact_name] = {('MIC', 'µM'): MIC_final}
                else:
                    del bact_measure_value_unit[bact_name]
                    # if MIC_um_found:
                    #     for (value_name, unit_name), concentration in list(value.items()):
                    #         if value_name != 'MIC' and unit_name != 'micromole/liter':
                    #             del value[(value_name, unit_name)]
                    # else:
                    #     pass


