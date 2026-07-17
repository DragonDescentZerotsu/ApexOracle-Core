import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle
from tqdm import tqdm
import pandas as pd
import json
import random

def read_json_file(file_path):
    """
    Reads and parses a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

# cntrl_count_dict_path = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_cntrl_merged_counts.DNA.json'  # workstation
# log_count_dict_path = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_log_merged_counts.DNA.json'  # workstation
# stat_count_dict_path = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_stat_merged_counts.DNA.json'  # workstation
cntrl_count_dict_path = '/data1/tianang/bulleye_data/BE_cntrl_merged_counts.DNA.json'  # node002
log_count_dict_path = '/data1/tianang/bulleye_data/BE_log_merged_counts.DNA.json'  # node002
stat_count_dict_path = '/data1/tianang/bulleye_data/BE_stat_merged_counts.DNA.json'  # node002
# log_mic_path = '/data/fangping/bulleye/APEX_results/log_average_mic_dict.pkl'
# stat_mic_path = '/data/fangping/bulleye/APEX_results/stat_average_mic_dict.pkl'

print('loading cntrl count')
cntrl_count_dict = read_json_file(cntrl_count_dict_path)
print('loading log count')
log_count_dict = read_json_file(log_count_dict_path)

print(' Finding common peptides')
cntrl_log_common_key = set(cntrl_count_dict.keys()).intersection(set(log_count_dict.keys()))
print(' Finding different peptides')
cntrl_log_diff_key = set(cntrl_count_dict.keys()).difference(set(log_count_dict.keys()))

del log_count_dict

print('loading stat count')
stat_count_dict = read_json_file(stat_count_dict_path)

# print('loading log mic')
# with open(log_mic_path, 'rb') as f:
#     log_mic = pickle.load(f)
# print('loading stat mic')
# with open(stat_mic_path, 'rb') as f:
#     stat_mic = pickle.load(f)
print(' Finding common peptides')
# cntrl_log_common_key = set(cntrl_count_dict.keys()).intersection(set(log_count_dict.keys()))
cntrl_stat_common_key = set(cntrl_count_dict.keys()).intersection(set(stat_count_dict.keys()))

# 获得在cntrl中，但是不在 log/stat 中的peptide
print(' Finding different peptides')
# cntrl_log_diff_key = set(cntrl_count_dict.keys()).difference(set(log_count_dict.keys()))
cntrl_stat_diff_key = set(cntrl_count_dict.keys()).difference(set(stat_count_dict.keys()))

positive_pep = cntrl_stat_diff_key & cntrl_log_diff_key
negative_pep = cntrl_stat_common_key & cntrl_log_common_key

print(f' num of positive peptides: {len(positive_pep)}\nnum of negative peptides: {len(negative_pep)}')

# output_file  = open('/data/fangping/bulleye/Bullseye_UPenn_dict_data/binary_classifi.fasta', 'w')

# binary_classify_data = []  # [peptide, 0/1]
rate_of_sample = 0.01
num_pos_sample = round(len(positive_pep) * rate_of_sample)
num_neg_sample = round(len(negative_pep) * rate_of_sample)
sampled_pos_pep = random.sample(list(positive_pep), num_pos_sample)
sampled_neg_pep = random.sample(list(negative_pep), num_pos_sample)

pos_lines = []
neg_lines = []

for peptide in tqdm(positive_pep, desc=' Storing positive data...'):
    pos_lines.append(f'>{peptide}\n{peptide}\n')
print('Writing positive data')
with open('/data1/tianang/bulleye_data/pos_pep.fasta', 'w') as output_file:
# binary_classify_data.append([peptide, 1])
    output_file.writelines(pos_lines)

for peptide in tqdm(negative_pep, desc=' Storing negative data...'):
    neg_lines.append(f'>{peptide}\n{peptide}\n')
print('Writing negative data')
with open('/data1/tianang/bulleye_data/neg_pep.fasta', 'w') as output_file:
# binary_classify_data.append([peptide, 0])
    output_file.writelines(neg_lines)
# output_file.close()

# print(' Saving binary label data...')
# df = pd.DataFrame(binary_classify_data, columns=['peptide', 'label'])
# df.to_csv('/data/fangping/bulleye/Bullseye_UPenn_dict_data/binary_classifi_label.csv', index=False)
# print(' Saved')

