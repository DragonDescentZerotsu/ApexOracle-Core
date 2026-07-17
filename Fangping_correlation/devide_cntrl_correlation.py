import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle
import json

def read_json_file(file_path):
    """
    Reads and parses a JSON file.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

cntrl_count_dict_path = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_cntrl_merged_counts.DNA.json'
log_count_dict_path = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_log_merged_counts.DNA.json'
stat_count_dict_path = '/data/fangping/bulleye/Bullseye UPenn dict data/BE_stat_merged_counts.DNA.json'
log_mic_path = '/data/fangping/bulleye/APEX_results/log_average_mic_dict.pkl'
stat_mic_path = '/data/fangping/bulleye/APEX_results/stat_average_mic_dict.pkl'

print('loading cntrl count')
cntrl_count_dict = read_json_file(cntrl_count_dict_path)
print('loading log count')
log_count_dict = read_json_file(log_count_dict_path)
print('loading stat count')
stat_count_dict = read_json_file(stat_count_dict_path)
print('loading log mic')
with open(log_mic_path, 'rb') as f:
    log_mic = pickle.load(f)
print('loading stat mic')
with open(stat_mic_path, 'rb') as f:
    stat_mic = pickle.load(f)

cntrl_log_common_key = set(cntrl_count_dict.keys()).intersection(set(log_count_dict.keys()))
cntrl_stat_common_key = set(cntrl_count_dict.keys()).intersection(set(stat_count_dict.keys()))

print(len(cntrl_log_common_key)/len(cntrl_count_dict.keys()))
print(len(cntrl_stat_common_key)/len(cntrl_count_dict.keys()))

log_divide_by_cntrl_count = []
log_divide_by_cntrl_count_corresponding_mic = []
stat_divide_by_cntrl_count = []
stat_divide_by_cntrl_count_corresponding_mic = []

for key in cntrl_log_common_key:
    log_divide_by_cntrl_count.append(sum(log_count_dict[key]['Count'])/sum(cntrl_count_dict[key]['Count']))
    log_divide_by_cntrl_count_corresponding_mic.append(log_mic[key])

for key in cntrl_stat_common_key:
    stat_divide_by_cntrl_count.append(sum(stat_count_dict[key]['Count'])/sum(cntrl_count_dict[key]['Count']))
    stat_divide_by_cntrl_count_corresponding_mic.append(stat_mic[key])


# print(len(count), len(mic))
# 计算皮尔逊相关系数
pearson_corr, _ = pearsonr(log_divide_by_cntrl_count, log_divide_by_cntrl_count_corresponding_mic)
print(f"log/cntrl皮尔逊相关系数: {pearson_corr:.4f}")

# 计算斯皮尔曼相关系数
spearman_corr, _ = spearmanr(log_divide_by_cntrl_count, log_divide_by_cntrl_count_corresponding_mic)
print(f"log/cntrl斯皮尔曼相关系数: {spearman_corr:.4f}")

pearson_corr, _ = pearsonr(stat_divide_by_cntrl_count, stat_divide_by_cntrl_count_corresponding_mic)
print(f"stat/cntrl皮尔逊相关系数: {pearson_corr:.4f}")

# 计算斯皮尔曼相关系数
spearman_corr, _ = spearmanr(stat_divide_by_cntrl_count, stat_divide_by_cntrl_count_corresponding_mic)
print(f"stat/cntrl斯皮尔曼相关系数: {spearman_corr:.4f}")