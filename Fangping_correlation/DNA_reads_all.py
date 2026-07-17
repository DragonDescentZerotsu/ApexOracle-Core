"""
获得 0 1 标签的 DNA 数据
"""
import numpy as np
import pickle
from tqdm import tqdm
import json
import pandas as pd

def get_DNA_reads_dict(category, data_path_format):
    category_DNA_reads_dict = {}
    # 在同一个种类的 3 个 repeat 上进行
    for complete_data_path in tqdm(
        [data_path_format.format(category, repeat_num) for repeat_num in range(1, 4)],
        desc=f' processing {category} files...'):

        print(f'processing {complete_data_path}')












        with open(complete_data_path, 'r', encoding='utf-8') as file:
            for line in iter(file.readline, ''):  # 迭代直到文件结束
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                contents = line.strip().split()
                if any(item in contents[0] for item in ['_', 'DNA']):
                    continue
                else:
                    DNA_seq = contents[0].strip()
                    if DNA_seq == 'TGCATTGTGTACCTGCCTGAT':
                        print(f'{DNA_seq}: count: {int(contents[2])}')
                    if category_DNA_reads_dict.get(DNA_seq, None) is None:
                        category_DNA_reads_dict[DNA_seq] = [int(contents[2])]
                    else:
                        category_DNA_reads_dict[DNA_seq].append(int(contents[2]))

    original_length = len(category_DNA_reads_dict)
    # 三个文件过完之后算均值 reads
    for DNA, reads_list in tqdm(list(category_DNA_reads_dict.items()), desc=' getting mean ...'):
        # 只有在 3 个 repeat 中都存在的 DNA_seq 才会被进一步处理
        if len(reads_list) == 3:
            category_DNA_reads_dict[DNA] = np.array(reads_list).mean()
        elif len(reads_list) < 3:
            del category_DNA_reads_dict[DNA]
        else:
            print(f'{DNA} count: {len(reads_list)}, please check')

    current_length = len(category_DNA_reads_dict)
    print(f' rate of repeating DNA: {current_length / original_length:.2f}')
    return category_DNA_reads_dict

# data_path_format = '/data/fangping/bulleye/Bullseye_UPenn_data/BE_{}_r{}_counts.DNA.txt David Orlando'  # workstation
data_path_format = '/data1/fangping/bulleye/Bullseye UPenn data/BE_{}_r{}_counts.DNA.txt David Orlando'  # node002

# save_path = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_cntrl_log_stat_merged_counts.DNA.csv'  # workstation
save_path = '/data1/tianang/bulleye_data/DNA_binary_classifi_label.csv'  # node002

categories = ['cntrl', 'log', 'stat']

all_DNA_reads_dict = {}

for category in categories:
    all_DNA_reads_dict[category] = get_DNA_reads_dict(category, data_path_format)

print(' Finding common peptides')
cntrl_log_common_key = set(all_DNA_reads_dict['cntrl'].keys()).intersection(set(all_DNA_reads_dict['log'].keys()))
cntrl_stat_common_key = set(all_DNA_reads_dict['cntrl'].keys()).intersection(set(all_DNA_reads_dict['stat'].keys()))

print(' Finding different peptides')
cntrl_log_diff_key = set(all_DNA_reads_dict['cntrl'].keys()).difference(set(all_DNA_reads_dict['log'].keys()))
cntrl_stat_diff_key = set(all_DNA_reads_dict['cntrl'].keys()).difference(set(all_DNA_reads_dict['stat'].keys()))

positive_DNA = cntrl_stat_diff_key & cntrl_log_diff_key
negative_DNA = cntrl_stat_common_key & cntrl_log_common_key

binary_classify_data = []  # [peptide, 0/1]
for peptide in tqdm(positive_DNA, desc=' Storing positive data...', total=len(positive_DNA)):
    binary_classify_data.append([peptide, 1])

for peptide in tqdm(negative_DNA, desc=' Storing negative data...', total=len(negative_DNA)):
    binary_classify_data.append([peptide, 0])

print(' Saving binary label data...')
df = pd.DataFrame(binary_classify_data, columns=['peptide', 'label'])
df.to_csv('/data1/tianang/bulleye_data/binary_classifi_label.csv', index=False)
print(' Saved')


