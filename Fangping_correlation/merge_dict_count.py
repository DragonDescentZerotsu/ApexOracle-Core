import pickle
from tqdm import tqdm
import os
import json

dict_file_path_format = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_{}_r{}_counts.DNA.pkl'
output_file_path_format = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_{}_merged_union_counts.DNA.json'

categories = ['cntrl', 'log', 'stat']

replicate_num = 3

for category in categories:
    dicts = []
    merged_dict = {}
    if os.path.exists(output_file_path_format.format(category)):
        print(f'{output_file_path_format.format(category).split("/")[-1]} already exists, skipping...')
        continue
    for i in range(replicate_num):
        complete_data_path = dict_file_path_format.format(category, i+1)
        with open(complete_data_path, 'rb') as file:
            print(f'loading {complete_data_path.split("/")[-1]}...')
            dicts.append(pickle.load(file))
            print(f'loaded {complete_data_path.split("/")[-1]}')

    # 找三个 dict.keys() 的交集
    # common_keys = set(dicts[0].keys()).intersection(*(d.keys() for d in dicts[1:]))
    # 找三个 dict.keys() 的并集
    common_keys = set(dicts[0].keys()).union(*(d.keys() for d in dicts[1:]))
    for key in tqdm(common_keys, desc=f"Merging {category} Counts"):
        merged_dict[key] = {
            'Count': [],
            'DNA': []
        }
        # 合并 Count 和 DNA 列表
        for d in dicts:
            if d.get(key, None) is not None:
                merged_dict[key]['Count'].extend(d[key]['Count'])
                merged_dict[key]['DNA'].extend(d[key]['DNA'])

    # 保存结果到一个新的 json 文件
    output_file = output_file_path_format.format(category)
    with open(output_file, 'w') as file:
        print(f'saving {output_file.split("/")[-1]}...')
        json.dump(merged_dict, file, indent=4)

    print(f"Merged dictionary saved to {output_file}")