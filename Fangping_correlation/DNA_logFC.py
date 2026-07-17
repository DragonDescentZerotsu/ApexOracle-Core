"""
直接生成所有的 r1~3 的 union DNA -> reads count 字典
处理好的 DNA: reads count 字典 保存在
"""

import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from triton.language import dtype


data_path_formats = \
    ['/data/fangping/bulleye/Bullseye_UPenn_data/BE_{}_r1_counts.DNA.txt David Orlando',
     '/data/fangping/bulleye/Bullseye_UPenn_data/BE_{}_r2_counts.DNA.txt David Orlando',
     '/data/fangping/bulleye/Bullseye_UPenn_data/BE_{}_r3_counts.DNA.txt David Orlando']

# save_path_formats = \
#     ['/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_r3_DNA_reads_dict.pkl',
#      '/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_r2_DNA_reads_dict.pkl',
#      '/data/fangping/bulleye/Bullseye UPenn dict data/BE_{}_r1_DNA_reads_dict.pkl']

save_path_format = '/data/fangping/bulleye/Bullseye_UPenn_dict_data/BE_{}_merged_DNA_reads_dict.pkl'

cntrl_reads_threshold = 10

categories = ['cntrl', 'log', 'stat']
DNA_reads_dicts_across_categories = []
for category in categories:
    # 如果这个文件已经处理好就直接跳过了
    if Path(save_path_format.format(category)).exists():
        continue

    # 记录整个 cntrl or log or stat 的 DNA：count 字典
    content_dict = {}
    for data_path in data_path_formats:

        complete_data_path = data_path.format(category)
        pbar = tqdm(desc=f' processing {complete_data_path.split('/')[-1]}', leave=True)
        # content_dict = {}
        with open(complete_data_path, 'r', encoding='utf-8') as file:
            for line in iter(file.readline, ''):  # 迭代直到文件结束

                # TODO: 开头是 # 的都是不需要的数据集描述, 应该就这一段比较能复用因为都要每个 replicate 重新处理 成
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                contents = line.strip().split()
                # 去掉那些 peptide 有问题的
                if any(item in contents[1] for item in ['*', '_', 'B', 'J', 'O', 'U', 'X', 'Z', 'b', 'j', 'o', 'u', 'x', 'z', 'Peptide']):
                    continue
                else:
                    pbar.update(1)
                    DNA_seq = contents[0].strip()
                    # if DNA_seq == 'TGCATGATGTACCAGCTGTGG':
                    #     print(f'{DNA_seq}: count: {int(contents[2])}')
                    if content_dict.get(DNA_seq, None) is None:
                        content_dict[DNA_seq] = int(contents[2])
                    else:
                        # print(f'{DNA_seq}')
                        content_dict[DNA_seq] += int(contents[2])
        pbar.close()
    DNA_reads_dicts_across_categories.append(content_dict)
    print(DNA_reads_dicts_across_categories[0]['TGCATGATGTACCAGCTGTGG'])

    # 把所有 replicate 的求均值
    for DNA, reads in content_dict.items():
        content_dict[DNA] = reads / 3

    # 保存字典
    complete_save_path = save_path_format.format(category)
    with open(complete_save_path, 'wb') as file:
        pickle.dump(content_dict, file)
        print(f' completed {complete_save_path}')

# 如果没有现场计算那就直接加载存好的
if len(DNA_reads_dicts_across_categories) == 0:
    for category in categories:
        full_path = save_path_format.format(category)
        with open(full_path, 'rb') as file:
            print(f' loading {full_path}')
            DNA_reads_dicts_across_categories.append(pickle.load(file))

print(f' median: {np.median(np.array(list(DNA_reads_dicts_across_categories[0].values())))}')

# 计算 logFC 和那些消失掉的 peptide 作为高杀菌类
cntrl_DNA_reads_dict = {}
# 过滤那些 cntrl 里面reads很少的，必须要 cntrl_reads_threshold 以上才行
for DNA, reads in tqdm(DNA_reads_dicts_across_categories[0].items(), desc=f' Filtering cntrl DNA for DNA reads mean higher than {cntrl_reads_threshold}'):
    if reads >= cntrl_reads_threshold:
        cntrl_DNA_reads_dict[DNA] = reads

print(f' len of cntrl DNA with reads >= 10: {len(cntrl_DNA_reads_dict)}')

# for DNA, reads in tqdm(DNA_reads_dicts_across_categories[0].items(), desc=f' Filtering mean reads for non-numeric error'):
#     if not isinstance(reads, (float, int)):
#         print(f' wrong {DNA}: {reads}')
#     if reads is None:
#         print(f' wrong None {DNA}: {reads}')

# num_bins = 500
# counts, bin_edges = np.histogram(np.array(list(DNA_reads_dicts_across_categories[0].values()), dtype=float), bins=num_bins)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# plt.barh(bin_centers, counts, height=(bin_edges[1] - bin_edges[0]))
#
# plt.xscale('log', base=10)
# # 添加标签
# plt.xlabel('Number of DNAs')
# plt.ylabel('reads')
# plt.title(f'reads, {num_bins} bins')
#
# # 显示图表
# plt.savefig('/home/tianang/Projects/Synergy/Fangping_correlation/reads_histo.png', bbox_inches='tight')


cntrl_qualify_DNA_set = set(cntrl_DNA_reads_dict.keys())
positive_DNA_set = cntrl_qualify_DNA_set - set(DNA_reads_dicts_across_categories[-1].keys())
log_FC_DNA_set = cntrl_qualify_DNA_set - positive_DNA_set

label_dict = {}
log_FCs = []
for DNA in positive_DNA_set:
    label_dict[DNA] = 1

print(f' len of 1 positive DNA: {len(label_dict)}')

for DNA in log_FC_DNA_set:
    log_FCs.append(np.log10(DNA_reads_dicts_across_categories[-1][DNA] / cntrl_DNA_reads_dict[DNA]))

num_bins = 30
counts, bin_edges = np.histogram(np.array(log_FCs), bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.barh(bin_centers, counts, height=(bin_edges[1] - bin_edges[0]))

plt.xscale('log', base=10)
# 添加标签
plt.xlabel('Number of DNAs')
plt.ylabel('logFC')
plt.title(f'logFC Cntrl vs Stat, {num_bins} bins, Cntrl min DNA reads limit: {cntrl_reads_threshold}')

# 显示图表
plt.savefig('/home/tianang/Projects/Synergy/Fangping_correlation/logFC_histo_DNA.png', bbox_inches='tight')
plt.show()