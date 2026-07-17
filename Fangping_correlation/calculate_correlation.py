import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle

count_format = '/data/fangping/bulleye/APEX_results/BE_{}_peptides_count_sum.pkl'
mic_format = '/data/fangping/bulleye/APEX_results/{}_average_mic.pkl'
# 假设有两组预测值

with open(count_format.format('log'), 'rb') as f:
    count = pickle.load(f)
with open(mic_format.format('log'), 'rb') as f:
    mic = pickle.load(f)

print(len(count), len(mic))
# 计算皮尔逊相关系数
pearson_corr, _ = pearsonr(count, mic)
print(f"皮尔逊相关系数: {pearson_corr:.4f}")

# 计算斯皮尔曼相关系数
spearman_corr, _ = spearmanr(count, mic)
print(f"斯皮尔曼相关系数: {spearman_corr:.4f}")