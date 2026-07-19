这个文件夹下面的代码用于在 Code Ocean 上给审稿人提交可复现的代码，Code Ocean里面可用的GPU资源只有一个15360MiB显存的Tesla T4 GPU  
要复现的结果在论文 capsule/ApexOracle_Nat_Biotech.pdf 中  
所有复现代码都只保留inference部分不做训练，所有需要的weight、data都要放到 capsule 中合适的地方方便我一起上传  

## 已完成：non-seen-strain MDLM-MTR 7-ensemble 复现

目标原始训练脚本已由当前仓库的统一 hierarchical runner 替代；旧文件只从
`legacy-code-snapshot-2026-07-17` tag 追溯。

判断依据：

- 这个脚本里 `num_ensembles = 7`
- 对应完整 checkpoint 目录是：
  - `Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles`
- 原目录里有 3 个 fold/group，每组 7 个 ensemble，共 21 个 `.pth`
- 原始 log 都显示跑到 `Ensemble 7/7 Epoch 25/25`

### Capsule 中新增的代码

- `code/reproduce_non_seen_strains_mdlm_mtr_fix.py`
  - Code Ocean 上实际运行的 inference-only 复现脚本
  - 只加载已有 checkpoint 做测试集预测，不训练
  - 会从 `data/` 里的原始资源重建原始 split
  - 会输出：
    - `results/non_seen_strains_mdlm_mtr_fix_metrics.json`
    - `results/non_seen_strains_mdlm_mtr_fix_predictions.csv`
  - 会记录 CUDA 峰值显存：
    - `_cuda_peak_memory.allocated_mib`
    - `_cuda_peak_memory.reserved_mib`
  - 有 `tqdm` 进度条：
    - fold 进度
    - ensemble 进度
    - genome+text loader batch 进度
    - text-only loader batch 进度
  - 默认参数：
    - `--device cuda:0`
    - `--batch-size 128`
    - `--num-ensembles 7`
  - `--batch-size 128` 是按 T4 15GB 显存短测后选择的；短测中 reserved peak 约 11.6GB。若 Code Ocean OOM，可降到 `96` 或 `64`。
  - `--max-batches N` 可用于 smoke test，只跑每个 loader 的前 N 个 batch。

- `code/run`
  - Code Ocean run entrypoint
  - 默认执行：
    - `python code/reproduce_non_seen_strains_mdlm_mtr_fix.py --data-root data --results-dir results`

- `README.md`
  - 面向 capsule 用户的简要运行说明

### 数据和权重如何迁移

旧资源准备脚本已在 unified runner 验证后删除；现有 capsule inference 与已准备资源仍可用，
旧打包逻辑从 legacy tag 恢复。

迁移到 `capsule/data` 的全量原始资源：

- `DataPrepare/Data/Genome_embs`
  - 原始 Evo2 genome embeddings
  - 约 3.3GB
- `DataPrepare/Data/Text_Description/ATCC/embeddings`
  - 有 genome 对应的 ATCC text embeddings
  - 约 2.0GB
- `DataPrepare/Data/Text_Description/wo_ATCC/embeddings`
  - 只有 text、没有 genome 的 strain embeddings
  - 约 3.9GB
- `DataPrepare/Data/Genome/ATCC`
  - 用于从 ATCC fasta 文件名重建 strain/species map
- `DataPrepare/Data/Pep_emb_dict.pt`
  - peptide molecule feature embedding cache
- `DataPrepare/Data/SM_emb_dict.pt`
  - small molecule feature embedding cache
- `DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv`
  - MIC regression 原始数据
- `DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data_SELFIES.csv`
  - small molecule binary auxiliary data
- `DataPrepare/Data/Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json`
  - 原始 strain name 到 standard ATCC/custom ID 的映射
- `DataPrepare/Data/Genome/old_to_new_NCBI_taxonomy.json`
  - NCBI taxonomy 新旧命名映射
- 原始训练脚本不再在 capsule 中保留第二份副本，统一从 legacy tag 追溯

checkpoint 迁移：

- 原始目录：
  - `Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles`
- capsule 目录：
  - `capsule/data/Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles`
- 迁移时对 21 个 `.pth` 去掉了：
  - `optimizer_state_dict`
- 每个瘦身 checkpoint 保留 keys：
  - `R2`
  - `re_head_state_dict`
  - `cls_head_state_dict`
  - `co_cross_attn_genome`
  - `co_cross_attn_text`
  - `learnable_embedding_weight`
- 原始单个 checkpoint 约 9.08GB；去 optimizer 后单个约 3.03GB。
- 21 个瘦身 checkpoint 总计约 60GB。

资源准备后写了 manifest：

- `data/non_seen_strains_mdlm_mtr_fix_resource_manifest.json`

### 当前验证记录

已验证：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/reproduce_non_seen_strains_mdlm_mtr_fix.py \
  --data-root capsule/data \
  --results-dir capsule/results/smoke_non_seen_strains \
  --device cuda:0 \
  --fold 0 \
  --num-ensembles 1 \
  --batch-size 8
```

这次完整跑完 fold 0 的 1 个 ensemble，测试样本数 71419，输出：

- R2: 0.3257729239
- spearman: 0.5928156183
- pearson: 0.6035505637

也验证了进度条和 `--max-batches` smoke：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/reproduce_non_seen_strains_mdlm_mtr_fix.py \
  --data-root capsule/data \
  --results-dir capsule/results/progress_smoke \
  --device cuda:0 \
  --fold 0 \
  --num-ensembles 1 \
  --batch-size 32 \
  --max-batches 1
```

### 正式运行命令

在 capsule 根目录下：

```bash
code/run
```

只跑单个 fold：

```bash
code/run --fold 0
```

快速 smoke test：

```bash
code/run --fold 0 --num-ensembles 1 --max-batches 2
```

如果 T4 OOM：

```bash
code/run --batch-size 96
```

或：

```bash
code/run --batch-size 64
```

### 可复用代码/模式

后续复现其它结果时可以复用以下现存部分：

- `code/reproduce_non_seen_strains_mdlm_mtr_fix.py`
  - `RegressionHead`
  - `FirstTokenAttentionGenome`
  - `calculate_r2`
  - `compute_metrics`
  - `move_batch_to_device`
  - `build_model`
  - `predict_loader`
  - CUDA peak memory 记录逻辑
  - `--max-batches` smoke-test 模式
  - `tqdm` fold/ensemble/loader 进度条模式

如果其它结果也用相同 architecture，只是 split、checkpoint 目录或数据源不同，优先复制这个 inference script 后只改：

- checkpoint directory
- split construction
- molecule embedding file names
- output file prefix
- expected metric aggregation方式

### 注意事项

- 复现代码不要重新训练；只做 inference。
- capsule 上传前确认 `capsule/data` 里已经包含所有需要资源。
- T4 只有 15360MiB 显存，默认 batch size 128 是基于短测选择；全量跑完后以 `results/*metrics.json` 里的 `_cuda_peak_memory.reserved_mib` 为最终显存依据。
- 当前 capsule 目录较大，主要来自 21 个去 optimizer checkpoint。

## 已新增：Fig. 1b zero-shot antibiotic classification 复现

目标原始训练脚本：

- `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py`

判断依据：

- 这个脚本对应 `group_names = ['#004', '17978', 'Staphylococcus aureus RN4220']`
- KFold target-strain small-molecule train/test split 被注释掉，直接用整个 target strain small-molecule set 做 test
- 对应完整 checkpoint 目录是：
  - `Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles`
- 原目录里有 3 个 group，每组 10 个 ensemble，共 30 个 `.pth`

### Capsule 中新增的代码

- `code/reproduce_zero_shot_antibiotic_classification.py`
  - Code Ocean 上实际运行的 inference-only 复现脚本
  - 只加载已有 checkpoint 做 zero-shot target strain 预测，不训练
  - 会输出：
    - `results/zero_shot_antibiotic_classification_metrics.json`
    - `results/zero_shot_antibiotic_classification_predictions.csv`
  - 默认参数：
    - `--device cuda:0`
    - `--batch-size 64`
    - `--num-ensembles 10`
  - `--group {0,1,2}` 可只跑单个 zero-shot target strain
  - `--max-batches N` 可用于 smoke test

- `code/prepare_zero_shot_antibiotic_classification_resources.py`
  - 本地准备 capsule 资源用，不是 Code Ocean 正式运行入口
  - 负责把 zero-shot 推理需要的数据/embedding/checkpoint 迁移到 `capsule/data`
  - 对 checkpoint 去掉 optimizer state，减少上传体积

- `code/run`
  - 默认仍执行 non-seen-strain 复现
  - zero-shot 复现入口：
    - `code/run zero-shot`

### 数据和权重如何迁移

资源准备命令：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/prepare_zero_shot_antibiotic_classification_resources.py \
  --repo-root /data2/tianang/projects/Synergy \
  --capsule-data /data2/tianang/projects/Synergy/capsule/data
```

checkpoint 迁移：

- 原始目录：
  - `Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles`
- capsule 目录：
  - `capsule/data/Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles`
- 迁移时对 30 个 `.pth` 去掉：
  - `optimizer_state_dict`
- 瘦身 checkpoint 保留 keys：
  - `auroc`
  - `auprc`
  - `re_head_state_dict`
  - `cls_head_state_dict`
  - `co_cross_attn_genome`
  - `co_cross_attn_text`
  - `learnable_embedding_weight`

正式运行命令：

```bash
code/run zero-shot
```

快速 smoke test：

```bash
code/run zero-shot --group 0 --num-ensembles 1 --max-batches 2
```

### 当前验证记录

已完成资源准备：

- 30/30 个 zero-shot checkpoint 已写入 `capsule/data`
- 每个瘦身 checkpoint 大小为 `3028055087` bytes
- manifest:
  - `data/zero_shot_antibiotic_classification_resource_manifest.json`

已验证 smoke test：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/reproduce_zero_shot_antibiotic_classification.py \
  --data-root capsule/data \
  --results-dir capsule/results/smoke_zero_shot \
  --device cuda:0 \
  --group 0 \
  --num-ensembles 1 \
  --batch-size 8 \
  --max-batches 1
```

已验证 text-only 路径：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/reproduce_zero_shot_antibiotic_classification.py \
  --data-root capsule/data \
  --results-dir capsule/results/smoke_zero_shot_group2 \
  --device cuda:0 \
  --group 2 \
  --num-ensembles 1 \
  --batch-size 8 \
  --max-batches 1
```

已验证 group 0 单 ensemble 全量推理：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/reproduce_zero_shot_antibiotic_classification.py \
  --data-root capsule/data \
  --results-dir capsule/results/verify_zero_shot_group0_e0 \
  --device cuda:0 \
  --group 0 \
  --num-ensembles 1 \
  --batch-size 64
```

输出：

- AUROC: `0.9350169300225734`
- AUPRC: `0.6120648595009186`
- CUDA reserved peak: `6688.0 MiB`

## 已新增：Fig. 2b MIC regression eval-only 复现

目标原始训练脚本：

- `/data2/tianang/projects/mdlm/DBAASP_MLM_MDLM.py`
- `fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py`
- `fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`
- `fix_ChemBERTa_MLM_mean_emb_on_DBAASP_SMILES_5_fold_mean_MIC.py`
- `fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py`
- `fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`
- `compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py`

判断依据：

- Fig. 2b 中 MTR+DLM 版本对应 DLM backbone `Checkpoints_fangping/best.ckpt`
- 原脚本使用 `DataPrepare/Data/DBAASP_id_SELFIES_bact_MICs.csv`
- 下游是 5-fold MIC regression head，`KFold(n_splits=5, shuffle=True, random_state=42)`
- baseline 脚本使用 `DataPrepare/Data/DBAASP_id_SMILES_bact_MICs.csv`，APEX 使用 `DataPrepare/Data/DBAASP_id_same_as_SMILES_AAseqs_bact_MICs_512_limit.csv`
- 本地已按同样 split、head、loss 和超参数抽取 eval-mode feature cache，并训练出每个模型 5 个 fold 的 best head checkpoint

### Capsule 中新增的代码

- `code/reproduce_fig2b_mic_regression.py`
  - Code Ocean 上实际运行的 inference-only 复现脚本
  - 只加载 capsule 中缓存好的 frozen features 和 regression head checkpoint
  - 不在 capsule 里运行 DLM/ChemBERTa/MoLFormer/PeptideCLM/APEX backbone
  - 不在 capsule 里训练 regression head
  - 会输出：
    - `results/fig2b_mdlm_dlm_mtr_metrics.json`
    - `results/fig2b_chemberta_mtr_metrics.json`
    - `results/fig2b_molformer_metrics.json`
    - `results/fig2b_apex_metrics.json`
    - `results/fig2b_peptideclm_metrics.json`
    - `results/fig2b_chemberta_mlm_mean_metrics.json`
    - `results/fig2b_chemberta_mlm_metrics.json`
    - `results/fig2b_mic_regression_summary.json`
  - 可选 `--write-predictions` 输出逐任务预测 CSV
  - 默认参数：
    - `--models mdlm_dlm_mtr chemberta_mtr molformer apex peptideclm chemberta_mlm_mean chemberta_mlm`
    - `--device cuda:0`，如果没有 CUDA 则自动是 CPU
    - `--batch-size 4096`

- `code/prepare_fig2b_mic_regression_resources.py`
  - 本地准备 capsule 资源用，不是 Code Ocean 正式运行入口
  - 负责把本地已生成的 feature cache、5 个 best head checkpoint、训练记录和 provenance 脚本复制到 `capsule/data`
  - 会验证 head checkpoint 中没有 `optimizer_state_dict`

- `code/run`
  - Fig. 2b 复现入口：
    - `code/run fig2b`

### 数据和权重如何迁移

资源准备命令：

```bash
/home/tianang/anaconda3/bin/conda run -n base python capsule/code/prepare_fig2b_mic_regression_resources.py \
  --repo-root /data2/tianang/projects/Synergy \
  --mdlm-root /data2/tianang/projects/mdlm \
  --capsule-data /data2/tianang/projects/Synergy/capsule/data
```

迁移到 `capsule/data` 的资源：

- `fig2b_mic_regression/<model>/features.pt`
  - 内容：对应 backbone 的 eval-mode frozen features、转换后的 19-task MIC labels、label mask、DBAASP ids、target columns
- `fig2b_mic_regression/<model>/fold_1/best_head.pt` 到 `fold_5/best_head.pt`
  - 每个 checkpoint 只包含 regression head state 和 eval 所需 metadata
  - 不包含 optimizer state
- `fig2b_mic_regression/<model>/metrics.json`
  - 本地 head 训练记录
- `fig2b_mic_regression/mdlm_dlm_mtr/embedding_cache_manifest.json`
  - feature cache 生成记录
- `source/reproduce_fig2b_mdlm_cached_5fold.py`
  - 本地生成 feature cache 和 head checkpoint 的脚本备份
- `source/reproduce_fig2b_baselines_cached_5fold.py`
  - 本地生成 baseline feature cache 和 head checkpoint 的脚本备份
- `source/DBAASP_MLM_MDLM.py`
  - 原始 DLM Fig. 2b 脚本备份

正式运行命令：

```bash
code/run fig2b
```

如果只想在 CPU 上验证：

```bash
code/run fig2b --device cpu
```

### 当前验证记录

本地生成 Fig. 2b DLM/MTR eval resources 的完整命令：

```bash
/home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm python scripts/reproduce_fig2b_mdlm_cached_5fold.py \
  --mode all \
  --gpus 0,1,2,3 \
  --extract-batch-size 200 \
  --batch-size 200 \
  --num-epochs 200 \
  --output-dir /data2/tianang/projects/Synergy/Checkpoints/fig2b_mdlm_cached_5fold \
  --force-cache \
  --log-every 10
```

生成的 DLM/MTR cached-feature eval 指标：

- mean R2 across 5 folds: `0.5207119213907342`
- fold 1: `0.5057967603206635`
- fold 2: `0.4968358513556029`
- fold 3: `0.5243468284606934`
- fold 4: `0.5479422556726556`
- fold 5: `0.5286379111440558`

本地生成 Fig. 2b baseline eval resources 的完整命令：

```bash
/home/tianang/anaconda3/bin/conda run --no-capture-output -n base python scripts/reproduce_fig2b_baselines_cached_5fold.py \
  --models all \
  --gpus 0,1,2,3 \
  --extract-batch-size 200 \
  --batch-size 200 \
  --num-epochs 200 \
  --output-dir /data2/tianang/projects/Synergy/Checkpoints/fig2b_baselines_cached_5fold \
  --force-cache \
  --log-every 10
```

生成的 baseline cached-feature eval 指标：

- ChemBERTa-MTR: `0.4518439405842831`
- MoLFormer: `0.4348622033470555`
- APEX: `0.4396152301838523`
- PeptideCLM: `0.40303157599348777`
- ChemBERTa-MLM mean pooling: `0.32906477890516583`
- ChemBERTa-MLM first-token: `0.3284056710569482`

已验证 capsule eval-only 入口：

```bash
bash capsule/code/run fig2b --device cpu --results-dir capsule/results/verify_fig2b_all
```

输出：

- DLM/MTR: `0.5207119254689468`
- ChemBERTa-MTR: `0.4518439402705745`
- MoLFormer: `0.43486219707288243`
- APEX: `0.4396152279878917`
- PeptideCLM: `0.403031579444283`
- ChemBERTa-MLM mean pooling: `0.3290647845519216`
- ChemBERTa-MLM first-token: `0.32840566478277505`

注意：

- 这条 capsule 复现的是固定 eval-mode feature cache + trained head checkpoint 的结果，适合 reviewer 在 Code Ocean 上稳定复现。
- 原始 online DLM 脚本记录的 W&B R2 约 `0.5393850571`，差异来自原始脚本每个 batch 在线运行 frozen DLM backbone 时的 training-mode/dropout/dynamic padding 行为；capsule 为了 eval-only 和可复现，使用固定 eval-mode features。
- baseline 资源同样使用固定 eval-mode features + trained head checkpoint；head 训练超参数保持原脚本的 5-fold、200 epoch、batch 200、Adam lr `1e-4`。
