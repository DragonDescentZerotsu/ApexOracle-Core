## AGENTS.md 维护语言

- 本项目的 `AGENTS.md` 应尽量使用中文记录和维护。
- 代码文件名、路径、命令、模型名称、指标缩写以及没有自然中文译名的专有名词可以保留英文。
- 新增审计结论时，应明确区分“已由代码和日志验证的事实”“根据现有证据作出的推断”和“仍待作者确认的事项”。
- 当前发布重构的阶段、已确认决策和验收标准记录在 `REFACTOR_PLAN.md`；执行过程中应同步更新其中的状态，避免计划与代码迁移脱节。

## 环境说明

- 默认使用 conda 的 `base` 环境。在机器 `sn4622119311` 上，conda 位于 `/home/tianang/anaconda3/bin/conda`。
- 不需要在沙箱中运行代码，因为沙箱可能阻止 GPU 访问。
- `/data2/tianang/projects/mdlm` 中的所有代码都必须在 `mdlm` conda 环境中运行。需要直接mdlm训练推理的代码可能都在那里面
- /data2/tianang/projects/discrete-diffusion-guidance 里面是所有我们使用的generate peptide用的代码和仓库
- /data2/tianang/projects/evo2 里面是我们使用的embedding genome的代码

## 论文及审稿回复路径

- 本项目对应论文：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`
- 审稿意见及回复草稿：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Response to reviewers letter.docx`

## 审稿回复辅助脚本

### `scripts/plot_evo2_genome_embedding_abs_mean_distribution.py`

- 功能：复现审稿回复中用于解释为什么采用固定 `1e14` genome embedding 缩放因子的 Evo-2 genome embedding `abs_mean` 分布。
- 默认行为：读取 `DataPrepare/Data/Genome_embs`，找出实际被 AMP MIC、小分子二分类和 synergy FICI 数据集匹配到的 genome ID，为每个 genome embedding 计算一个 `mean(abs(E))`，并输出 PNG、PDF 和 CSV。
- 默认命令：

  ```bash
  python scripts/plot_evo2_genome_embedding_abs_mean_distribution.py
  ```

- 默认输出：
  - `paper_figs/evo2_genome_embedding_abs_mean_distribution.png`
  - `paper_figs/evo2_genome_embedding_abs_mean_distribution.pdf`
  - `paper_figs/evo2_genome_embedding_abs_mean_distribution.csv`
- 使用 `--all-embeddings` 可以统计 `Genome_embs` 中的全部文件，而不是只统计被数据集匹配到的 genome ID。

## 代码库审计：论文、代码与数据血缘

本节记录了 2026-07-16 在代码库清理前完成的静态审计。审计覆盖当前代码库中的全部 Python、shell、notebook、capsule、checkpoint 日志以及相关数据和配置文件，并与 `sn-article.tex` 进行了对照。本次审计没有重新运行任何实验。

### 如何理解本审计中的结论

- `最终版 / 高置信度`：脚本配置、checkpoint 目录、完整日志、论文实验协议或论文报告数值彼此一致。
- `可能的最终版 / 中等置信度`：脚本是当前最接近论文协议的实现，但现存文件无法完整恢复论文中的部分数值、fold 或方法细节。
- `历史版本`：已经被同一代码家族中的后续版本取代的旧副本或原型。
- `论文后 / 审稿阶段`：论文主体实验之后增加的分析、前瞻性数据实验或审稿复现代码；仍有价值，但不是原始论文实验代码。
- 文件修改时间只作为弱证据。更强的判断依据依次包括：输入数据路径、数据划分方式、模型或 backbone、ensemble 数量、checkpoint 路径、日志是否完整结束，以及日志指标能否对应论文数值。
- 当前仓库没有可用的 Git 历史；`.git` 不是一个可工作的 Git 仓库。因此无法恢复已删除版本或精确 commit 来源。不能仅根据文件名中的 `fix`、`new` 或 `old` 判断终版。
- 大量脚本是整段复制后进行少量修改的单文件程序。清理时应先保留最终行为，再把公共的数据集处理、划分逻辑、融合模块、预测头、指标计算和 checkpoint 加载逻辑抽取为共享模块。

### 论文实验与代码的核心对应关系

| 论文内容 | 最匹配的代码或资源 | 判断及证据 |
| --- | --- | --- |
| Fig. 1a strain-wise 泛化；Fig. 2c 最终 DLM 和 7 模型 ensemble | `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py`；checkpoint：`Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles` | **最终版 / 高置信度。** 三个完整 group 的 ensemble R2 分别为 0.4057、0.6889、0.6434，平均值恰好为论文中的 0.5793；每组均有 7 个模型。 |
| Fig. 1a / Fig. 2f phylum-wise holdout | `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble.py`；checkpoint 位于 `.../3_species_w_SM/7_fold_ensembles` | **可能的最终版 / 中等置信度。** 实现了论文中的三个 division；文件名虽然写着 `5_ensemble`，实际使用 7 个模型。现存 R2 为 0.2194、0.3612、0.3367，平均 0.3058，与论文 0.3744 不一致。后续 MDLM checkpoint 目录中只有 Fungi 结果 0.2920，因此论文最终绘图所用完整运行可能已经缺失。 |
| Fig. 1a / Fig. 2d、g species-wise 11-cluster holdout | `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble_MDLM_cls_fix.py`；checkpoint 位于 `.../11_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles` | **可能的最终版 / 中等置信度。** 这是最接近论文的 DLM/token cache 版本，但现存完整日志只有 group 6–10。论文中 11 个 cluster 平均 R2 0.3809、去掉 Mycoplasmatota cluster 9 后为 0.4337 的完整运行没有保留下来。 |
| Fig. 2c strain-wise molecular encoder 比较 | 上述最终脚本，以及 `..._ChemBERTa_MLM.py`、`..._ChemBERTa_MTR.py`、`..._MolFormer.py`、`..._PeptideCLM.py` | DLM 终版为高置信度。其余脚本是对应的 strain-aware comparator，但部分 comparator checkpoint 或日志不完整。`*_cls_wo_padding*`、`*_mean_wo_padding*` 和 `*_eval.py` 是 pooling、cache 或 eval 实验，不是论文最终 7 模型 DLM 结果。 |
| Fig. 2c Evo-2 与 k-mer 消融 | 当前没有对应源代码；`Checkpoints/KMER_genome_text_learnable_emb` 下只有 2026 年的部分或失败日志 | **缺失。** 论文报告 R2 下降 11.6%，但当前 KMER 日志没有完整结束，仓库内也没有包含 k-mer 实现的 Python 文件。不能声称当前仓库能够复现该结果。 |
| Fig. 2b 不使用 strain knowledge 的五折 molecular representation benchmark | DLM 原始脚本位于外部 `/data2/tianang/projects/mdlm/DBAASP_MLM_MDLM.py`，capsule 的 `data/source` 中有副本。baseline 为 `fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`；APEX 为 `compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py` | 这是原论文实验代码家族。当前论文图中数值目测约为：DLM MTR+DLM 0.530、ChemBERTa MTR 0.417、DLM MLM 0.408、ChemBERTa MLM 0.226、PeptideCLM 0.376、MolFormer 0.371、APEX 0.403。审稿阶段建立的 cache 复现资源并不能精确对应当前图中的全部数值，只能视为派生复现产物。 |
| Fig. 1b 严格 target-strain zero-shot 小分子分类 | `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py`；checkpoint：`.../antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles` | **最终版 / 高置信度。** 脚本注释掉了目标 strain 内部的 KFold，并在完整 held-out target-strain 数据集上测试。完整 ensemble 指标：E. coli `#004` 为 0.9360 AUROC / 0.5890 AUPRC；A. baumannii 17978 为 0.7262 / 0.3243；S. aureus RN4220 为 0.7679 / 0.1655。 |
| Fig. 1b fine-tuned ApexOracle | `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad.py`；checkpoint：`MDLM_fix_cls_10_fold_ensembles` | **可能的最终版 / 证据不完整。** 该脚本在目标 strain 上进行 KFold fine-tuning，但保留下来的日志只完成了前几个 fold。`..._wo_SAND.py` 去掉了 strain-aware 融合，只在分子 DLM embedding 上训练分类头，因此是 molecule-only baseline 或消融，不是完整 fine-tuned ApexOracle。 |
| Synergy 二分类结果 | `synergy_Evo_train_new_reg_MDLM_one_base_model_classification.py`；checkpoint：`.../strain_wise_synergy/MDLM_3_fold_ensembles_1_base_model_cls` | **可能的最终版 / 中等置信度。** 使用 `synergistic_pairs_Evo.csv`、strain-wise 三折划分、单个完整 MIC base checkpoint、FICI 二分类和 7 个 ensemble。现存三个 fold 的 AUROC/AUPRC 分别为 0.6690/0.6159、0.7614/0.6853、0.8489/0.9307，未加权平均约为 0.7598/0.7440，与论文 0.7539/0.7454 接近但不完全相同。论文写 LoRA rank 64，而该 CV 脚本对 fusion 使用 1024、对 head 使用 256；rank 64 出现在后续 all-data noisy classifier 中，因此需要作者进一步确认精确版本。 |
| Synergy 或 guidance 之前使用的完整数据 MIC base model | `train_on_all_data.py`；checkpoint 家族 `Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask`，尤其是 `noise_guidance_best_R2_all_peptide_epoch_100.pth` | 这是最终 synergy 和 noisy guidance 脚本共同使用的最匹配 base model。`train_on_all_data.py` 本身保存到 `all_AMP_SM_data_train/MDLM_MTR_fix_cls_wo_pad`；路径和命名在开发过程中发生过变化，重构时必须保留实际 checkpoint 来源。 |
| Noisy synergy/peptide guidance classifier | `synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py` 和后续较干净的 `..._all_data_classification_clean.py` | **论文后实现支持。** `clean` 版本使用 rank-64 LoRA，并在 `.../synergy_judger/cls` 下保存 noisy synergy classifier。它们是 all-data guidance head，不是论文三折 synergy benchmark。`DataPrepare/MDLM/label_pep_nonpep.py` 用于准备 peptide/non-peptide 标签。 |
| Guided molecule generation、remasking sampler 和 256-step 三阶段 guidance | 不在当前仓库；分析脚本指向 `/data2/tianang/projects/discrete-diffusion-guidance` | **外部依赖 / 缺失。** 当前仓库包含 predictor、保存的输出或图片以及 similarity 分析，但不包含实际生成 ApexOracle-3/12/23 的 sampler。论文参数为 256 steps、MIC target 1、sigma 从 0.5 线性降到 0.2、`t_on=0.55`、`t_off=0.45`，两个阶段的 guidance strength 为 15。 |
| Fig. 3（TeX 中使用文件 `Fig4.pdf`）guided/unguided 预测和最终验证分子 | `DataPrepare/Morgan_fingerprint_sim_generation*.py` 中的生成结果分析、`synergy_Evo_train_on_DBAASP_screen_inhouse_pairs.py` 中的筛选逻辑，以及 `paper_figs/` 下的 PDF | 当前只剩部分派生分析。候选生成和最终选择流程依赖外部代码或已经不完整；湿实验结果也没有在本仓库形成计算复现流程。 |
| 附录 modality ablation | 较早的 genome-only、text-only、genome+text 脚本家族，以及 `Checkpoints/genome`、`Checkpoints/text`、`Checkpoints/genome_text` | **可能的来源家族 / 中低置信度。** 当前没有一个干净的入口或完整指标表可以把现存 checkpoint 精确连接到最终附录图。重建前应先核对原始绘图数据。 |
| Attention 或耐药基因解释 | `DataPrepare/ATCC_genome_annotation_get.py`、`DataPrepare/resistant_gene_check.py`、`DataPrepare/train_genome_mcr_check.py`，以及大型训练脚本中的 attention 输出 | 属于探索或审稿支持代码；不存在自包含的最终 attention figure 流程。 |
| ApexOracle-3/12/23 sequence similarity 表 | `DataPrepare/get_similarity/` | **最终版 / 高置信度。** 实现了当前 Methods 中的定义：Biopython global alignment、BLOSUM62、gap-open 10、gap-extension 0.5、exact-match PID，以及 cyclic peptide 的穷举旋转。 |
| 审稿回复中的 Evo-2 embedding 缩放说明 | `scripts/plot_evo2_genome_embedding_abs_mean_distribution.py` | **审稿阶段 / 高置信度。** 生成支持固定 `1e14` 缩放因子的 CSV、PNG 和 PDF，统计范围为 563 个实际匹配的 embedding。 |

### 主要训练脚本的版本家族

#### 分层 MIC 回归

- `MIC_with_genome.py`、`MIC_with_genome_no_AMP.py`、`MIC_with_genome_test_on_non_seen_species.py`、`MIC_with_genome_test_on_non_seen_species_3_species.py`、`MIC_with_genome_test_on_non_seen_species_5_fold.py`、`MIC_with_genome_test_on_non_seen_species_{3,11}_species_5_ensemble.py`、`MIC_with_genome_test_on_non_seen_strains.py` 和 `MIC_with_genome_test_on_non_seen_pep_&_non_seen_species.py` 是早期 genome-only/ChemBERTa 原型及划分实验，属于**历史版本**。
- `MIC_with_text_test_on_non_seen_{strains,species_3_species_5_ensemble,species_11_species_5_ensemble}.py` 是 text-only 消融；`MIC_with_text_genome_test_on_non_seen_*.py` 是早期双模态版本。它们属于 modality ablation 血缘，而不是最终 DLM 模型。
- `DP_inhouse_MIC_with_text_genome_test_on_non_seen_*.py` 加入了 in-house AMP 数据和 DataParallel 时代的结构。以 `_old.py` 结尾的文件已经明确被后续版本取代。
- `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*.py` 进一步加入 small-molecule binary auxiliary task，是论文时期的代码家族。
- 11-cluster 家族中应优先保留 `..._MDLM_cls_fix.py`，而不是非 MDLM 或 `_old.py` 副本，但必须同时记录只有五个最终日志仍然存在。
- strain-wise 最终训练和 inference 应优先使用 `..._MDLM_MTR_fix.py`。`..._MDLM_MTR_fix_cls_wo_padding.py` 和 `..._fix_mean_wo_padding.py` 用于比较 first-token、mean pooling 或预计算 feature；它们的 `_eval.py` 版本切换到 `*_eval.pt` cache 和 eval 行为。这些是一折实验，不是论文 ensemble。
- `..._ChemBERTa_MLM.py`、`..._ChemBERTa_MTR.py`、`..._MolFormer.py` 和 `..._PeptideCLM.py` 是 Fig. 2c 的 strain-wise encoder comparator。

#### Molecule-only Fig. 2b benchmark

- `fine_tune_on_DBAASP_SMILES_5_fold_mean_MIC.py` 是早期 ChemBERTa mean-MIC benchmark。`fine_tune_on_DBAASP_SMILES_5_fold_compare.py`、`_pre_SSL.py` 和 `_SSL.py` 是表示或预训练比较实验。`fine_tune_on_DBAASP_SMILES_all_data_for_training.py` 在全部 benchmark 数据上训练；`fine_tune_on_inhouse_SMILES_CV_MIC.py` 是 in-house-only 版本。相对于 `fix_*` 脚本，它们都属于**历史版本**。
- ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM 应优先使用 `fix_*_on_DBAASP_SMILES_5_fold_mean_MIC.py`。`fix_ChemBERTa_MLM_mean_emb_*` 是 mean-pooling 消融；当前论文图中的 ChemBERTa-MLM bar 使用 first-token 版本。
- DLM/MTR benchmark 源代码位于外部 `mdlm` 项目；capsule 中的副本只是审计快照，不能视为第二套 canonical 实现。
- **已由 checkpoint 和日志验证的事实：** `/data2/tianang/projects/mdlm/Checkpoints_fangping/best.ckpt` 是 24-layer、hidden size 1024、global step 800046 的 checkpoint，并含有 `backbone.regression.*` 四个 209-descriptor regression 参数，因此属于 MTR+DLM 家族；当前 `capsule_fig2` 将它用于 `mdlm_dlm_mtr` 是有模型结构依据的。
- **已由 checkpoint 验证的事实：** `best_1.ckpt`、`best_2.ckpt`、`best_3.ckpt`、`1-314000.ckpt`、`2-471000.ckpt` 和 `4-750000.ckpt` 均为 12-layer、hidden size 768 的纯 DLM checkpoint，不含 `backbone.regression.*`。DLM-only 的基础训练实现位于外部仓库的 `diffusion.py`、`models/dit.py` 和相关配置；`DBAASP_MLM_MDLM.py` 是其下游五折 MIC head 代码家族。
- **根据现有证据作出的推断：** `wandb/run-20250421_231424-s58d1559/files/output.log` 的五个 fold 最佳 mean R2 为 0.4132、0.3529、0.4400、0.4134、0.4222，均值约 0.4083，与论文 DLM MLM bar 对应。结合运行时间，最可能使用的是当时已存在的 `best_2.ckpt`；旧日志没有保存实际 checkpoint 路径，因此仍需用新 benchmark 在共享数据协议下核验，不能把 `best_2.ckpt` 记为已完全确认的论文终版。
- 新公平 benchmark 必须分别加载 12-layer DLM-only 和 24-layer MTR+DLM 配置，不能仅通过同一个 `best.ckpt` 生成两个不同标签的结果。
- **已由新代码和真实数据验证的事实：** `src/apexoracle/benchmarks/molecule_encoders/protocol.py` 已实现 reviewer 要求的共享样本和共享五折协议。源表 11,401 个 molecule 中有 11,398 个进入公共集合，3 个 DBAASP ID（20480、20527、20979）因原始 sequence 缺失而在划分前显式排除；fold 大小依次为 2,280、2,280、2,280、2,279、2,279。原始数据和生成的 CSV 不进入 Git。
- **已由新代码和真实数据验证的事实：** APEX 投影中有 1,689 条记录含 noncanonical residue、1,460 条含 D-residue、2,335 条含被线性化的 bond/multichain topology，92 条超过 50 residues 并被截断。DBAASP 的正确字段名是 `intrachainBonds`、`interchainBonds` 和 `coordinationBonds`；早先按 `intraChainBonds`/`interChainBonds` 检查得到的“字段为空”结论无效，后续不得沿用。
- **已实现的协议选择：** noncanonical residue 显式映射为 `X`；重构版 APEX adapter 为 `X` 分配独立 index 23，冻结 AAindex 向量使用 20 种 canonical residue 向量的均值。旧 `compare_APEX/utils.py` 会把未知字符静默留在 padding index 0，新 benchmark 不复用该行为。
- **已由代码审计验证的事实：** 旧 `scripts/reproduce_fig2b_baselines_online_5fold.py` 和 capsule 资源不满足新版公平协议：各 encoder 在自己的过滤结果上重新 KFold，训练时每个 epoch 都在 outer test fold 上评估并用它选择 best checkpoint；APEX 使用 `512→256` head，而其他 comparator 使用 `384→128` head。`scripts/reproduce_fig2b_apex_original_5fold.py` 还在 validation 时保持 regression head 为 train mode，使 dropout 参与 checkpoint 选择。因此旧 capsule 数值只能作为历史派生结果，不能作为 reviewer 要求的新正式结果。
- **已实现的新版协议：** outer 五折由唯一 `folds.csv` 冻结；每个 outer training fold 内再以 `seed=42+outer_fold` 固定划出 10% molecule 作 validation；所有 frozen encoder 在 eval mode 产生 feature；所有 comparator 使用相同 `384→128→19` head；checkpoint 只由 validation macro-task R2 选择，outer test 最后只评估一次。
- **已实现并由测试验证的基础设施：** `feature_cache.py` 定义不使用 pickle 的 `.npz` feature-cache 契约，并要求 cache 与 11,398 个公共 ID 完全相等后才按 canonical ID 顺序重排；`training.py` 是所有 encoder 共用的 head trainer；`scripts/reproduce/run_fig2b_shared_heads.py` 是统一入口。旧 capsule `.pt` cache 缺少这一协议版本和完整 ID 契约，不能直接冒充新版公平结果。
- **已由真实 checkpoint 和全量 feature cache 验证的事实：** 重构版 APEX adapter 从旧 checkpoint 中保留除 `peptideEmb.aa_embedding.weight` 外的全部参数，把 23-row AAindex embedding 扩展为含独立 `X` 的 24 rows；严格加载后只有该 embedding key 是预期 missing key。全部 11,398 条公共样本在 CPU 上成功产生并严格回读 `(11398, 128)` feature；本地 cache 位于被 Git 忽略的 `Checkpoints/fig2b_shared_v1/apex/features.npz`，约 5.8 MB，不应提交。
- **已由本地权重小样本 smoke test 验证的事实：** ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM 均可由 `encoders.py` 在 eval mode 产生 feature，维度依次为 384、384、768、768。全量 tokenizer 审计均保留 11,398 个 ID：前三者各截断 512 条且无 UNK；PeptideCLM 截断 24 条并有 8,150 条含 `[UNK]`。高 UNK 比例必须在最终公平结果中披露，不能对 PeptideCLM 单独删样本。
- **尚待全量验证的实现：** 当前 GPU driver 不可用，因此四个 Hugging Face comparator 尚未生成全量 feature cache；不得把 adapter smoke test 写成“已完成正式 benchmark”。
- `compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py` 是最终 APEX benchmark 版本。`APEX_train_DBAASP_MIC.py`、`APEX_train_DBAASP_MIC_5_fold_mean.py`、`APEX_train_inhouse_MIC.py`、`fine_tune_on_DBAASP_SMILES.py` 和 `deubg.py` 是早期或 debug driver。`APEX_models.py`、`APEX_trainer_CV.py` 和 `utils.py` 是复制过来的 APEX 支持代码。`APEX_all_data.sh` 是历史集群启动脚本，其中包含必须撤销和删除的明文 W&B 凭据。

#### 小分子抗生素分类

- `antibiotic_3_strain_compare.py` 是 DLM 之前的 ChemBERTa 版本。
- `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py` 是论文严格 zero-shot 实现。
- `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad.py` 是在目标 strain 上进行 KFold fine-tuning 的版本。
- `antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_wo_SAND.py` 注释掉 AMP/pathogen fusion 训练，只在 DLM embedding 上使用 molecule-only classification head。因此这里的 “wo_SAND” 表示没有 strain-aware 数据或融合的 baseline，而不是 strict zero-shot ApexOracle。
- `bash/3-strain-compare-CESGA.sh` 和 `bash/3_strain_compare.sh` 是包含机器特定环境和路径的旧 SLURM/手工启动脚本。

#### Synergy 与后续 in-house 实验

- `synergy_train.py`、`synergy_train_simple.py`、`synergy_train_no_pretrain.py` 和 `synergy_train_no_old_cls_emb.py` 是早期 ChemBERTa 或原型实验，属于**历史版本**。
- `synergy_Evo_train.py` 和 `synergy_Evo_train_new_reg.py` 引入 Evo/text mapping 和修改后的 regression。`synergy_Evo_train_new_reg_MDLM.py` 将 molecule feature 切换为 DLM。`..._one_base_model.py` 从一个 MIC base 初始化全部 fold，但仍然预测连续 FICI。论文最终任务是二分类，因此论文结果血缘应优先使用 `..._one_base_model_classification.py`。
- `..._one_base_model_all_data_train.py` 在组合 synergy 数据上训练而不进行 CV。`..._all_data_classification.py` 和 `_clean.py` 为 guidance 训练 noisy all-data classifier；`_clean.py` 是较新的可读版本。
- `synergy_Evo_train_on_DBAASP_test_on_inhouse.py` 和 `_classification.py` 用于测试向 prospective in-house pair 的迁移。`_few_shot.py`、`_classification_few_shot.py`、`_inner_prod.py`、`_no_lora.py` 和 `_w_pred_MIC.py` 是后续 few-shot 架构变体，属于**论文后代码**，不能与论文 DBAASP 三折指标混用。
- `synergy_Evo_train_on_DBAASP_screen_inhouse_pairs.py` 筛选 prospective in-house pair，目标 strain 硬编码为 BAA-3170。`synergy_Evo_test_inhouse_MDLM.py` 加载 all-data synergy 模型并测试处理后的 in-house 数据。这些服务于后续发现工作，不是论文核心 CV。

### 论文时期训练使用的最终数据

| 模态 | 论文来源和数量 | 最终本地训练数据 | 需要注意的区别 |
| --- | --- | --- | --- |
| AMP MIC | DBAASP 下载于 2024-09-27：16,408 个 peptide、5,630 个 strain、105,547 条 MIC；in-house：1,642 个 peptide、11 个 strain、15,718 条 MIC；论文合并后为 17,988 个 peptide、5,632 个 strain、121,265 条 MIC | 化学结构终版：`DataPrepare/Data/DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv`，121,265 行；token cache 终版：`DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv`，120,955 行 | 论文数量是 tokenizer、长度和 UNK 过滤之前的数量。训练脚本通常读取 120,955 行 token 文件，然后继续根据 genome/text embedding 可用性和 token 长度过滤。标签在单位和操作符处理后转换为 `-log10(MIC/10)`。 |
| 小分子抗生素二分类 | 共 49,331 个 molecule-strain pair：RN4220 39,312；BW25113 2,335；ATCC 17978 7,684 | `small_molecule/processed/small_molecule_Evo_binary_data.csv` 有 49,331 行；tokenized `..._SELFIES.csv` 有 49,330 行 | SELFIES token 过滤损失 1 行。内部标准 strain 名为 `Staphylococcus aureus RN4220`、`#004`（BW25113）和 `17978`。当前仓库没有把三篇论文原始数据完整合并到此文件的全流程脚本。 |
| Genome | 除 MG1665、UMNK88、PA14、USA300 来自 NCBI 外，其余来自 ATCC | FASTA 位于 `DataPrepare/Data/Genome`；最终 tensor 位于 `Genome_embs`；最终名称映射为 `Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json` | 当前训练和评估实际匹配到 563 个 embedding。仓库缺少 Evo-2 提取代码。论文方法为：11,000 nt window、10,000 nt step、Evo-2-40B 第 46 层、每个 window 按长度求均值、window 之间不 pooling，并使用固定 `1e14` 缩放。 |
| Strain text | Qwen2.5-Max，版本 `qwen-max-0125`，基于文献检索生成描述 | 文本和 tensor 位于 `Text_Description/ATCC` 与 `Text_Description/wo_ATCC` | 最终模型使用 Me-LLaMA3-8B 倒数第二层的 token embedding，不做 mean pooling；精确 strain 名会替换为 “this strain”。`Get_text_embedding_wo_genome.py` 是本地 canonical embedding 脚本。 |
| Synergy | 论文为 2,732 个通过筛选的 unique molecule-synergy-strain pair，其中 88% 为 AMP-small molecule、12% 为 AMP-AMP；FICI < 0.5 标为 1 | 原始扩充表 `synergistic_pairs_Evo.csv` 有 4,285 行；后续组合表 `synergy_DBAASP_inhouse_Evo.csv` 有 4,342 行 | 论文中的 2,732 是 molecule、strain 和 embedding 过滤后的 eligible/curated 子集，不是当前原始表行数。最终 CV 脚本动态过滤 `synergistic_pairs_Evo.csv`；当前没有保存不可变的 2,732 行快照，这是一个复现缺口。 |
| DLM 预训练 | PubChem、SmProt、UniRef 中长度不超过 50 aa 的序列、UniProt 中长度不超过 50 aa 的序列和 CycloPS；论文报告 121.6M 条有效去重序列及 209 个 RDKit descriptor | 主要位于外部 `/data2/tianang/projects/mdlm`；`DataPrepare/MDLM` 中只保留部分预处理 | 训练实现和 checkpoint 在外部项目中。不能声称仅使用 `Synergy` 仓库即可端到端训练 DLM。 |

Strain count mapping 的演化顺序如下：

- `Evo_edition_1_MIC_data_count_105547.json`：原始 DBAASP 的 5,630 个 strain 计数。
- `Evo_ATCC_only_edition_1_MIC_data_count_43366.json` 与 `Evo_no_ATCC_only_edition_1_MIC_data_count_62181.json`：ATCC-like 与 non-ATCC 分区。
- `Evo_edition_2_MIC_data_handcrafted.json`：手工修正 mapping。
- `Evo_edition_3_MIC_data_handcrafted_no_ATCC_to_custom_ATCC.json`：把 non-ATCC 名称映射到 custom 或标准 genome identity。
- `Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json`：加入 in-house strain 后的**最终 5,632-strain mapping**，最终训练脚本使用该文件。
- `Evo_synergy_edition_1_FICI_data_count_4591.json`：早期 synergy strain count mapping。`Evo_synergy_edition_2_FICI_data.json` 不是合法 JSON，至少包含一个多余的 `k`，不能作为已验证终版数据。

### 数据处理代码清单与血缘

#### DBAASP 下载、PepLink、MIC 转换与合并

- `DataPrepare/DownloadAllPep.py`：下载全部 DBAASP peptide record 到 `all_peptides_data.json`；依赖硬编码的旧 API 和路径假设。
- `DownloadOnePep.py` 和 `ExtractSynData.py`：单条记录或 synergy API 的探索和 debug，不是正式 pipeline 步骤。
- `aa_seq_to_smiles.py`：核心 **PepLink** 实现。根据 DBAASP metadata 构建 canonical/noncanonical residue、terminal modification，以及 residue 内或 residue 间的 bond/cycle。这是最重要的可复用预处理代码。
- `try.py`：为缺少 PubChem SMILES 的 DBAASP peptide 补结构的早期原型，输出缺失结构的中间 CSV。
- `correct_SMILES_offered_by_DBAASP.py`：重建 DBAASP 提供的 peptide structure 以保留 stereochemistry，并更新 merged SMILES/Evo MIC 文件。它晚于 `try.py`，属于最终化学结构清理血缘。
- `concentration_unit_transfer.py`：最早的 MIC 单位转换原型。`concentration_unit_transfer_new.py` 面向 19-task wide table；`concentration_unit_transfer_all_bact.py` 扩展到全部 bacteria 并计算 mean；`concentration_unit_transfer_Evo.py` 是最终 long-format DBAASP converter，生成 `DBAASP_id_bact_name_SMILES_MIC_Evo.csv` 和 strain count。
- `APEX_in_house_to_SMILES.py`：把 in-house APEX 表转换为早期 wide SMILES 格式。`APEX_in_house_to_SMILES_merge_w_DBAASP.py` 合并该早期格式。`APEX_in_house_to_SMILES_Evo.py` 是最终 long-format merge，生成 `DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv`。
- `convert_EVO_smiles_MIC_to_SELFIES_tokens.py`：最终 AMP SMILES→SELFIES→IBM tokenizer ID cache，并删除 invalid、UNK 或超过 1024 token 的记录。`convert_EVO_smiles_MIC_to_SELFIES_token_SM.py` 是对应的小分子二分类转换器。
- `DBAASP_SELFIES_Token_see.py` 和 `debug_notebook.py`：只用于 tokenizer vocabulary 检查。`debug.py` 把小分子 SELFIES 导出给外部 `mdlm` 项目。
- `bacteria_get.py`：统计 DBAASP JSON 中 strain/species 出现次数和 activity unit 变体，为 mapping 构建提供探索性支持。
- `canonical-peptide-check.py`：检查 canonical peptide 内容。`smiles_to_peptide.py` 把 peptide-like SMILES 反向转换为 D/L residue sequence，并被 peptide/non-peptide 标签脚本复用。
- `DataCheck.ipynb`：大型探索性数据检查 notebook，不是确定性的 build step。

#### Genome 与 text

- `ATCC_genome_get.py`：使用 ATCC API 下载 FASTA。当前 active ID set 只有 BAA-3170/BAA-3197；更早的 mapping 构建行为已被注释或属于历史代码。
- `ATCC_genome_annotation_get.py`：对应的 ATCC GenBank annotation 下载器，当前同样只面向 BAA-3170/BAA-3197。
- **缺失代码：** genome window 划分和 Evo-2-40B 第 46 层 feature extraction；当前只保留最终 `.pt` embedding。
- `discription_generation.py` 与 `discription_generation_w_ATCC.py`：为 ATCC-mapped strain 调用 Qwen 检索和生成描述的近重复脚本。`discription_generation_wo_ATCC.py` 判断并生成没有 genome 的 strain 描述。`discription` 的拼写错误属于历史命名。
- `Get_text_embedding_wo_genome.py`：最终 text-only strain 的 Me-LLaMA embedding 生成脚本。`Get_text_embedding.py` 当前硬编码到无关的 `Ben_ApexOracle_test` 路径，而 canonical 路径已被注释，因此应视为后续测试副本，不是最终 build driver。
- `rename_judge_text_file.py`：把文件名中的下划线替换为全角分隔符，以匹配 strain naming。
- `resistant_gene_check.py`：解析 BAA-3170 annotation，并请求 Gemini 判断 gene/product 是否可能参与耐药；属于解释性探索。`train_genome_mcr_check.py` 扫描训练 annotation 中的 MCR product。二者都不生成核心训练标签。

#### Synergy

- `get_synergy.py`：较早的 DBAASP-only synergy 提取脚本。
- `get_synergy_Evo.py`：后续 DBAASP 提取、strain name mapping、text 可用性判断和 partner antibiotic PubChem 查询，生成 `synergistic_pairs_Evo.csv`；这是论文 synergy 原始数据血缘。
- `DataPrepare/Data/inhouse_synergy/get_inhouse_synergy.py`：把两个 strain 的 in-house FICI 原始表和 peptide master list 转换为 sequence pair。
- `DataPrepare/Data/inhouse_synergy/combine_creat_inhouse_synergy.py`：枚举 master peptide list 的所有 pair，用于后续 screening；会生成很大的未标注候选表，不是论文训练数据。
- `synergy_inhouse_to_SMILES.py` 与 `synergy_inhouse_to_SMILES_parallel.py`：使用 PepLink 把 in-house pair sequence 转换为 molecular SMILES；parallel 版本更新且更快。

#### Similarity 与结构探索

- `DataPrepare/get_similarity/extract_training_peptides.py`：从最终 AMP 数据和 DBAASP JSON 构建 linear/cyclic training sequence cache。
- `compute_percent_identity.py`：与当前 Methods 一致的最终 linear/cyclic exhaustive alignment 实现。
- `extract_top_similarity_hits.py`：提取 best hit 和论文格式 summary；`validate_similarity_outputs.py` 检查内部一致性。
- `compare_linear_query_to_apex11.py`：额外与 in-house APEX 1.1 collection 比较，不是论文主表。
- `Morgan_Fingerprint_Similarity.py`：旧的 all-pairs Morgan fingerprint similarity。
- `Morgan_fingerprint_sim_generation.py`：把外部 generation 仓库中的 BAA-3170 生成分子与训练集进行 Morgan similarity 比较；`_SM_rediscover.py` 是 small-molecule rediscovery 版本。
- `group_peptides.py`：较早的 BLOSUM50 clustering/similarity matrix 实验；论文 similarity 表已由 `get_similarity/` 取代。
- `compare_all_mol_diff.py`：serial 且带 debug 限制的 maximum-common-substructure 扫描。`compare_all_mol_diff_parallel.py` 是后续 parallel 版本。`clean_smiles_compare.py` 删除错误等价或 stereochemistry case；`complete_compare_smiles_MIC.py` 连接 mean MIC；`visualize_mol_diff.py` 绘制结构差异。这些属于探索性 structure-activity 分析，不是论文主训练 pipeline。

#### 当前仓库保留的 DLM 数据预处理

- `DataPrepare/MDLM/split_selfies_csv_file.py`：把大型 SELFIES CSV 分成 120 个 shard。
- `tokenize_SELFIES_descriptors_hf.py`：转换为不超过 1024 的 IBM SELFIES token ID，过滤 UNK/invalid molecule，计算 RDKit descriptor，并写入 Hugging Face Dataset shard。
- `filter_non_209.py`：把后续 216-descriptor schema 投影回论文使用的 209 descriptor。
- `label_pep_nonpep.py`：通过 `smiles_to_peptide.py` 给结构标注 peptide/non-peptide，用于 generation guidance classifier。当前存在疑似 bug：`original_columns + 'label'`，同时硬编码 `/data1` 路径。
- `debug_notebook.py` 和 `MDLM_data.ipynb`：descriptor/schema 探索。
- `bash/node_descriptor_H100.sh` 与 `node_descriptor_node_001.sh`：机器特定的 shard launcher；其中 `/data1` 和 node 路径不可移植。

### 不属于 ApexOracle 论文复现主线的代码

- `Fangping_correlation/`：Bullseye/UPenn DNA 和 peptide count 处理。`DNA_reads_all.py`、`DNA_logFC.py` 和 `Peptide_logFC.py` 生成 count、label 或 logFC；`merge_dict_count.py` 合并 replicate dictionary；`peptide_to_APEX_form.py` 导出 peptide list；`prptide_count_average.py` 聚合 count；`get_absent_peptides.py` 比较 absent/present peptide；`calculate_correlation.py` 和 `devide_cntrl_correlation.py` 计算 abundance change 与 APEX MIC prediction 的相关性；`try_load_dict.py` 与 `Correlation.ipynb` 用于诊断。全部依赖外部 `/data/fangping` 或 `/data1` 输入，且没有被论文训练脚本引用，应视为独立的未发表旁支项目。
- `e3nn_playground/`：`e3nn_convolution.py`、`e3nn_tetris_gate.py`、`e3nn_tetris_polynomial.py` 和 `try_e3nn.py` 是 e3nn 教程或学习实验，没有被 ApexOracle import。公开发布分支应删除或移出。
- `PeptideCLM/`：打包进来的第三方 PeptideCLM tokenizer、example 和 notebook，用于 PeptideCLM comparator，但不是 ApexOracle 自有核心代码。必须保留其 README/LICENSE，并明确标注 vendored 来源。
- `GPU_eye.py` 只打印当前 GPU 的空闲、已用和总显存。`run_full.py` 是轮询 scheduler，当 GPU 显存占用低于阈值时启动 `run.py`。`run.py` 会故意占用约 95% 的全部 GPU 显存并等待输入。这些是资源管理或占卡工具，不是科学代码，不应进入公开包。
- `bash/PeptideCLM_benchmarking.sh`：PeptideCLM strain-wise benchmark 的历史 CESGA launcher。
- `environment.yml`：名为 `cold_base` 的完整 Anaconda 环境导出，规模过大，不是最小可复现环境。发布清理时应替换为经过筛选的 environment 或 lockfile。
- `Readme.md`：只有几行的过期说明，仍指向旧 mean-MIC 文件和 `fine_tune_on_DBAASP_SMILES_5_fold_mean_MIC.py`；必须重写，不能作为当前文档使用。

### 复制版本和第三方辅助文件的显式文件名索引

本索引用于确保以后搜索任意被通配符归类的文件名时，都能直接命中本审计；具体语义和终版判断仍以上文为准。

- 较早的双模态或 DP 副本：`MIC_with_text_genome_test_on_non_seen_strains.py`、`MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble.py`、`MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble.py`、`MIC_with_text_test_on_non_seen_strains.py`、`MIC_with_text_test_on_non_seen_species_3_species_5_ensemble.py`、`MIC_with_text_test_on_non_seen_species_11_species_5_ensemble.py`、`DP_inhouse_MIC_with_text_genome_test_on_non_seen_strains.py`、`DP_inhouse_MIC_with_text_genome_test_on_non_seen_species_3_species_5_ensemble.py`、`DP_inhouse_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble.py`、`DP_inhouse_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble_old.py`。
- 论文时期 SM auxiliary 副本：`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_species_11_species_5_ensemble_old.py`。
- Strain-wise comparator 或 feature 变体：`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_ChemBERTa_MLM.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_ChemBERTa_MTR.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MolFormer.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_PeptideCLM.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix_cls_wo_padding.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix_cls_wo_padding_eval.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix_mean_wo_padding.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix_mean_wo_padding_eval.py`。
- 早期 genome ensemble 的完整文件名：`MIC_with_genome_test_on_non_seen_species_3_species_5_ensemble.py` 和 `MIC_with_genome_test_on_non_seen_species_11_species_5_ensemble.py`。
- Fig. 2b 副本：`fine_tune_on_DBAASP_SMILES_5_fold_compare_pre_SSL.py`、`fine_tune_on_DBAASP_SMILES_5_fold_compare_SSL.py`、`fix_ChemBERTa_MLM_mean_emb_on_DBAASP_SMILES_5_fold_mean_MIC.py`。
- Synergy 副本：`synergy_Evo_train_new_reg_MDLM_one_base_model.py`、`synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_train.py`、`synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification_clean.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_classification.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_few_shot.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_classification_few_shot.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_classification_few_shot_inner_prod.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_classification_few_shot_no_lora.py`、`synergy_Evo_train_on_DBAASP_test_on_inhouse_classification_few_shot_w_pred_MIC.py`。
- 生成分子的 fingerprint 变体：`DataPrepare/Morgan_fingerprint_sim_generation_SM_rediscover.py`。
- `DataPrepare/__init__.py`、`compare_APEX/__init__.py`、`PeptideCLM/__init__.py` 和 `PeptideCLM/tokenizer/__init__.py` 只是 package marker，不包含实验逻辑。
- 第三方 PeptideCLM 文件：`PeptideCLM/example_training_script.py`、`PeptideCLM/tokenizer/my_tokenizers.py`、`PeptideCLM/All_CycPeptMPDB_Predictions.ipynb`、`PeptideCLM/CycPeptMPDB_clustering_and_analysis.ipynb`。它们属于上游 comparator 或 tutorial bundle，不属于 ApexOracle pipeline。

### Capsule 与审稿复现历史

- `capsule/` 是大型本地 staging capsule，包含三个 inference-only 路径：最终 strain-wise Fig. 1a/Fig. 2c DLM ensemble、Fig. 1b strict zero-shot classification 和 Fig. 2b cached benchmark。其规模约 157 GB，主要因为 strain-wise checkpoint 和 genome/text embedding 很大。该目录自己的 `AGENTS.md` 记录了详细 resource manifest 和 T4 显存说明。
- `capsule_fig2/` 是实际受存储限制的 Code Ocean capsule，只包含 **Fig. 2b**，约 212 MB。`code/run` 重新加载 frozen feature cache 和五个 regression head，不进行训练或 backbone feature extraction。
- `capsule*/data/source/` 中的文件是从主仓库或外部 `mdlm` 复制的 provenance snapshot，不是额外的 canonical 版本。
- `capsule/code/prepare_non_seen_strains_resources.py` 和 `reproduce_non_seen_strains_mdlm_mtr_fix.py` 打包并重新评估最终 3-group × 7-ensemble strain-wise 结果。`prepare_zero_shot_antibiotic_classification_resources.py` 和 `reproduce_zero_shot_antibiotic_classification.py` 对 3-group × 10-ensemble strict zero-shot 结果执行同样操作。`prepare_fig2b_mic_regression_resources.py` 和 `reproduce_fig2b_mic_regression.py` 构建或运行 cached Fig. 2b 路径。`capsule/code/run` 在三种模式之间分派。
- `capsule_fig2/code/prepare_fig2b_mic_regression_resources.py`、`reproduce_fig2b_mic_regression.py` 和 `code/run` 是只保留 Fig. 2b 的精简副本；其 `data/source/` benchmark 脚本与 `capsule/data/source/` 中的 provenance snapshot 重复。
- 审稿阶段的 `scripts/`：
  - `reproduce_fig2b_mdlm_cached_5fold.py`：使用外部 `mdlm` cache 或评估 DLM feature。
  - `reproduce_fig2b_baselines_online_5fold.py`：在线提取 baseline feature 并训练或评估 head。
  - `cache_fig2b_molformer_fold_eval_features.py`：MolFormer cache builder。
  - `reproduce_fig2b_baselines_cached_5fold.py`：基于 cache 的 baseline evaluation。
  - `reproduce_fig2b_apex_original_5fold.py`：APEX reproduction driver。
  - `sweep_fig2b_cached_baseline_seeds.py`：用于诊断或协调 Fig. 2b 差异的 seed sweep。
- Capsule 的派生 Fig. 2b 指标不能精确匹配当前论文图中的每一个值，而且包含一个当前图中未绘制的 ChemBERTa-MLM mean-pooling 结果。应把 capsule 保留为审稿复现产物，不能用它悄悄替代原始论文结果来源。

### 已知复现缺口与不一致

1. 当前没有 Git 历史，因此不能确定论文最终版本对应的精确 commit。
2. DLM 训练代码、checkpoint 和 Fig. 2b DLM 源代码依赖外部 `/data2/tianang/projects/mdlm`。
3. Evo-2 genome window feature extraction 实现缺失。
4. k-mer ablation 源代码和成功日志缺失。
5. 实际 discrete guided-generation/remasking sampler 位于外部仓库。
6. 最终 species-wise 和 phylum-wise checkpoint/log 无法对应当前论文 aggregate 数字，且若干 fold 缺失。
7. 现存 synergy CV 脚本最接近论文，但 LoRA rank 与论文描述不一致，aggregate 指标也略有差异。
8. 论文描述的不可变 2,732 行 synergy 数据没有保存；当前只有更大的原始表和动态过滤逻辑。
9. 三个已发表 small-molecule 原始数据集存在，但完整 merge/clean 脚本缺失。
10. 湿实验 MIC heatmap、toxicity、in-vivo 分析和最终候选选择记录没有形成可复现代码或数据 pipeline。
11. 大多数脚本包含绝对路径、重复模型定义、隐式全局状态，以及硬编码 device/fold；许多脚本无法从干净 checkout 直接运行。
12. 一些日志或 checkpoint 不完整、损坏或体积极大。存在 checkpoint 不代表训练已经完成；必须同时核对配套日志和预期 ensemble 数量。

### 公开发布前的安全阻塞项

- 以下文件中包含明文 API 或服务凭据：`DataPrepare/discription_generation.py`、`discription_generation_w_ATCC.py`、`discription_generation_wo_ATCC.py`、`get_synergy_Evo.py`、`resistant_gene_check.py` 和 `compare_APEX/APEX_all_data.sh`。
- 所有嵌入代码的凭据都应视为已经泄露：必须撤销或轮换，从全部文件和未来 Git 历史中清除，并改为通过环境变量或有文档说明的 secret manager 读取。
- 不要把凭据值复制到 issue、日志、文档或审稿复现产物中。

### 建议的代码库清理顺序

1. 撤销并删除全部明文凭据，同时加入 secret scanning。
2. 为最终 AMP、小分子、synergy、genome、text 和 checkpoint 产物建立不可变 manifest，记录 hash 和行数。
3. 抽取一个共享 ApexOracle library，统一 dataset/mapping、split protocol、fusion block、head、metric 和 checkpoint schema；每个论文实验只保留小型 config-driven 入口。
4. 首先把高置信度 strain-wise 和 strict zero-shot 路径做成正式支持的 quickstart。
5. 在声称完整复现之前，解决或明确归档 species/phylum、synergy rank、k-mer 和 Fig. 2b 指标不一致。
6. 清晰拆分外部项目：要么在许可证允许的前提下 vendoring 固定版本的 DLM/generation 代码，要么把它们声明为带版本的外部依赖。
7. 把历史副本、W&B 日志、notebook、旁支项目、巨型 checkpoint 和 reviewer capsule 移出源代码包；不要盲目删除 provenance，而应保留机器可读的 archive manifest。
8. 用最小且经过测试的安装说明、数据下载说明、实验配置、预期指标和容差、license/citation 以及清晰的支持矩阵，替换当前 `environment.yml` 和 `Readme.md`。
