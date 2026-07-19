## AGENTS.md 维护语言

- 本项目的 `AGENTS.md` 应尽量使用中文记录和维护。
- 代码文件名、路径、命令、模型名称、指标缩写以及没有自然中文译名的专有名词可以保留英文。
- 新增审计结论时，应明确区分“已由代码和日志验证的事实”“根据现有证据作出的推断”和“仍待作者确认的事项”。
- 当前发布重构的阶段、已确认决策和验收标准记录在 `REFACTOR_PLAN.md`；执行过程中应同步更新其中的状态，避免计划与代码迁移脱节。

## 环境说明

- 默认使用 conda 的 `base` 环境。在机器 `sn4622119311` 上，conda 位于 `/home/tianang/anaconda3/bin/conda`。
- 不需要在沙箱中运行代码，因为沙箱可能阻止 GPU 访问。
- `/data2/tianang/projects/mdlm` 中的所有代码都必须在 `mdlm` conda 环境中运行。需要直接mdlm训练推理的代码可能都在那里面
- 之前还有很多实验是在node002上面做的，你可以找到对应的代码在node002的 /data1/tianang/Projects/Synergy。在node002上我们同样是使用的conda的base环境完成的Synergy的实验。node002的conda路径是：/data1/tianang/anaconda3/bin/conda
- /data2/tianang/projects/discrete-diffusion-guidance 里面是所有我们使用的generate peptide用的代码和仓库
- /data2/tianang/projects/evo2 里面是我们使用的embedding genome的代码
- **作者于 2026-07-19 确认的边界：** 当前仓库不重构或重跑 Evo-2 genome embedding extraction，直接消费 `DataPrepare/Data/Genome_embs` 中的预计算 tensor。未来整合后的 ApexOracle 主仓库可以在 `external/evo2` 使用固定 clean commit 的 Git submodule；权重和 embedding 数据不进入 submodule。
- **已验证事实：** 当前 567 个 embedding 共 3,437,540,485 bytes，逐文件 SHA-256 manifest 位于 `experiments/evo2_genome_embeddings/file_manifest.csv`，其中三份论文数据匹配 563 个。全部已匹配 tensor 为 `torch.bfloat16`、hidden dimension 8192。重构后的只读 safe loader 完整重算 reviewer scaling CSV/PNG 后逐字节一致。当前外部 Evo-2 HEAD `afd0dae0a4bb25f3ca55f171fbdac4907b937afd` 的 commit object 存在，但 checkout dirty，且没有原始 extraction log 证明该 commit 是精确 producer，因此仅作为未来 submodule candidate。
- 论文最终绘图在 SSH host alias `Mac` 上维护；主 notebook 为 `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`。
- Mac 的 conda 位于 `/Users/kirianozan/Documents/anaconda/anaconda3/bin/conda`。后续论文绘图统一使用其 `base` 环境；已验证包含 Matplotlib 3.7.1、Seaborn 0.12.2、NumPy 1.24.3 和 nbformat 5.7.0。
- 通过非交互 SSH 生成图片时可使用 `MPLBACKEND=Agg .../conda run --no-capture-output -n base python ...`；在 notebook 中交互运行时继续使用 base kernel 即可。

## Git 与发布状态

- 当前 Codex 工作区的 `.git` 是只读保护挂载，本地可用 Git metadata 位于被忽略的 `.git-state/`；操作命令需要使用 `git --git-dir=.git-state --work-tree=.`。
- 本地 `main` 已对齐并跟踪远程 `origin/main`；annotated tag `legacy-code-snapshot-2026-07-17` 指向脱敏 legacy 快照血缘，已完成的 Fig. 2b 重构分支为 `agent/paper-release-refactor`。
- `DragonDescentZerotsu/Synergy` 已完成前两批发布：初始 PR #1 已合并到远程 `main`（merge commit `9427374`）；Fig. 2b paper-compatible wrappers、MolFormer revision 固定、正式 35-fold 结果和对应审计文档通过 PR #2 合并（merge commit `24d975c`）。
- `agent/paper-release-refactor` 远程分支保留 PR #2 的提交历史；本地历史曾因 GitHub App 重建 parent 而拥有不同 commit SHA，但最终 tree 已在合并前逐层核验一致。判断早期同步内容一致性时应比较 tree SHA。
- **2026-07-18 GitHub 状态核验：** 本机已安装 `gh` 2.96.0，并以 `DragonDescentZerotsu` 成功认证；Git operations protocol 为 HTTPS，token scopes 包含 `repo` 和 `workflow`。仓库 `origin` 已切换为 `https://github.com/DragonDescentZerotsu/Synergy.git`，后续可使用普通 `git fetch/push` 和 `gh` 工作流。
- **已由 GitHub PR API 验证的事实：** PR #2 已合并；合并前没有评论、review thread、commit status 或 GitHub Actions run。仓库当前仍未配置针对该 PR 的自动 CI，合并依据是本地 11 项测试、脚本检查、结果审计和论文编译核验。
- 本地 annotated tag `legacy-code-snapshot-2026-07-17` 已成功推送到 GitHub；`archive/legacy-code-snapshot-2026-07-17` branch 继续作为额外恢复点，不再代替正式 tag。

## 模型权重统一登记

- `configs/model_weights.yaml` 是权重当前位置、文件身份、消费实验和未来迁移路径的 canonical manifest；面向维护者的说明位于 `MODEL_WEIGHTS.md`。
- 权重二进制不进入 Git。未来统一本地根目录约定为 `${APEXORACLE_WEIGHTS_DIR:-weights}`；实际移动前必须先核验 SHA-256、下载 URI 和再分发许可，并让加载代码通过 manifest ID 解析。
- **作者于 2026-07-18 确认的决定：** 修订后的 Fig. 2b DLM-only benchmark 使用 `/data2/tianang/projects/mdlm/Checkpoints_fangping/best_2.ckpt`，SHA-256 为 `fbbcc65f85013297212342e7d3286fc9b3ab6fbf0d9b28a0407e11d63b875e59`。这是新 benchmark 的已确认权重；它是否是旧论文运行的精确 checkpoint 仍应标为高置信度推断。
- **已由 node002 原始目录和源码验证的事实：** 24-layer、hidden size 1024 的 `best.ckpt` 原始训练目录为 `node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.05.06/112126/checkpoints`；该目录现存 `best.ckpt`、`last.ckpt` 和 step 960000–1000000 的 checkpoint，均为 5,268,558,165 bytes。原始 `diffusion.py` 从训练目标中直接返回 `loss + 0.1*reg_mse`，`models/dit.py` 构建 209-descriptor regression head，因此这是从训练开始就使用联合 DLM+MTR 目标的模型。
- **已由 node002 checkpoint 验证的事实：** 存在 12-layer、hidden size 768 的 joint DLM+MTR checkpoint：`node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.04.29/165523/checkpoints/best.ckpt`，global step 650032，SHA-256 `3c612c9c68b9ee72c077dc1492153fa30d5c9fa4cb1753355bf146cff616c9d6`，包含四个 `backbone.regression.*` 参数。这与 12-layer/768 的 DLM-only `best_2.ckpt` 构成容量匹配候选对。
- **仍待实验核验的事项：** 上述 12-layer 配对尚未运行共同数据五折 benchmark，而且两次预训练的 learning rate、global batch size 和最佳 step 不同，因此只能称为容量匹配比较，不能称为除 objective 外所有条件完全相同的单变量消融。在当前机器、node002、W&B 和公开 Hugging Face 权重中仍未找到 24-layer/1024 的纯 DLM checkpoint。
- Fig. 2b 的两个 DLM 本地 checkpoint、APEX checkpoint 和四个 Hugging Face 模型均已进入 manifest。ChemBERTa-MTR、ChemBERTa-MLM 和 PeptideCLM 的上游 revision 尚待固定；MolFormer revision 已固定。

## 论文及审稿回复路径

- 本项目对应论文：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`
- 审稿意见及回复草稿：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Response to reviewers letter.docx`
- **2026-07-18 已完成的 Fig. 2b 修订：** 回复信中关于各 encoder 是否使用相同数据、五折不确定性和原 27.1\% 表述的回答已经改为已完成实验及正式数值；论文 Fig. 2b 图注、Results 和 Methods 已同步修改。作者随后更新了完整 `Fig2_2.pdf`；经实际渲染核验，panel b 现为 10,886 个共享分子的七模型结果，显示五折 sample s.d. error bars，柱上三位小数与正式结果一致。最新 TeX 已再次完整编译为 28 页。
- **当前正式结果：** 文稿、回复信和图片使用 24-layer joint DLM `0.5386 ± 0.0250` 与 12-layer DLM-only `0.3765 ± 0.0239`，相对第二名提升表述为 29.1\%。正文当前仍把优势解释为 joint DLM+MTR objective，但没有明确写出两个 DLM checkpoint 的容量不同；因此“objective 导致提升”仍应视为尚未完成容量控制核验的解释，而不是现有 benchmark 已证明的事实。
- **仍待实验核验的事项：** 如果后续采用 12-layer joint 候选作为主比较，必须再次同步 Fig. 2b、Results、回复信结果段和 29.1\% 相对提升；在该实验完成前，当前 24-layer joint 正式结果和图片保持不变。

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

## 2026-07-19 hierarchical MIC 统一重构审计

### 已由代码、日志、checkpoint 和跨机器核验的事实

- Fig. 1a / Fig. 2c 最终 strain-wise driver、MIC CSV、small-molecule auxiliary CSV、strain
  mapping 和 taxonomy mapping 在本机与 node002 上的 SHA-256 一致；genome embedding、
  ATCC text embedding、text-only embedding 和 FASTA 的文件数量及文件名清单也一致。
- 历史 split 构造使用无序 `set`，并把 taxonomy alias 直接 `extend` 到共享 list。三个 fold
  由独立 Python 进程运行，而日志没有记录 `PYTHONHASHSEED`。当前源码重新运行得到的 membership
  因此不能被视为 2025 年 checkpoint 的精确历史 membership。
- `experiments/hierarchical_mic/strain/legacy_protocol_manifest.json` 明确区分了
  `PYTHONHASHSEED=0` 的确定性候选 membership 与历史日志中的权威样本计数；任何新运行不得把
  候选 manifest 标成精确历史 split。
- 21 个 checkpoint 的 `3 × 7` 网格完整。group 0/2 的 14 个文件只保存实际消费的
  fusion/head state；group 1 的 7 个文件还包含名为 `ChemBERTa_state_dict` 的 131-key、
  12-layer/768 MDLM backbone。三个 group 的 optimizer 都具有相同的 5 个参数组和 49 个
  state entries，额外 backbone 不在下游 optimizer 中。
- strain/species/phylum 三条路径已统一到 `src/apexoracle/`，canonical runner 为
  `scripts/reproduce/run_hierarchical_mic.py`，唯一配置为
  `configs/hierarchical_mic/legacy_mdlm.yaml`。模型、四路训练、评估、checkpoint 和 ensemble
  只保留一份；三个协议只通过 split adapter、group 名称与输出路径区分。
- 四条单批次训练路径位于 `src/apexoracle/training/hierarchical_mic.py`。逐项测试在固定随机状态下
  比较 logits、loss、所有参数 gradient 和 Adam 更新后的
  参数，四种 modality/task 组合均完全一致。强制触发 epoch-5000 clipping 后也确认：历史实现
  不裁剪 text attention，classification 分支裁剪的是 `reg_head` 而不是 `cls_head`；为保持行为
  当前共享实现显式保留这一异常契约。
- 四条 modality/task optimizer-step 等价测试在 CPU float32 下逐位通过；genome+text
  regression 另在 H100 CUDA autocast + GradScaler 下逐位通过。H100 合成测试使用有限的
  `init_scale=128` 并要求 gradients 全部有限；默认 `65536` 会让这个小型合成 fixture 的两侧
  同时 overflow 并跳过 step，因此不能用该 overflow case 宣称参数更新已验证。正式 driver
  继续使用历史默认动态 GradScaler。
- 统一 outer runner 已覆盖 epoch-0 baseline 与逐 epoch evaluation、不同长度 DataLoader 的
  `zip_longest(fillvalue=None)` 顺序、CosineAnnealingLR 序列、prediction/loss 分区、species
  插入顺序、`len <= 1` 分区指标的 `-1000` sentinel、strict `>` best-metric tracker 和七键
  checkpoint payload。评估 helper 不切换 module mode，因此 held-out-fold selection 继续受到
  train-mode dropout 影响；这是经测试冻结的历史行为，不是推荐的新评估协议。
- **已由 node002 源码和日志验证的事实：** 找回的 phylum-wise MDLM 终版候选 SHA-256 为
  `36ef70bc4a20f2d94294e40b027be7b41c0c8a722c97a09bee856916622789e1`，模型与训练契约同
  strain/species。统一 adapter 的 Fungi 数据计数与 node002 终版日志逐项一致；species group 0
  计数也与本机 MDLM 日志一致。三个协议的真实数据 dry-run 和 H100 一轮四路训练集成 smoke
  均已通过。
- 被统一 runner 替代的 15 个 root DP/in-house/SM/pooling/eval 脚本、capsule 中第二份 strain
  driver 和旧打包脚本已删除；完整恢复点为 `legacy-code-snapshot-2026-07-17`。Fig. 2c 的四个
  不同 encoder comparator 与尚未迁移的 modality ablation 血缘仍保留。

### 根据现有证据作出的推断

- node002 当前 driver 的修改时间位于 group 1 长任务运行期间；结合 group 1 独有的额外 MDLM
  payload，该进程很可能持有修改前的内存代码版本。修改时间只是弱证据，不能据此恢复旧实现。

### 仍待作者或旧源码确认的事项

- group 1 当时是否在线调用 frozen MDLM backbone 生成 molecule feature，还是只把未消费的
  backbone 一并写入 checkpoint；“不在 optimizer 中”只能证明没有被下游训练更新。
- 三个历史 fold 的精确 `PYTHONHASHSEED` / membership 仍未恢复。当前正式数值继续以保存的
  checkpoint 和完整日志为准，不以候选 split 重新训练后覆盖。
- 其余 20 个 strain checkpoint 的逐文件 SHA-256 与固定批次推理仍未全部登记；这不影响统一
  runner 的代码迁移结论，但仍限制对全部历史 checkpoint 逐文件身份的声称。

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
| Fig. 1a strain-wise 泛化；Fig. 2c 最终 DLM 和 7 模型 ensemble | `scripts/reproduce/run_hierarchical_mic.py --protocol strain`；checkpoint：`Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles` | **最终版 / 高置信度。** 三个完整 group 的 ensemble R2 分别为 0.4057、0.6889、0.6434，平均值恰好为论文中的 0.5793；每组均有 7 个模型。 |
| Fig. 1a / Fig. 2f phylum-wise holdout | `scripts/reproduce/run_hierarchical_mic.py --protocol phylum`；checkpoint 位于 `.../3_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles` | **代码路径已统一并验证；历史结果仍为中等置信度。** node002 找回三个 MDLM 终版日志/checkpoint，Fungi 分区与新 adapter 完全一致；现存指标与论文 0.3744 仍不一致，因此不能声称完整恢复论文绘图运行。 |
| Fig. 1a / Fig. 2d、g species-wise 11-cluster holdout | `scripts/reproduce/run_hierarchical_mic.py --protocol species`；checkpoint 位于 `.../11_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles` | **代码路径已统一并验证；历史结果仍为中等置信度。** 现存终版日志只有 group 6–10；论文中完整 11-cluster 汇总运行没有保留下来。 |
| Fig. 2c strain-wise molecular encoder 比较 | 统一 runner 的 DLM，以及保留的 `..._ChemBERTa_MLM.py`、`..._ChemBERTa_MTR.py`、`..._MolFormer.py`、`..._PeptideCLM.py` | DLM 终版为高置信度。其余脚本是不同 encoder comparator，不属于本次删除的同模型复制版本；部分 comparator checkpoint 或日志仍不完整。 |
| Fig. 2c Evo-2 与 k-mer 消融 | 当前没有对应源代码；`Checkpoints/KMER_genome_text_learnable_emb` 下只有 2026 年的部分或失败日志 | **缺失。** 论文报告 R2 下降 11.6%，但当前 KMER 日志没有完整结束，仓库内也没有包含 k-mer 实现的 Python 文件。不能声称当前仓库能够复现该结果。 |
| Fig. 2b 不使用 strain knowledge 的五折 molecular representation benchmark | DLM 原始脚本位于外部 `/data2/tianang/projects/mdlm/DBAASP_MLM_MDLM.py`，capsule 的 `data/source` 中有副本。baseline 为 `fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py`、`fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py`；APEX 为 `compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py` | 这是原论文实验代码家族。当前论文图中数值目测约为：DLM MTR+DLM 0.530、ChemBERTa MTR 0.417、DLM MLM 0.408、ChemBERTa MLM 0.226、PeptideCLM 0.376、MolFormer 0.371、APEX 0.403。审稿阶段建立的 cache 复现资源并不能精确对应当前图中的全部数值，只能视为派生复现产物。 |
| Fig. 1b 严格 target-strain zero-shot 小分子分类 | canonical 入口 `scripts/reproduce/run_antibiotic_classification.py --mode strict-zero-shot`；checkpoint：`.../antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles` | **最终版 / 高置信度，已完成行为保持迁移。** 完整 held-out target-strain 数据不进入训练，但仍逐 epoch 用于 best-AUROC checkpoint selection。完整 ensemble 指标：E. coli `#004` 为 0.9360 AUROC / 0.5890 AUPRC；A. baumannii 17978 为 0.7262 / 0.3243；S. aureus RN4220 为 0.7679 / 0.1655。30 个 checkpoint 和 3 个完成日志网格完整；group 0 / ensemble 0 已在 H100 上与 capsule 做到 2,335 条 logit 逐值一致。 |
| Fig. 1b fine-tuned ApexOracle | 同一入口的 `--mode fine-tune --fold N`；checkpoint：`MDLM_fix_cls_10_fold_ensembles` | **代码路径已统一 / 历史结果证据不完整。** 该模式保留目标 strain 上的五折 KFold fine-tuning，但现存 checkpoint 只有 77/150，14 个日志中只有 6 个包含最终汇总。`--mode molecule-only` 对应旧 `wo_SAND`，完整保留仅用 DLM molecule embedding 的对照；它不是 strict zero-shot ApexOracle。 |
| Synergy 二分类结果 | `synergy_Evo_train_new_reg_MDLM_one_base_model_classification.py`；checkpoint：`.../strain_wise_synergy/MDLM_3_fold_ensembles_1_base_model_cls` | **可能的最终版 / 中等置信度。** 使用 `synergistic_pairs_Evo.csv`、strain-wise 三折划分、单个完整 MIC base checkpoint、FICI 二分类和 7 个 ensemble。现存三个 fold 的 AUROC/AUPRC 分别为 0.6690/0.6159、0.7614/0.6853、0.8489/0.9307，未加权平均约为 0.7598/0.7440，与论文 0.7539/0.7454 接近但不完全相同。论文写 LoRA rank 64，而该 CV 脚本对 fusion 使用 1024、对 head 使用 256；rank 64 出现在后续 all-data noisy classifier 中，因此需要作者进一步确认精确版本。 |
| Synergy 或 guidance 之前使用的完整数据 MIC base model | `train_on_all_data.py`；checkpoint 家族 `Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask`，尤其是 `noise_guidance_best_R2_all_peptide_epoch_100.pth` | 这是最终 synergy 和 noisy guidance 脚本共同使用的最匹配 base model。`train_on_all_data.py` 本身保存到 `all_AMP_SM_data_train/MDLM_MTR_fix_cls_wo_pad`；路径和命名在开发过程中发生过变化，重构时必须保留实际 checkpoint 来源。 |
| Noisy synergy/peptide guidance classifier | `synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py` 和后续较干净的 `..._all_data_classification_clean.py` | **论文后实现支持。** `clean` 版本使用 rank-64 LoRA，并在 `.../synergy_judger/cls` 下保存 noisy synergy classifier。它们是 all-data guidance head，不是论文三折 synergy benchmark。`DataPrepare/MDLM/label_pep_nonpep.py` 用于准备 peptide/non-peptide 标签。 |
| Guided molecule generation、remasking sampler 和 256-step 三阶段 guidance | 不在当前仓库；分析脚本指向 `/data2/tianang/projects/discrete-diffusion-guidance` | **外部依赖 / 缺失。** 当前仓库包含 predictor、保存的输出或图片以及 similarity 分析，但不包含实际生成 ApexOracle-3/12/23 的 sampler。论文参数为 256 steps、MIC target 1、sigma 从 0.5 线性降到 0.2、`t_on=0.55`、`t_off=0.45`，两个阶段的 guidance strength 为 15。 |
| Fig. 3（TeX 中使用文件 `Fig4.pdf`）guided/unguided 预测和最终验证分子 | `DataPrepare/Morgan_fingerprint_sim_generation*.py` 中的生成结果分析、`synergy_Evo_train_on_DBAASP_screen_inhouse_pairs.py` 中的筛选逻辑，以及 `paper_figs/` 下的 PDF | 当前只剩部分派生分析。候选生成和最终选择流程依赖外部代码或已经不完整；湿实验结果也没有在本仓库形成计算复现流程。 |
| 附录 modality ablation | 较早的 genome-only、text-only、genome+text 脚本家族，以及 `Checkpoints/genome`、`Checkpoints/text`、`Checkpoints/genome_text` | **可能的来源家族 / 中低置信度。** 当前没有一个干净的入口或完整指标表可以把现存 checkpoint 精确连接到最终附录图。重建前应先核对原始绘图数据。 |
| Attention 或耐药基因解释 | `DataPrepare/ATCC_genome_annotation_get.py`、`DataPrepare/resistant_gene_check.py`、`DataPrepare/train_genome_mcr_check.py`，以及大型训练脚本中的 attention 输出 | 属于探索或审稿支持代码；不存在自包含的最终 attention figure 流程。 |
| ApexOracle-3/12/23 sequence similarity 表 | `scripts/reproduce/run_sequence_similarity.py`；`src/apexoracle/evaluation/sequence_similarity/` | **canonical / 已验证。** 实现 Methods 中的 Biopython global alignment、BLOSUM62、gap-open 10、gap-extension 0.5、exact-match PID 和 cyclic exhaustive rotations。ApexOracle-3/23 全量输出与历史 CSV 逐字节一致；ApexOracle-12 的论文数值已复算，但旧 full CSV 未保存且 top hit 有四个 complete ties。 |
| 审稿回复中的 Evo-2 embedding 缩放说明 | `scripts/plot_evo2_genome_embedding_abs_mean_distribution.py` | **审稿阶段 / 高置信度。** 生成支持固定 `1e14` 缩放因子的 CSV、PNG 和 PDF，统计范围为 563 个实际匹配的 embedding。 |

### 主要训练脚本的版本家族

#### 分层 MIC 回归

- `MIC_with_genome.py`、`MIC_with_genome_no_AMP.py`、`MIC_with_genome_test_on_non_seen_species.py`、`MIC_with_genome_test_on_non_seen_species_3_species.py`、`MIC_with_genome_test_on_non_seen_species_5_fold.py`、`MIC_with_genome_test_on_non_seen_species_{3,11}_species_5_ensemble.py`、`MIC_with_genome_test_on_non_seen_strains.py` 和 `MIC_with_genome_test_on_non_seen_pep_&_non_seen_species.py` 是早期 genome-only/ChemBERTa 原型及划分实验，属于**历史版本**。
- `MIC_with_text_test_on_non_seen_{strains,species_3_species_5_ensemble,species_11_species_5_ensemble}.py` 是 text-only 消融；`MIC_with_text_genome_test_on_non_seen_*.py` 是早期双模态版本。它们属于 modality ablation 血缘，而不是最终 DLM 模型。
- `DP_inhouse_MIC_with_text_genome_test_on_non_seen_*.py` 和同模型的
  `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*.py` 曾构成论文时期的复制脚本家族；
  其 hierarchical MIC 版本现已由统一 runner 替代并从工作树删除，只在 legacy tag 中保留。
- 11-cluster、3-cluster 与 strain-wise 的最终 MDLM 路径现在都应使用
  `scripts/reproduce/run_hierarchical_mic.py`。旧 `..._MDLM_cls_fix.py`、
  `..._MDLM_MTR_fix.py`、pooling 和预计算 feature 变体仅作为历史血缘记录，不再是活跃入口。
- `..._ChemBERTa_MLM.py`、`..._ChemBERTa_MTR.py`、`..._MolFormer.py` 和 `..._PeptideCLM.py` 是 Fig. 2c 的 strain-wise encoder comparator。

#### Molecule-only Fig. 2b benchmark

- `fine_tune_on_DBAASP_SMILES_5_fold_mean_MIC.py` 是早期 ChemBERTa mean-MIC benchmark。`fine_tune_on_DBAASP_SMILES_5_fold_compare.py`、`_pre_SSL.py` 和 `_SSL.py` 是表示或预训练比较实验。`fine_tune_on_DBAASP_SMILES_all_data_for_training.py` 在全部 benchmark 数据上训练；`fine_tune_on_inhouse_SMILES_CV_MIC.py` 是 in-house-only 版本。相对于 `fix_*` 脚本，它们都属于**历史版本**。
- ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM 应优先使用 `fix_*_on_DBAASP_SMILES_5_fold_mean_MIC.py`。`fix_ChemBERTa_MLM_mean_emb_*` 是 mean-pooling 消融；当前论文图中的 ChemBERTa-MLM bar 使用 first-token 版本。
- DLM/MTR benchmark 源代码位于外部 `mdlm` 项目；capsule 中的副本只是审计快照，不能视为第二套 canonical 实现。
- **已由 checkpoint 和日志验证的事实：** `/data2/tianang/projects/mdlm/Checkpoints_fangping/best.ckpt` 是 24-layer、hidden size 1024、global step 800046 的 checkpoint，并含有 `backbone.regression.*` 四个 209-descriptor regression 参数，因此属于 MTR+DLM 家族；当前 `capsule_fig2` 将它用于 `mdlm_dlm_mtr` 是有模型结构依据的。
- **已由 checkpoint 验证的事实：** `best_1.ckpt`、`best_2.ckpt`、`best_3.ckpt`、`1-314000.ckpt`、`2-471000.ckpt` 和 `4-750000.ckpt` 均为 12-layer、hidden size 768 的纯 DLM checkpoint，不含 `backbone.regression.*`。DLM-only 的基础训练实现位于外部仓库的 `diffusion.py`、`models/dit.py` 和相关配置；`DBAASP_MLM_MDLM.py` 是其下游五折 MIC head 代码家族。
- **根据现有证据作出的推断：** `wandb/run-20250421_231424-s58d1559/files/output.log` 的五个 fold 最佳 mean R2 为 0.4132、0.3529、0.4400、0.4134、0.4222，均值约 0.4083，与论文 DLM MLM bar 对应。结合运行时间，最可能使用的是当时已存在的 `best_2.ckpt`；旧日志没有保存实际 checkpoint 路径，因此仍需用新 benchmark 在共享数据协议下核验，不能把 `best_2.ckpt` 记为已完全确认的论文终版。
- 新公平 benchmark 必须分别加载 12-layer DLM-only 和 24-layer MTR+DLM 配置，不能仅通过同一个 `best.ckpt` 生成两个不同标签的结果。
- **已由 node002 原始训练资源验证的事实：** 联合模型的原始目录是 `node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.05.06/112126/checkpoints`。现存七个 5,268,558,165-byte checkpoint 仅覆盖 `best`、`last` 和 step 960000–1000000；早期 checkpoint 已不在该目录。原始 `/data1/fangping/mdlm/diffusion.py` 第 424 行使用 `loss + 0.1*reg_mse`，`models/dit.py` 为 24-layer backbone 构建 209-descriptor regression head，证明该 run 从一开始就是联合 DLM+MTR 训练，而不是先完成纯 DLM 再加入 MTR。
- **已完成的 24-layer 搜索：** 已检查本机和 node002 的 `Checkpoints_fangping`、`/data1/tianang/Projects/Synergy`、`/data1/tianang/Projects/mdlm`、Fangping 原始 output、相关 W&B projects 和公开 Hugging Face repository；没有发现 24-layer、hidden size 1024 且不含 MTR 的纯 DLM checkpoint 或代码路径记录。不得通过删除 `best.ckpt` 的四个 regression 参数把它重新标注为 DLM-only，因为其 backbone 参数也已受联合目标优化。
- **后续发现的 12-layer 容量匹配候选：** `node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.04.29/165523/checkpoints/best.ckpt` 是 12-layer、hidden size 768、step 650032 的 joint DLM+MTR checkpoint，包含 `768→768→209` regression head。它与 DLM-only `best_2.ckpt` 使用相同 small architecture、相同数据配置和长度 1024；但 joint run 使用 learning rate `1e-4`、global batch size 480，而 DLM-only 使用 `3e-4`、768，最佳 step 也分别为 650032 与 621036。
- **解释边界：** 正式共同数据 benchmark 的 `0.3765 ± 0.0239` 与 `0.5386 ± 0.0250` 分别对应 12-layer DLM-only 和 24-layer MTR+DLM。它们可支持“修订实验中的联合目标 DLM checkpoint 优于现有 DLM-only checkpoint”，但不能单独支持“性能差异由 MTR objective 导致”的容量受控消融结论。
- **已由 reviewer 回复原文验证的事实：** reviewer 对 molecular-representation benchmark 的具体问题是各 encoder 是否使用了相同的 train/test 数据。回复承诺取“所有 encoder 都能处理的 molecule ID 交集”，再按 molecule ID 生成唯一一份固定随机种子的五折划分，并让所有模型使用完全相同的 partitions。回复没有承诺为这个 benchmark 改用 scaffold split，也没有承诺新增 validation split。
- **已由新代码和真实数据验证的事实：** 原始 11,401 个 molecule 按论文各脚本自身的 native preprocessing 规则后，ChemBERTa-MTR、ChemBERTa-MLM 和 MolFormer 各保留 10,889 个，PeptideCLM 保留 11,377 个，DLM MTR+DLM 与 DLM-only 各保留 11,082 个，按已确认输入投影并保留原实现的 APEX 保留 11,321 个；全部 encoder 的共同交集为 10,886 个。共享五折大小依次为 2,178、2,177、2,177、2,177、2,177。`eligibility.py` 和 `protocol.py` 分别负责原生可处理性审计与交集/划分；原始数据和生成 CSV 不进入 Git。
- **已由新代码和真实数据验证的事实：** APEX 输入投影中有 1,689 条记录含 noncanonical residue、1,460 条含 D-residue、2,335 条含被线性化的 bond/multichain topology。DBAASP 的正确字段名是 `intrachainBonds`、`interchainBonds` 和 `coordinationBonds`；早先按 `intraChainBonds`/`interChainBonds` 检查得到的“字段为空”结论无效，后续不得沿用。
- **已确认的 APEX 兼容规则：** noncanonical residue 在输入字符串中写为 `X`，cyclic peptide 使用线性 residue 顺序；但 APEX 原始 23-token vocabulary、AAindex embedding、encoder、checkpoint 和 `512→256` regression head 均不得修改。原 APEX 没有 `X` token，因此 `X` 必须按 `compare_APEX/utils.py::onehot_encoding` 的原行为保留在 index 0，不能新增 index 23 或平均 AAindex embedding。
- **已由代码审计验证的事实：** 原始各模型脚本先按各自规则过滤数据，再分别运行 KFold，因此旧结果的 retained molecule set 和具体 fold membership 并不完全相同。旧脚本还会逐 epoch 在 held-out fold 上评估并据此选择 best checkpoint；APEX 的验证路径会保持 regression head 为 train mode，使 dropout 参与选择。这是原论文实现应披露的局限，但 reviewer 本轮没有要求修改模型、head 或 checkpoint-selection 行为。
- **已撤回的错误方案：** 曾计划新增训练折内 10% validation、把所有 comparator 改为统一 `384→128→19` head，并为 APEX 新增 `X` embedding；这些改变超出 reviewer 要求，也会破坏与原模型的公平比较，现已删除。正式修订版只统一 10,886 个 molecule ID 和五个 folds；如以后做严格 train/validation/test 或 scaffold-split sensitivity analysis，必须单独标注，不能替代 paper-compatible benchmark。
- **已实现并由测试验证的基础设施：** `feature_cache.py` 提供带完整 ID 契约的 `.npz` cache，仅用于输入/encoder 审计，不代表统一训练协议。正式训练仍需为各原始训练脚本增加读取共同 IDs/folds 的薄 wrapper，并保留各自模型与 head。
- **已由真实 checkpoint 和全量 cache 验证的事实：** APEX adapter 使用原 23-token embedding 并对完整原 checkpoint 执行 `strict=True` 加载；10,886 条共同样本在 CPU 上成功产生并严格回读 `(10886, 128)` feature。cache 位于被 Git 忽略的 `Checkpoints/fig2b_shared_v1/apex/features.npz`，只作为 adapter 审计产物，不应提交或冒充正式五折结果。
- **已由宿主机诊断验证的事实：** GPU driver 正常，宿主机可见 4 张 NVIDIA H100 PCIe、driver 580.159.03 和 CUDA 13.0。Codex 文件沙箱用隔离的 `/dev` 隐藏了 `/dev/nvidia*`，所以沙箱内的 `nvidia-smi`/PyTorch 会误报 CUDA 不可用；需要 GPU 的命令应按项目约定在沙箱外执行。此前“GPU driver 不可用”的结论错误，不得沿用。
- **已由全量单 epoch smoke test 验证的事实：** `scripts/reproduce_fig2b_baselines_online_5fold.py` 已能让 ChemBERTa、MolFormer、PeptideCLM 和未修改的 APEX 读取共同 10,886 IDs/folds；`scripts/reproduce/run_fig2b_shared_mdlm_online.py` 已分别接入 12-layer `best_2.ckpt` 和 24-layer `best.ckpt`。24-layer checkpoint 的 backbone 参数全部匹配，只有预训练 209-descriptor regression branch 的四个参数作为预期 unexpected keys 被旧协议的 `strict=False` 忽略。
- **已由正式 35-fold 运行验证的事实：** 7-model × 5-fold 共享数据实验已于 2026-07-18 全部完成，35 个模型-fold 组合无缺失、无重复；每个模型的五个 test fold 合计恰好覆盖 10,886 个 molecule。完整输出位于被 Git 忽略的 `results/fig2b_shared_original_protocol/`，小型正式报告位于 `experiments/fig2b_molecule_encoders/results_shared_5fold.md`。
- **正式共同数据结果（五折 mean R² ± sample SD）：** DLM MTR+DLM `0.5386 ± 0.0250`、ChemBERTa MTR `0.4172 ± 0.0275`、APEX `0.4014 ± 0.0146`、PeptideCLM `0.3836 ± 0.0244`、DLM-only `0.3765 ± 0.0239`、MolFormer `0.3678 ± 0.0198`、ChemBERTa MLM `0.2247 ± 0.0131`。相对各自原 retained-set rerun 的绝对变化依次为 `+0.0179`、`-0.0025`、`-0.0036`、`+0.0069`、`-0.0318`、`-0.0048`、`-0.0055`。DLM-only 是唯一变化超过 0.02 的模型，后续论文图和 reviewer response 应使用新值。
- **正式运行保持的原协议：** 每个 fold 使用 200 epochs、batch size 200、Adam、learning rate `1e-4`、模型特有 head、frozen backbone 的训练模式以及 held-out-fold checkpoint selection。加速只包括 fold/GPU 并行和一次性缓存 held-out fold 上确定性的 frozen-backbone `eval()` feature；训练阶段的 backbone dropout 仍逐 batch 开启，APEX held-out head dropout 也保持原行为。
- **随机性边界：** 原训练脚本没有显式设置 PyTorch seed，本次没有额外添加 seed；每个任务在 `metrics.json` 中记录了 `initial_torch_seed`。因此正式表是一轮忠实的 stochastic rerun，不应声称跨硬件逐 bit 可复现。固定 `random_state=42` 的是共同 molecule-level 五折 membership。
- **MolFormer 兼容性固定：** 正式运行使用本地完整缓存的 `ibm/MoLFormer-XL-both-10pct` revision `7b12d946c181a37f6012b9dc3b002275de070314`。Hugging Face 当前 `main` revision 依赖本环境不存在的 `transformers.masking_utils`；固定历史 revision 只恢复原兼容代码和已有权重，没有改变模型结构、checkpoint 或训练超参数。
- **运行环境事实：** 正式任务中 GPU 1 曾达到 88°C 并触发 software thermal slowdown，因此未完成的任务被重新分配到其他健康 GPU；GPU driver 本身正常。最终纳入汇总的所有 fold 都是完整从 epoch 1 跑到 200 且退出码为 0 的任务。
- `compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py` 是最终 APEX benchmark 版本。`APEX_train_DBAASP_MIC.py`、`APEX_train_DBAASP_MIC_5_fold_mean.py`、`APEX_train_inhouse_MIC.py`、`fine_tune_on_DBAASP_SMILES.py` 和 `deubg.py` 是早期或 debug driver。`APEX_models.py`、`APEX_trainer_CV.py` 和 `utils.py` 是复制过来的 APEX 支持代码。`APEX_all_data.sh` 是历史集群启动脚本，其中包含必须撤销和删除的明文 W&B 凭据。

#### 小分子抗生素分类

- 三菌株分类家族已迁移到 `scripts/reproduce/run_antibiotic_classification.py` 和
  `configs/antibiotic_classification/legacy_three_strain.yaml`；三份 MDLM root driver、早期
  ChemBERTa driver 及两份机器专用 launcher 已删除并由 `legacy-code-snapshot-2026-07-17`
  tag 保留。
- `strict-zero-shot` 不在目标 strain 上训练；`fine-tune` 使用目标 strain 五折；
  `molecule-only` 对应旧 “wo_SAND”，仅训练 DLM molecule embedding classification head。
- full-fusion 的 held-target selection 中 classification head 历史上未调用 `eval()`，dropout
  仍然开启；checkpoint 的 AUPRC 又在 AUROC 改善保存之后才更新。这两项不理想行为均由
  canonical runner 和测试明确冻结，不得在无新实验的情况下“修正”。

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
- PepLink chemistry core 已由作者独立发布为 `PepLink==0.1.1`：仓库
  `https://github.com/DragonDescentZerotsu/PepLink`，tag `v0.1.1`，commit
  `cec2a02427766e4ba95806924801af31bdcc9939`，MIT license。ApexOracle 不 vendor、不使用
  submodule，只通过 `src/apexoracle/data/peplink_adapter.py` 调用公开 API。旧
  `aa_seq_to_smiles.py` 仅因仍有 legacy data driver import 而暂留，不再作为 canonical 实现。
- **已验证事实：** 独立 PepLink 测试 22 passed；内置 amino-acid mapping 与论文文件 SHA
  一致。179 条历史 structure correction 中 177 条与 v0.1.1 逐字符串一致；DBAASP 19000
  和 21769 的差异仅为 v0.1.1 `FragmentParent` 移除游离 fragment，全部 179 条均为 fragment
  parent equivalent。这两个 ID 在最终 token cache 中影响 9 行。论文复现必须使用 frozen
  paper CSV 的 SHA，新数据才使用 v0.1.1 normalization。
- `try.py`：为缺少 PubChem SMILES 的 DBAASP peptide 补结构的早期原型，输出缺失结构的中间 CSV。
- `correct_SMILES_offered_by_DBAASP.py`：重建 DBAASP 提供的 peptide structure 以保留 stereochemistry，并更新 merged SMILES/Evo MIC 文件。它晚于 `try.py`，属于最终化学结构清理血缘。
- `concentration_unit_transfer.py`：最早的 MIC 单位转换原型。`concentration_unit_transfer_new.py` 面向 19-task wide table；`concentration_unit_transfer_all_bact.py` 扩展到全部 bacteria 并计算 mean；历史最终 long-format 行为现由 `src/apexoracle/data/amp_mic.py` 和只读入口 `scripts/prepare_data/build_amp_mic_dataset.py` 取代。
- `APEX_in_house_to_SMILES.py`：把 in-house APEX 表转换为早期 wide SMILES 格式。`APEX_in_house_to_SMILES_merge_w_DBAASP.py` 合并该早期格式。最终 long-format、merge 和 AMP token filter 现由 `src/apexoracle/data/amp_training_data.py` 与 `scripts/prepare_data/build_amp_training_dataset.py` 取代；`convert_EVO_smiles_MIC_to_SELFIES_token_SM.py` 仍是尚未迁移的小分子二分类转换器。
- **2026-07-19 已验证事实：** canonical MIC 重建得到相同 105,547 行，ID/strain/SMILES 精确一致，MIC 最大绝对误差 `4.55e-13`，没有记录超出 `1e-12` tolerance。历史 structure correction 是先用旧 SMILES 分子量换算 MIC、再原地替换展示 SMILES；新实现通过 179 条只用于分子量的 override 显式复现，不覆盖任何原始文件。frozen in-house long table 合并后的 121,265 行 CSV 和固定 IBM tokenizer revision 后的 120,955 行 token cache 均逐字节一致；310 行仅因超过 1024 tokens 排除，invalid/UNK 为 0。PepLink 新建 in-house structure 与 legacy 只差 terminal `[OH]`/canonical `O`，归一化后 15,718/15,718 行一致。
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

- sequence similarity 已迁移到 `src/apexoracle/data/peptide_similarity.py` 和
  `src/apexoracle/evaluation/sequence_similarity/`，唯一入口为
  `scripts/reproduce/run_sequence_similarity.py`；旧 `DataPrepare/get_similarity` driver 由
  `legacy-code-snapshot-2026-07-17` tag 追溯。
- **已验证事实：** paper cache 使用 uppercase training sequences；当前源数据可逐字节重建
  13,077 条 linear 和 1,039 条 cyclic cache。canonical 全量重算与 ApexOracle-3/23 的四份
  历史核心 CSV 逐字节相同，三条 lead 的最大 PID 为 0.3667/0.3571/0.3684。
- **证据边界：** ApexOracle-12 未保存历史 full CSV；其最大 PID 有四个 complete ties。论文
  展示 DBAASP 15510，稳定输入顺序选择 9800，二者指标完全相同。保留当前 DBAASP sequence
  大小写会把 ApexOracle-23 最大 PID 改为 0.3158，只能作为 chirality sensitivity。
- 旧 `compare_linear_query_to_apex11.py` 是额外的 in-house APEX 1.1 collection 对照，不是
  论文主表。
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

- **已删除并由 legacy tag 保留：** 旧 DP/in-house/SM hierarchical drivers、11/3 species
  复制版本、strain MDLM root driver 以及 `*_cls_wo_padding*`、`*_mean_wo_padding*`、`*_eval.py`
  feature 变体。canonical 替代入口是 `scripts/reproduce/run_hierarchical_mic.py`。
- **仍保留且不得当作重复版本删除：** `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_ChemBERTa_MLM.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_ChemBERTa_MTR.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MolFormer.py`、`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_PeptideCLM.py`。它们是 Fig. 2c 的不同 encoder comparator。
- 较早的 genome-only、text-only 与 genome+text 文件仍作为未完成核验的 modality ablation
  血缘暂留；在建立对应统一入口前不得删除。
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
- `capsule/code/reproduce_non_seen_strains_mdlm_mtr_fix.py` 重新评估最终 3-group × 7-ensemble strain-wise 结果；旧资源打包脚本已在 unified runner 完成后删除并由 legacy tag 保留。`prepare_zero_shot_antibiotic_classification_resources.py` 和 `reproduce_zero_shot_antibiotic_classification.py` 对 3-group × 10-ensemble strict zero-shot 结果执行同样操作。`prepare_fig2b_mic_regression_resources.py` 和 `reproduce_fig2b_mic_regression.py` 构建或运行 cached Fig. 2b 路径。`capsule/code/run` 在三种模式之间分派。
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
4. 高置信度 hierarchical MIC 和 strict zero-shot 分类路径已经完成统一入口；下一步把这些
   已验证入口纳入正式 quickstart，并继续迁移 sequence similarity。
5. 在声称完整复现之前，解决或明确归档 species/phylum、synergy rank、k-mer 和 Fig. 2b 指标不一致。
6. 清晰拆分外部项目：要么在许可证允许的前提下 vendoring 固定版本的 DLM/generation 代码，要么把它们声明为带版本的外部依赖。
7. 把历史副本、W&B 日志、notebook、旁支项目、巨型 checkpoint 和 reviewer capsule 移出源代码包；不要盲目删除 provenance，而应保留机器可读的 archive manifest。
8. 用最小且经过测试的安装说明、数据下载说明、实验配置、预期指标和容差、license/citation 以及清晰的支持矩阵，替换当前 `environment.yml` 和 `Readme.md`。
