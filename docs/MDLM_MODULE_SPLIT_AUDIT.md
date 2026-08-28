# DLM 预训练与 downstream MDLM 模块拆分审计

> 审计日期：2026-08-09
> 状态：历史模块拆分审计已完成。本文主体保留 2026-08-09 的决策与执行快照；截至 2026-08-10，
> DLM-Pretraining、downstream MDLM、Evo-2、Generation、Core 和 super-repo 均已完成独立发布与固定
> 上位计划：`REFACTOR_PLAN.md`、`docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md`

## 1. 结论

最终 super-repo 不使用一个含义模糊的 MDLM submodule，而是固定保留两个职责不同的模块：

| Super-repo 路径 | 目标 remote | 单一职责 | 当前来源 |
| --- | --- | --- | --- |
| `modules/dlm_pretrain/` | `DragonDescentZerotsu/ApexOracle-DLM-Pretraining` | DLM 与 209-descriptor MTR 联合预训练；产生 molecule encoder checkpoint | public legacy `ApexOracle/DLM_pretrain/` |
| `modules/mdlm/` | `DragonDescentZerotsu/ApexOracle-MDLM` | 加载 DLM checkpoint，提取 molecule embedding，训练 generation guidance 所需的 MIC/classifier heads，并执行候选打分/后处理 | 本地 `/data2/tianang/projects/mdlm` 的 clean source-only 子集 |

第一个模块使用 `DLM-Pretraining` 命名，是因为论文与现有 README 对基础模型使用 DLM 名称；第二个
保留已有公开 remote `ApexOracle-MDLM`，避免无必要地改变现有链接。二者都作为固定 commit 的 Git
submodule 接入，不把源码搬进 Core，也不合并彼此的环境。

## 2. 已由 Git 和源码验证的事实

### 2.1 预训练 producer 已经位于 reviewer-facing public repo

- `/data2/tianang/projects/ApexOracle_github/DLM_pretrain/README.md` 明确给出 molecular SELFIES 上
  DLM + multitask descriptor regression 的训练命令。
- `diffusion.py` 读取 `batch['descriptors']`，计算 `reg_mse`，训练目标为
  `loss + 0.1 * reg_mse`。
- `models/dit.py` 定义 209-output regression head；该代码家族与现有 joint DLM+MTR checkpoint 中
  的 `backbone.regression.*` 参数结构一致。
- `DLM_pretrain/` 是独立的 56-file、约 149 KB tracked source tree，而不是本地 downstream
  `/mdlm` 的简单副本。
- 当前 public legacy checkout HEAD 为 `2f29dee9cf6b7750425414f66c1a2d67998cb87f`；
  `DLM_pretrain/` 最近一次 path-changing commit 为
  `9032ee0289b378ec89ea578dfcdae21365917fe2`。这些 SHA 只记录当前公开来源，不等于已经证明与每个
  历史 checkpoint 精确一一对应。
- 当前 public legacy repo 的 `mdlm/` 与本地 `/mdlm` 是同一 downstream 工作树家族；它还跟踪
  1,099 个 `temp_data/` 文件，整个 tracked tree 约 222 MB，不适合作为新的 clean submodule 来源。

### 2.2 审计时的 public 预训练目录还不是 portable release

- `dataloader.py` 将 train/validation Arrow datasets 固定到合作者机器的
  `/data1/fangping2/hf_descriptor_1024_train_test_filtered_padded/`。
- tokenizer/generative-perplexity cache 和 descriptor statistics 也包含 `/data1/fangping/...` 绝对路径。
- README 示例使用 `model=small`（hidden size 768），但 `models/dit.py` 当前把 regression head
  hard-code 为 hidden size 1024；这与 `model=medium` 对齐，却与 README 示例不一致。
- 因此现有代码足以确认预训练目标和历史 producer 家族，但尚未验证从 fresh clone 按 README
  可以完成训练。发布前必须参数化路径、让 regression head 从 config 读取 hidden size，并做
  synthetic-batch smoke；不能把这一步写成已经完成。

### 2.3 本地 `/mdlm` 是 downstream 混合工作区，不是上述预训练 producer

- 重构前 Git 状态为 `master...origin/master [ahead 1, behind 6]`；`origin` 是上游
  `kuleshov-group/mdlm`，`custom` 是 public `DragonDescentZerotsu/ApexOracle-MDLM`。
- 本地 HEAD 与 `custom/master` 均为 `7a6a7d1a13ad10755fcb50962c3e022eda32bcbe`；这证明现有 public
  remote 保存了本地 tracked snapshot 的基线，不证明 dirty/untracked 工作已经发布。
- 重构前修改和未跟踪源码已冻结到 source-only commit `79eed10` 与 annotated tag
  `legacy-code-snapshot-2026-08-09`；审计后的 clean branch `refactor/apexoracle-mdlm` 已 push 到 `custom`，当时 HEAD
  为 `4c7cb63be57893646f620e8f9d2adab75c7dc3ee`，全仓 118 tests 与 14 项跨仓库 source contracts 通过。远端
  shallow-clone wheel/install/import/CLI、全部 tests、显式资产 20 checks 和 annotated recovery-tag fetch 通过，
  已可作为 super-repo submodule candidate；full Generation sampler parity 与 Core bridge 切换留在跨模块阶段。
  全程未推 `origin`。
- 工作区约 321 GB，主要由 `molecule_data/`、checkpoints、W&B、cache、outputs 和 wheel files
  构成；这些不是源码，全部不得进入 Git。
- 132 个 tracked 文件合计约 6.14 MB，实际问题是功能混杂、复制 driver、绝对路径和缺少稳定入口，而不是
  需要把整个 321 GB 目录迁移。

## 3. 本地 `/mdlm` 功能清单与发布去向

| 功能家族 | 代表文件 | 实际作用 | 最终处理 |
| --- | --- | --- | --- |
| 上游 MDLM runtime | `main.py`、`diffusion.py`、`dataloader.py`、`noise_schedule.py`、`models/`、`configs/` | Hydra train/eval/sample、masked diffusion、tokenizer/dataloader、DiT/DiMamba | 保留最小 checkpoint-load/embedding runtime；记录上游来源与许可；不宣称是 ApexOracle joint-pretraining canonical producer |
| Molecule embedding | `save_DBAASP_id_emb_dict.py`、`save_synergy_mol_id_emb_dict.py`、`save_inhouse_synergy_mol_id_emb_dict.py` | 从 DLM checkpoint 为 MIC/synergy 数据预计算 embedding | 整理成一个参数化 CLI，作为 downstream MDLM 的首要公开入口 |
| Fig. 2b molecule-only benchmark | `DBAASP_MLM_MDLM.py` | DLM representation + downstream regression benchmark | 历史 producer 留作 provenance；公共 benchmark canonical runner 继续位于 Core，downstream MDLM 只提供 encoder adapter |
| Hierarchical MIC legacy drivers | snapshot 中的 `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*.py` | strain/species/phylum holdout 的 11 套复制训练脚本 | 已完成逐文件 handoff、Core 43 tests 与 local-output inventory；root copies 删除，由 snapshot/lineage 恢复 |
| MIC guidance regressor | snapshot 中的 `guaidance_regressor_all_data*.py` | 六份 legacy source 已归并为五个 strain-conditioned clean/noisy/padding/non-pad profiles；正式 checkpoint 与 Generation regression parity 已验证 | `ApexOracle-MDLM` package/data contract/参数化 trainer；root copies 已由 snapshot 接管后删除 |
| Peptide/small-molecule guidance classifier | `guaidance_classifier_all_data*.py` | 训练 generation 的 molecule-type guidance head | v1 与 v2 数据/label 边界必须分开；只有能连接正式 checkpoint 的版本才是 canonical，其他标 experimental/provenance |
| Synergy guidance | snapshot 中的 `synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification*.py` | 三份 root source 已归并为 first-clean/second-noisy 与 both-clean 两个 all-data profiles；正式 encoder/candidate GPU parity 与两个 checkpoint schema 已验证 | `ApexOracle-MDLM` canonical model/prepared-data/training CLI；旧 copies 已删除，`v1.0` 只标 experimental；论文 synergy CV 仍由 Core 维护，不进入默认 quickstart |
| Candidate MIC/synergy scoring | `judge_generated_mols_MIC.py`、`judge_generated_mols_synergy.py`、`judge_mol_*_with_fig.py` | 加载 DLM 与 prediction head，对生成或给定结构打分并绘图 | 将无副作用的 scoring 抽为 CLI/library；绘图和历史文件扫描拆为 examples 或 provenance |
| Chemistry/post-processing | snapshot 中的四个 root utilities | SELFIES/SMILES/peptide 转换、catalog exact matching | peptide parser 继续优先复用 `PepLink==0.1.2`；table conversion/catalog matching 已迁为 supplier-neutral API，root copies 删除；正式 conversion byte-exact、588 万条 catalogue rescan semantic-exact |
| Hugging Face export | `huggingface/`、`huggingface_push.py` | tokenizer/model wrapper 与模型发布 | 待核对权重、model card、license 和现有 HF revision 后决定；不随本地大权重直接提交 |
| Interpretability/case studies | `visualize_attn*.py`、`temp_milk_embedding.py`、notebooks | attention 可视化与 milk/case-study 分析 | 默认不进入 canonical API；必要时放 `examples/legacy/` 或只在 source manifest 中追溯 |
| Debug/临时工具 | `debug*.py`、`temp_*.py`、`p_value_reference.py` | 一次性诊断、绘图或未纳入 Git 的新尝试 | 不从 dirty working tree 直接发布；逐文件审计后决定删除、归档或重写 |

Generation sampler、remasking schedule 和 sampling loop 继续属于
`modules/generation/`。Downstream MDLM 可以提供被 sampler 加载的 encoder/guidance heads，但不得
再复制一份 sampler，以免形成两个 generation producer。

## 4. 两个模块的 public contract

### 4.1 `ApexOracle-DLM-Pretraining`

必须公开：

- 模型结构、DLM objective、209-descriptor MTR objective 和 loss weight；
- tokenizer/model revision 与 descriptor schema、顺序、normalization statistics；
- 参数化的数据 schema/loader、small/medium configs 和 checkpoint manifest；
- `train` 命令、synthetic-batch CPU/CUDA smoke 和 checkpoint save/load test；
- 上游 MDLM attribution、license/NOTICE、训练数据可再分发边界。

不进入 Git：121.6M 条训练数据、Arrow cache、checkpoint、W&B、机器 cache 和任何绝对路径资产。

### 4.2 `ApexOracle-MDLM`

`v1.0` canonical 入口限定为：

1. DLM checkpoint + tokenizer 的固定输入 embedding smoke；
2. 参数化 molecule embedding CLI；
3. 经血缘确认的 MIC guidance regressor 与 peptide classifier trainer/loader；
4. 对结构列表或 generation output 的无副作用 MIC scoring CLI。

Core 已经接管的 hierarchical MIC/synergy CV 不在这里再维护第二份 canonical runner。Synergy
guidance、interpretability、milk case study 和旧图形脚本可以保留为 experimental/provenance，但
不能和上面四个入口并列成默认 quickstart。

## 5. 执行顺序与验收

1. 冻结 public `DLM_pretrain/`、本地 `/mdlm` tracked/modified/untracked、现有 checkpoint/HF
   revisions 的 source/asset manifest；外部 checkout 不 reset、不 clean。
2. 从 legacy public repo 的 `DLM_pretrain/` 建立 clean-history
   `ApexOracle-DLM-Pretraining`，先修复 config/path contract，再做 synthetic smoke。
3. 从本地 `/mdlm` 建立 source-only release branch，按上表先分类再收敛 canonical CLI；显式推送
   `custom`，不得 push `origin`。
4. 分别验证 pretraining checkpoint schema 与 downstream loader schema，再把两个 clean commit
   固定进 super-repo。
5. 根 quickstart 只消费 downstream MDLM；pretraining module 在 README 的“Train from scratch”路径
   单独出现，不成为预测/生成用户的强制安装项。

以下为 2026-08-09 的 downstream 执行快照：M0 source snapshot 已完成；M1 checkpoint/embedding I/O 在 `87fe50d`
完成，真实 567/568/1,079 个 filename mapping 均为 0 mismatch；M3a shared guidance heads 在
`136905c` 完成；M3b cross-repo contracts 在 `4521c53` 完成，累计 28 CPU tests 与 7 项跨仓库
source/AST checks passed。四个正式 generation/scoring checkpoints 已通过 schema validation，canonical
guidance/classifier heads 已完成真实 state-dict strict meta load；正式 noisy guidance bfloat16 GPU
head parity 的 genome/text/regression outputs 均 `torch.equal`、最大差异 `0.0`；仅 filename parser caller
已迁移。未删除任何 legacy source/asset；DLM encoder/full sampler、candidate scorer end-to-end parity、
在该快照时其余 caller、CLI 和 clean release smoke 仍待执行。详细契约位于 MDLM 的 `docs/CROSS_REPO_CONTRACTS.md` 与
`reproducibility/cross_repo_contracts.json`。

后续状态：上述验收已在 2026-08-10 的独立 release、fresh-clone 和 smoke gates 中完成；super-repo
`v0.2.0` 已固定五个模块。以下验收条件保留为当时的判定依据：两个 repo 独立 clone 后职责不重叠；
pretraining 可在小数据上完成一次 train/save/load；
downstream 可对固定 SELFIES 复现 embedding，并能加载 manifest 指定的 guidance checkpoint；任何命令
都不依赖作者机器绝对路径。

## 6. 推断与待确认事项

### 根据现有证据作出的判断

- public `DLM_pretrain/` 是合作者上传的 joint DLM+MTR 预训练代码家族；本地 `/mdlm` 的核心价值是
  downstream 使用与 guidance support，而不是重新代表预训练来源。
- 先拆清职责、再在各自 repo 内做最小清理，比把本地 `/mdlm` 原样发布或把两者合并更容易解释
  checkpoint provenance，也更符合 reviewer 对可复现性的预期。

### 审计时仍待作者/资产确认的事项

- 新预训练 remote 名称已在本计划固定为 `ApexOracle-DLM-Pretraining`；创建前仍需只读核验名称
  可用性，若不可用则必须由作者确认替代名，不能静默改名。
- public `DLM_pretrain/` 的哪个 commit 精确对应 12-layer 与 24-layer joint checkpoints；代码家族
  已匹配，但 commit-to-checkpoint lineage 尚未由保存的 launch snapshot 完全闭合。
- 哪个历史脚本精确产生 reviewer generation 使用的 v1 peptide classifier；现有同名 trainer 已有
  v2 数据语义，不能仅按文件名反向认定。
- Hugging Face 模型文件的 canonical revision、权重再分发许可和 model card 是否需要重发。
