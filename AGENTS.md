## AGENTS.md 维护语言

- 本项目的 `AGENTS.md` 应尽量使用中文记录和维护。
- 代码文件名、路径、命令、模型名称、指标缩写以及没有自然中文译名的专有名词可以保留英文。
- 新增审计结论时，应明确区分“已由代码和日志验证的事实”“根据现有证据作出的推断”和“仍待作者确认的事项”。
- 当前发布重构的阶段、已确认决策和验收标准记录在 `REFACTOR_PLAN.md`；执行过程中应同步更新其中的状态，避免计划与代码迁移脱节。
- 机器职责、环境、共享文件系统、数据/权重/外部仓库位置和当前 reviewer 任务记录在
  `docs/COMPUTE_AND_ASSET_MAP.md`。任何机器或资产迁移都必须同步更新该文件；node001 与
  node002 共享同一个 `/data1/tianang/Projects/Synergy_release`，不得把它们当作两个独立 checkout
  同时执行 Git 更新。

## 环境说明

- 默认使用 conda 的 `base` 环境。在机器 `sn4622119311` 上，conda 位于 `/home/tianang/anaconda3/bin/conda`。
- 不需要在沙箱中运行代码，因为沙箱可能阻止 GPU 访问。
- `/data2/tianang/projects/mdlm` 中的所有代码都必须在 `mdlm` conda 环境中运行。需要直接mdlm训练推理的代码可能都在那里面
- 之前还有很多实验是在node002上面做的，你可以找到对应的代码在node002的 /data1/tianang/Projects/Synergy。在node002上我们同样是使用的conda的base环境完成的Synergy的实验。node002的conda路径是：/data1/tianang/anaconda3/bin/conda
- /data2/tianang/projects/discrete-diffusion-guidance 里面是所有我们使用的generate peptide用的代码和仓库
- /data2/tianang/projects/evo2 里面是我们使用的embedding genome的代码
- **作者于 2026-07-19 确认的 guided generation 边界：** sampler 继续保留在外部
  `/data2/tianang/projects/discrete-diffusion-guidance`，与 DLM pretraining 和 Evo-2 producer 一样
  不复制进当前 Synergy 源码；未来 ApexOracle 总仓库只在外部仓库拥有 clean、固定 commit 后将其
  作为 submodule 或版本化链接。
- **作者于 2026-07-19 确认的边界：** 当前仓库不重构或重跑 Evo-2 genome embedding extraction，直接消费 `DataPrepare/Data/Genome_embs` 中的预计算 tensor。未来整合后的 ApexOracle 主仓库可以在 `external/evo2` 使用固定 clean commit 的 Git submodule；权重和 embedding 数据不进入 submodule。
- **已验证事实：** 当前 567 个 embedding 共 3,437,540,485 bytes，逐文件 SHA-256 manifest 位于 `experiments/evo2_genome_embeddings/file_manifest.csv`，其中三份论文数据匹配 563 个。全部已匹配 tensor 为 `torch.bfloat16`、hidden dimension 8192。重构后的只读 safe loader 完整重算 reviewer scaling CSV/PNG 后逐字节一致。当前外部 Evo-2 HEAD `afd0dae0a4bb25f3ca55f171fbdac4907b937afd` 的 commit object 存在，但 checkout dirty，且没有原始 extraction log 证明该 commit 是精确 producer，因此仅作为未来 submodule candidate。
- 论文最终绘图在 SSH host alias `Mac` 上维护；主 notebook 为 `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`。
- Mac 的 conda 位于 `/Users/kirianozan/Documents/anaconda/anaconda3/bin/conda`。后续论文绘图统一使用其 `base` 环境；已验证包含 Matplotlib 3.7.1、Seaborn 0.12.2、NumPy 1.24.3 和 nbformat 5.7.0。
- 通过非交互 SSH 生成图片时可使用 `MPLBACKEND=Agg .../conda run --no-capture-output -n base python ...`；在 notebook 中交互运行时继续使用 base kernel 即可。
- **作者于 2026-07-20 确认的绘图边界：** 不允许重写论文原始绘图 cell，也不允许代码自动覆盖或
  拼接论文总图 PDF。Fig. 1b 原 cell `220739609a526f79` 已从修改前备份逐字节恢复；reviewer
  sensitivity 位于新 cell `fig1b-reviewer-sensitivity-20260720`，输出独立 panel 文件。后续完整
  ensemble 图继续使用新的 cell，论文总图编辑由作者完成。
- Fig. 1b Chemprop 最终重跑环境分别为本机
  `/data2/tianang/projects/.venvs/fig1b-chemprop-v1` 和 node002
  `/data1/tianang/Projects/.venvs/fig1b-chemprop-v1`。两者核心版本均已核验为 Chemprop 1.5.2、
  Torch 2.7.1+cu126、NumPy 1.26.4、pandas 2.2.2、scikit-learn 1.8.0 和 RDKit 2025.03.5。
- Fig. 1b 补实验实时状态统一使用
  `python scripts/reproduce/monitor_fig1b_revision.py`；需要连续刷新时使用
  `watch -n 30 python scripts/reproduce/monitor_fig1b_revision.py`。该入口只读本机和 node002
  的日志/产物与 GPU 状态，不修改 checkpoint 或原始数据。
- **2026-07-20 node001 扩容事实：** node001 与 node002 共同挂载 `bright91:/data1`；release
  driver 的 inode/size/hash 一致，8 张 A100 80GB 和共享 base/Chemprop 环境已通过 CUDA import。
  node002 各队列的第 3 个待补 member 已各分配一个到 node001；训练进入 steady state 后利用率
  86--100%、温度 56--68°C。每个 node001 worker 随后再运行一个完整 baseline fold；node002
  baseline 队列以 `metrics.json` 为完成条件跳过共享文件系统中已完成的 fold。监控只从 `/data1`
  汇总一次任务进度，但分别显示 node001/node002 GPU，避免共享产物被重复计数。
- **2026-07-21 Fig. 1b 负载重排事实：** Apex fine-tune 已达到 `150/150`，本轮 `45/45` 缺失
  member 均已完成，15 个 fold 的最终 10-member prediction 也已组装。RN4220/Wong baseline
  folds 3/4 已完成并固定使用前 10 members；folds 0/1/2 当前分别在 node001 GPU0/1/2 运行。
  node002 后启动且会并发写共享 fold 3/4 的重复 worker 已停止。
  本机 Chemprop venv 已按 node producer commit 补齐 `descriptastorus 2.7.0.3`。作者随后决定
  三个 baseline 均与 ApexOracle 对齐为 10-member ensemble；RN4220 fold 0/1/2 已按 10 members
  重启，fold 3/4 和 E. coli 的既有 20-member 目录只选择 `model_0`--`model_9` 做最终 prediction。
  历史 checkpoint 分散在两台机器，推理必须在实际拥有对应文件的机器运行；不要重新启动已停止的
  duplicate baseline session。
- **2026-07-22 Fig. 1b 最终补实验事实：** 三个 baseline 的 `15/15` folds 已完成，每折最终
  固定消费 10 members。fold-mean AUPRC/AUROC 为 E. coli `0.54535/0.85932`、A. baumannii
  no-RDKit `0.30355/0.77589`、RN4220 `0.36616/0.94337`。共同样本上 5,000 次 paired
  bootstrap/prediction-swap 显示 fine-tune 的 6/6 指标均高于 baseline 且 Holm-adjusted
  `p < 0.05`。strict zero-shot 只有 E. coli AUROC 显著更高；A. baumannii AUROC 和 RN4220
  两项显著更低。最终统计 SHA-256 为
  `12c1d603b29679ee5f9b1bd8fcd5a60a78b5beb622fe593ca8fd6e7003577ae0`。
- **2026-07-22 Fig. 1b 最终绘图事实：** Mac canonical notebook 已新增独立 cell
  `fig1b-final-10member-dual-metric-20260722`，左 AUPRC、右 AUROC；每个柱为五折均值，error bar
  为 sample s.d.，bracket 为 exact Holm-adjusted paired prediction-swap `p`。输入冻结在
  `experiments/fig1b_antibiotic_classification/final_10member_dual_metric.csv`；新输出为
  `3-strain-antibiotics-final-10member-dual-metric.{pdf,png}`。原 cell `220739609a526f79` 的源码和
  output 语义 hash 均未变化，旧 panel 和论文总图均未覆盖。
- **2026-07-22 Fig. 1b AUPRC-only 绘图事实：** 作者新增的 AUPRC-only placeholder 已填写为
  cell `fig1b-final-10member-auprc-only-20260722`，独立输出
  `3-strain-antibiotics-final-10member-auprc-only.{pdf,png}`。柱为五折 mean AUPRC，error bar 为
  sample s.d.，bracket 为 pooled sample-level paired test 的 Holm-adjusted `p`。原始与双指标
  cell 均经 hash 确认未变化；论文总图未自动覆盖。最终英文回复及修改清单位于
  `experiments/fig1b_antibiotic_classification/reviewer_response_auprc_final.md`。
- **2026-07-22 AUPRC-only layout 修订事实：** 已按作者反馈压缩 legend 与坐标区留白，并将
  PDF 页面精确匹配旧论文 panel 的 `741.12 × 380.724 pt`；仅 AUPRC-only cell 发生变化。
- **2026-07-22 Fig. 1b 文稿最终边界：** `Methods / Data / Small Molecule Antibiotics` 保持纯数据
  描述；作者删除了先前迁入 `Implementation Details` 的细碎 Fig. 1b protocol，因此 Methods 不再
  新增该实验的实现细节。TeX 只更新结果概览、Fig. 1b 图注和 Results；实际 response-letter docx
  已换成最终 10-member AUPRC 数值，并保留指定的 Stokes/Liu/Wong common-fold 说明。TeX 临时
  编译为 28 页、Word 临时转换为 25 页；未覆盖论文 PDF、figure 或论文总图。

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
- Fig. 2b 的两个 DLM 本地 checkpoint、APEX checkpoint 和四个 Hugging Face 模型均已进入
  manifest。ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM 的复现 revision 均已
  固定；ChemBERTa-MTR、ChemBERTa-MLM 和 PeptideCLM 的原始 2025 run 没有记录精确 upstream
  commit，因此状态明确写为“固定复现锚点，旧 run revision 未记录”。

## 论文及审稿回复路径

- 本项目对应论文：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`
- 审稿意见及回复草稿：`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Response to reviewers letter.docx`
- **2026-07-18 已完成的 Fig. 2b 修订：** 回复信中关于各 encoder 是否使用相同数据、五折不确定性和原 27.1\% 表述的回答已经改为已完成实验及正式数值；论文 Fig. 2b 图注、Results 和 Methods 已同步修改。作者随后更新了完整 `Fig2_2.pdf`；经实际渲染核验，panel b 现为 10,886 个共享分子的七模型结果，显示五折 sample s.d. error bars，柱上三位小数与正式结果一致。最新 TeX 已再次完整编译为 28 页。
- **当前正式结果：** 文稿、回复信和图片使用 24-layer joint DLM `0.5386 ± 0.0250` 与 12-layer DLM-only `0.3765 ± 0.0239`，相对第二名提升表述为 29.1\%。正文当前仍把优势解释为 joint DLM+MTR objective，但没有明确写出两个 DLM checkpoint 的容量不同；因此“objective 导致提升”仍应视为尚未完成容量控制核验的解释，而不是现有 benchmark 已证明的事实。
- **仍待实验核验的事项：** 如果后续采用 12-layer joint 候选作为主比较，必须再次同步 Fig. 2b、Results、回复信结果段和 29.1\% 相对提升；在该实验完成前，当前 24-layer joint 正式结果和图片保持不变。

## 2026-07-20 Reviewer 4 unseen-species 初筛

### 已由代码、私有表格 hash 和 producer 过滤路径验证的事实

- Mac 的私有 in-house AMP 表格已复制到被 Git 忽略的
  `DataPrepare/Data/private_inhouse_amp/Master_List_Peptides_Antimicrobial_Activity.xlsx`；Mac 与
  本机 SHA-256 均为 `956ff3d60364a113c2149a63b74b65d4f81ec03741fefbac1841558ed54744a4`。
  原始 Excel、派生逐表头 CSV 和执行后的 notebook 均不得上传 GitHub。
- `guaidance_regressor_all_data_pad_no_mask.py` 实际消费的不是源 CSV 中全部 5,614 个 strain 名，
  而是经过 genome/text embedding 过滤后的 1,599 个标准化 strain ID；它们全部可回映到 389 个
  producer-era species 名。审计入口为
  `scripts/audit/audit_reviewer4_inhouse_species_coverage.py`。
- Excel 两个 sheet 共 76 个 assay 表头，归一化为 35 个 species。结合项目 frozen taxonomy alias
  与 2026-07-20 NCBI Datasets v2 名称核对后，11 个 species 没有进入 guidance regressor 的实际
  训练 exposure，其中 10 个已有 MIC 数值：`[Clostridium] scindens` 468、
  `Collinsella aerofaciens` 404、`Akkermansia muciniphila` 363、
  `Parabacteroides distasonis` 271、`Bacteroides uniformis` 230、`Segatella copri` 161、
  `Bacteroides eggerthii` 143、`Bacteroides ovatus` 99、`Agathobacter rectalis` 83、
  `[Clostridium] symbiosum` 53；`Staphylococcus capitis` 只有表头、没有测量值。
- 上述 11 个候选当前均不具备 generation 所需的完整 exact-target genome+text embedding 组合；
  其中 9 个表头提供了 exact strain accession，但现存两个 embedding 目录对这些 exact target 均为
  0 个匹配，另两个表头没有给出 exact strain ID。

### 根据现有证据作出的推断

- 当前表格可以支持“存在 species-level 未见候选”的数据结论，但不能单独支持 reviewer 要求的
  broad efficacy：每个未见 species 目前最多只有一个有数据的 strain，不能把单 strain 活性写成
  对整个 species 或 genus 的广谱有效性。
- 现有高覆盖候选多为肠道 commensal/opportunist；仅按 assay 数排序不能建立临床合理性。临床 target
  应由作者和 microbiology 团队另行确认，并配套多个独立 isolate。

### 仍待作者确认或新增资产的事项

- 从 10 个有 MIC 的 unseen species 中选择临床 target，并确定至少一个多-isolate 的 species/genus
  panel；如果没有现成 panel，应先补菌株而不是直接把单列结果写入 reviewer response。
- 目标确定后，Evo-2 genome embedding 与 strain-text embedding 仍由外部 producer 生成并登记
  exact genome assembly、text source、SHA-256 和 producer commit；在这些资产就绪前不启动 sampler。

## 2026-07-21 Reviewer 4 可购买 unseen 靶点初筛

审计入口 `scripts/audit/audit_reviewer4_unseen_atcc_pathogens.py`，输出与完整结论位于
`experiments/reviewer4_unseen_targets/`。该初筛回答的是「要做 broad-efficacy 实验应该买什么」，
与 in-house workbook 初筛（「已有 MIC 的未见 species 有哪些」）互补。

### 已由代码和公开数据源验证的事实

- 训练暴露沿用同一冻结过滤：1,599 个 strain ID、389 个 producer-era species 名、425 个 canonical 名。
- 425 个训练名全部由 NCBI Taxonomy 解析成功，归一到当前 165 个 genus；unseen 判定在当前名上
  进行。这一步会改变结论：`Ochrobactrum anthropi` 已并入 `Brucella`，而 `Brucella melitensis`
  和 `Brucella abortus` 在训练集中，因此它被正确判为 species-only 而非未见 genus。
- 81 个策展候选中，62 个为 genus-level unseen，19 个为 species-only。
- ATCC 目录经站点公开搜索索引读取；产品模板名取自实时 facet，细菌为
  `Bacteria and Bacteriophages`、真菌为 `Mycology`、原生动物为 `Protistology`。用错模板名会让
  所有真核候选静默返回 0 个产品。
- 以「在线可订购 + 已有 ATCC Genome Portal assembly」为门槛，18 个 genus-level unseen species
  有 >= 3 个可用 isolate。前三名为 *Morganella morganii*（24）、*Pantoea agglomerans*（15）和
  *Providencia stuartii*（11）。*Providencia* 三个 species 全部未见，合计 21 个可用 isolate。
- 抽查 ATCC 33672 与 ATCC 49042 的实时产品页，均可访问且 Genome Portal genome ID 与索引一致。

### 根据现有证据作出的推断

- 未见 genus 不等于未见 family，回复中必须主动披露。*Providencia* 与 *Morganella* 同属
  Morganellaceae，而三个 *Proteus* species 在训练集中。family 也完全未见的候选只有
  *Pantoea*、*Erysipelothrix*、*Brevundimonas*、*Leptospira* 和 *Treponema*。
- Morganellaceae 候选对 polymyxin 天然耐药（lipid A L-Ara4N），临床需求叙事更强但 AMP 起效
  概率更低；这是从已知耐药机制推断的，不是本仓库数据证明的。

### 仍待作者确认的事项

- 选定靶点，并决定走 genus-level 主张（*Providencia*）还是 family 也未见的单 species 主张。
- 逐条核对实时产品页的 BSL、运输限制与机构生物安全审批；厌氧、fastidious 和螺旋体候选的 MIC
  方案与常规 CLSI 肉汤稀释不同，需先确认可行性。
- 本初筛只证明「买得到且有基因组」。模型在这些靶点上的预测性能尚未评估，也没有任何湿实验数据；
  不得在回复信中把候选清单表述为已完成的 broad-efficacy 结果。

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
- 被统一 runner 替代的 root DP/in-house/SM/pooling/eval 脚本、capsule 中第二份 strain
  driver 和旧打包脚本已删除；完整恢复点为 `legacy-code-snapshot-2026-07-17`。Fig. 2c 的四个
  online encoder comparator 也已迁入同一 runner 的显式 profiles 并删除 root 复制 driver；
  modality ablation 的 15 个无消费者/会覆盖数据的 legacy driver 也已独立审计并归档删除。

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
| Fig. 1a / Fig. 2f phylum-wise holdout | `scripts/reproduce/run_hierarchical_mic.py --protocol phylum`；checkpoint 位于 `.../3_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles` | **代码路径已统一并验证；历史结果仍为中等置信度。** node002 的完整 MDLM 候选为 Fungi `0.3034`、Pseudomonadati `0.4152`、Bacillati `0.4346`，均值 `0.3844`，比论文 `0.3744` 高 `0.0100`。作者决定暂不继续追查精确论文运行。 |
| Fig. 1a / Fig. 2d、g species-wise 11-cluster holdout | `scripts/reproduce/run_hierarchical_mic.py --protocol species`；checkpoint 位于 `.../11_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles` | **代码路径已统一并验证；历史结果仍为中等置信度。** 合并 node002 group 0–5 和本机 group 6–10 后已覆盖全部 11 组；10 个非异常值重算均值 `0.43366`，与论文 `0.4337` 一致。唯一差异是现存异常组 `-0.3467`，而 Mac 绘图为 `-0.1467`。作者决定暂不继续追查。 |
| Fig. 2c strain-wise molecular encoder 比较 | `scripts/reproduce/run_hierarchical_mic.py --protocol strain --molecule-encoder {chemberta_mtr,chemberta_mlm,molformer,peptideclm}`；配置 `configs/hierarchical_mic/legacy_fig2c_comparators.yaml` | **代码路径已统一并通过真实模型 H100 smoke。** profiles 保留各 encoder 的 online tokenization、first-token pooling、mode、freeze epoch、ensemble 数和 optimizer 差异。MTR/MolFormer/PeptideCLM 现存 checkpoint 已严格匹配固定 revision；node002 MLM checkpoint 的 state schema 精确匹配。PeptideCLM fixed 与 node002 早期 7-member 变体不同，论文精确 ensemble 血缘仍未完全恢复。 |
| Fig. 2c Evo-2 与 k-mer 消融 | canonical producer 为 `scripts/reproduce/build_kmer_embeddings.py`；consumer 使用共享 hierarchical runner 和 `configs/hierarchical_mic/legacy_kmer_reconstruction.yaml`；血缘见 `experiments/kmer_ablation/` | **代码迁移完成；论文精确训练血缘仍缺失。** Mac 最终图的单模型 k-mer 为 R²/Spearman/Pearson `0.4507/0.6688/0.6793`。`/data/fangping/kmer_baseline` 是 2026 年 post-paper reconstruction：完整 global/25-epoch/7-member 三组均值 R² `0.5276`，协议与论文不同。567 个 global 和 567 个 windowed tensor 已建立逐文件 manifest；两种 canonical producer 在真实 E. coli ATCC 25922 上均与现存 tensor 逐值相同。 |
| Fig. 2b 不使用 strain knowledge 的五折 molecular representation benchmark | canonical 模块为 `src/apexoracle/benchmarks/molecule_encoders/`，正式结果见 `experiments/fig2b_molecule_encoders/results_shared_5fold.md`；DLM 预训练代码仍位于外部 `/data2/tianang/projects/mdlm` | **正式修订版 / 高置信度。** 7-model × 5-fold 已在 10,886 个共享 molecule 上完成；旧 root/capsule source 副本由 legacy tag 和 migration audit 追溯。 |
| Fig. 1b 严格 target-strain zero-shot 小分子分类 | canonical 入口 `scripts/reproduce/run_antibiotic_classification.py --mode strict-zero-shot`；checkpoint：`.../antibiotic_3_strain_compare/MDLM_fix_cls_sm_all_test_10_fold_ensembles` | **最终版 / 高置信度，已完成行为保持迁移。** 完整 held-out target-strain 数据不进入训练，但仍逐 epoch 用于 best-AUROC checkpoint selection。完整 ensemble 指标：E. coli `#004` 为 0.9360 AUROC / 0.5890 AUPRC；A. baumannii 17978 为 0.7262 / 0.3243；S. aureus RN4220 为 0.7679 / 0.1655。30 个 checkpoint 和 3 个完成日志网格完整；group 0 / ensemble 0 已在 H100 上与 capsule 做到 2,335 条 logit 逐值一致。 |
| Fig. 1b fine-tuned ApexOracle | 同一入口的 `--mode fine-tune --fold N`；checkpoint：`MDLM_fix_cls_10_fold_ensembles` | **完整 10-member ensemble / 最终补实验完成。** fine-tune 网格为 `3 strains × 5 outer folds × 10 members = 150 checkpoints`；strict zero-shot 是无 KFold 的 `3 × 10 = 30`。15 个 fold 已逐项验证恰好 10 members。fold-mean AUPRC/AUROC 为 E. coli `0.71205/0.95884`、A. baumannii `0.43436/0.82200`、RN4220 `0.40127/0.95309`；与三个最终 10-member baseline 的 6/6 paired comparisons 均为 Holm-adjusted `p < 0.05`。单成员 pooled OOF 只保留为 sensitivity。`--mode molecule-only` 对应旧 `wo_SAND`。 |
| Synergy 二分类结果 | canonical 入口 `scripts/reproduce/run_synergy_cv.py`；配置 `configs/synergy/legacy_cv.yaml`；checkpoint：`.../strain_wise_synergy/MDLM_3_fold_ensembles_1_base_model_cls` | **论文高置信度复现候选 / 重构完成。** 使用 `synergistic_pairs_Evo.csv`、strain-wise 三折、单个完整 MIC base 和 7-member ensemble。现存 mean AUROC/AUPRC `0.7598/0.7440` 与论文 `0.7539/0.7454` 的绝对差为 `0.0059/0.0014`；作者确认接受为论文实现的高置信度复现候选。它不是精确原始 checkpoint 声明；rank 1024/100-epoch base 与 Methods rank 64/13 epochs 的差异继续披露。root driver 已归档删除。 |
| Synergy 候选使用的完整数据 MIC base model | 外部 MDLM producer `/data2/tianang/projects/mdlm/guaidance_regressor_all_data_pad_no_mask.py`；checkpoint `Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth` | 这是现存三折候选实际加载的最匹配 base model，但 Methods 写 13 epochs，而该文件来自 100-epoch run，精确论文身份仍未解决。历史 `train_on_all_data.py` 不是该权重的 producer，已按 SHA 归档删除。 |
| Guided molecule generation、remasking sampler 和 256-step 三阶段 guidance | 不在当前仓库；分析脚本指向 `/data2/tianang/projects/discrete-diffusion-guidance` | **外部依赖 / 缺失。** 当前仓库包含 predictor、保存的输出或图片以及 similarity 分析，但不包含实际生成 ApexOracle-3/12/23 的 sampler。论文参数为 256 steps、MIC target 1、sigma 从 0.5 线性降到 0.2、`t_on=0.55`、`t_off=0.45`，两个阶段的 guidance strength 为 15。 |
| Fig. 3（TeX 中使用文件 `Fig4.pdf`）guided/unguided 预测和最终验证分子 | `DataPrepare/Morgan_fingerprint_sim_generation*.py` 中的派生分析和外部 guidance 仓库 | **复现链仍不完整。** 实际 guided sampler、候选生成和湿实验选择链尚未形成自包含流程；论文后 BAA-3170 prospective synergy screening 不是该论文结果，已按 paper-only 决策移出当前工作树。 |
| 附录 modality ablation | 最终绘图值：`experiments/modality_ablation/paper_values.csv`；绘图入口：`scripts/reproduce/plot_modality_ablation.py`；候选血缘审计：`experiments/modality_ablation/candidate_lineage.json` | **论文图可复现 / 代码收尾完成。** Mac 最终 notebook 的 12 个 R² 已逐项冻结；现存 checkpoint 仍无法精确连接到 ensemble 数值，因此不支持训练重跑。15 个会原地覆盖中间 CSV 的 legacy driver 已按哈希归档删除，原始数据未修改。 |
| Attention 或耐药基因解释 | `DataPrepare/ATCC_genome_annotation_get.py`、`DataPrepare/resistant_gene_check.py`、`DataPrepare/train_genome_mcr_check.py`，以及大型训练脚本中的 attention 输出 | 属于探索或审稿支持代码；不存在自包含的最终 attention figure 流程。 |
| ApexOracle-3/12/23 sequence similarity 表 | `scripts/reproduce/run_sequence_similarity.py`；`src/apexoracle/evaluation/sequence_similarity/` | **canonical / 已验证。** 实现 Methods 中的 Biopython global alignment、BLOSUM62、gap-open 10、gap-extension 0.5、exact-match PID 和 cyclic exhaustive rotations。ApexOracle-3/23 全量输出与历史 CSV 逐字节一致；ApexOracle-12 的论文数值已复算，但旧 full CSV 未保存且 top hit 有四个 complete ties。 |
| 审稿回复中的 Evo-2 embedding 缩放说明 | `scripts/plot_evo2_genome_embedding_abs_mean_distribution.py` | **审稿阶段 / 高置信度。** 生成支持固定 `1e14` 缩放因子的 CSV、PNG 和 PDF，统计范围为 563 个实际匹配的 embedding。 |

### 主要训练脚本的版本家族

#### 分层 MIC 回归

- 早期 `MIC_with_genome*`、`MIC_with_text_test*` 和 `MIC_with_text_genome*` 共 15 个文件属于
  genome-only/text-only/genome+text modality ablation 候选或更早 prototype；它们已经按
  `experiments/modality_ablation/legacy_cleanup.json` 归档删除，只在 legacy tag 中保留。
- `DP_inhouse_MIC_with_text_genome_test_on_non_seen_*.py` 和同模型的
  `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*.py` 曾构成论文时期的复制脚本家族；
  其 hierarchical MIC 版本现已由统一 runner 替代并从工作树删除，只在 legacy tag 中保留。
- 11-cluster、3-cluster 与 strain-wise 的最终 MDLM 路径现在都应使用
  `scripts/reproduce/run_hierarchical_mic.py`。旧 `..._MDLM_cls_fix.py`、
  `..._MDLM_MTR_fix.py`、pooling 和预计算 feature 变体仅作为历史血缘记录，不再是活跃入口。
- Fig. 2c 的 ChemBERTa-MLM、ChemBERTa-MTR、MolFormer 和 PeptideCLM root drivers 已由
  `legacy_fig2c_comparators.yaml` 的四个 profiles 替代；原文件只在
  `legacy-code-snapshot-2026-07-17` 中保留。

#### Molecule-only Fig. 2b benchmark

- `fine_tune_on_DBAASP_SMILES_5_fold_mean_MIC.py`、三个 `*_compare*`、all-data 和 in-house
  ChemBERTa driver 均为历史版本，已于 2026-07-19 从工作树删除；由
  `legacy-code-snapshot-2026-07-17` tag 恢复。
- ChemBERTa-MTR、ChemBERTa-MLM、MolFormer、PeptideCLM 和 APEX 统一使用
  `scripts/reproduce_fig2b_baselines_online_5fold.py`。五个 root `fix_*` 只通过
  `legacy-code-snapshot-2026-07-17` 和清理 manifest 追溯；未汇报的 mean-pooling 变体不再位于发布入口。
- DLM/MTR benchmark 源代码位于外部 `mdlm` 项目；当前仓库只维护 canonical thin runner，
  不再保留 capsule 内第二份外部原始 driver。
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
- **已确认的 APEX 兼容规则：** noncanonical residue 在输入字符串中写为 `X`，cyclic peptide 使用线性 residue 顺序；但 APEX 原始 23-token vocabulary、AAindex embedding、encoder、checkpoint 和 `512→256` regression head 均不得修改。原 APEX 没有 `X` token，因此 `X` 必须按 `apex_adapter.py::legacy_onehot_encoding` 冻结的历史行为保留在 index 0，不能新增 index 23 或平均 AAindex embedding。
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
- 历史 `compare_APEX` 的源码已由 legacy tag 追溯。实际消费的 APEX encoder、AAindex loader、
  token adapter、masked loss 和 R² 已迁入 `src/apexoracle/benchmarks/molecule_encoders/`；最佳
  checkpoint 已迁至 `${APEXORACLE_WEIGHTS_DIR:-weights}/molecule_encoders/apex/` 并由 manifest
  ID 解析，AAindex 位于 ignored `resources/reference/apex/`。其余历史 checkpoint 和 265 个
  W&B 文件完整迁到 `/data2/tianang/projects/ApexOracle_legacy_assets/compare_APEX_archive_2026-07-20`，
  没有删除。

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
- **2026-07-20 reviewer 修订运行：** strict zero-shot 的 30 个 checkpoint 已全部完成确定性
  H100 推理并导出样本级预测；三个目标的 AUROC/AUPRC 分别为
  `0.93504/0.58738`、`0.72408/0.32098`、`0.76741/0.16562`。统一 Chemprop baseline 的
  15 个 fold 已全部完成，pooled OOF AUROC/AUPRC 为 E. coli `0.85711/0.50752`、
  A. baumannii `0.77750/0.31967`、RN4220 `0.92848/0.32873`。
- **已由 5,000 次配对统计验证的事实：** strict zero-shot 相对 baseline 的 Holm 校正结果为：
  E. coli AUPRC 差 `+0.07986`、`p=0.4167`，AUROC 差 `+0.07791`、`p=0.01360`；
  A. baumannii AUPRC 差 `+0.00132`、`p=0.9660`，AUROC 差 `-0.05342`、`p=0.03059`；
  RN4220 AUPRC/AUROC 差 `-0.16311/-0.16107`，两者 `p=0.00060`。因此旧的普遍优势表述
  不成立，修订稿必须按菌株和指标分别陈述。
- **baseline 血缘纠正：** 当前 Fig. 1b 对 A. baumannii 使用的 `0.756/0.266` 是 Liu 2023
  的 no-RDKit ablation；该论文 RDKit 主模型约为 `0.792/0.337`。ApexOracle 自身没有 RDKit
  feature augmentation；作者确认修订比较继续采用 no-RDKit
  ablation，以保持 feature 条件一致，不采用 RDKit 增强的最强版本。
- **已由 GPU telemetry 验证的事实：** 2026-07-20 16:55 尝试把本机 GPU1 加入 Fig. 1b 补训；
  两分钟内 GPU/显存分别升到 90°C/88°C，并出现 hardware/software thermal slowdown。当前账号
  无权下调 350W power limit，因此 worker 在首个 epoch 前停止，GPU1 不再参与本轮实验；对应
  node002 队列保持不变。完整 30 秒采样记录位于 ignored runtime 文件
  `results/fig1b_revision/gpu1_thermal.csv`。
- **Chemprop eligibility 边界：** E. coli 的 `ce_2244` 与 RN4220 的 `na_20640` 是同一条
  RDKit 拒绝的异常铝配位结构；既有 KFold 不重排，配对比较时从双方同时排除并登记。
- **fine-tune sensitivity 已完成：** 为避免混用残缺 ensemble，每个 outer fold 固定只使用
  `ensemble_0`。14/15 个 fold 复用历史 checkpoint；RN4220 fold 4 以
  `PYTHONHASHSEED=0`、ensemble seed 42 按旧 25-epoch 协议补训。pooled OOF AUPRC/AUROC
  分别为 E. coli `0.66655/0.95529`、A. baumannii `0.35294/0.77698`、RN4220
  `0.34518/0.92278`。相对 common-fold baseline，Holm 校正后只有 E. coli AUPRC
  (`p=0.03539`) 和 AUROC (`p=0.00180`) 显著更高；其余四项均不显著。该结果是
  单模型/折 sensitivity，不是旧完整 ensemble 的恢复。
- **2026-07-20 指标口径诊断：** 当前 fine-tune sensitivity 不是旧论文柱子的等价重算。
  旧图使用历史 multi-member ensemble 的 fold-level 汇总并硬编码混合指标；当前使用每折固定
  `ensemble_0` 的 pooled OOF。RN4220 旧 AUPRC `0.408` 与现存前三折 10-member ensemble
  均值 `0.40730` 一致；A. baumannii 仅恢复两折，均值 `0.45595`，不能验证旧图 `0.4344`。
  因此当前 `0.34518/0.35294` 的下降应标为协议口径差异，不是已证实的重构行为漂移。
- **Fig. 1b 绘图入口：** canonical 实现为 `src/apexoracle/evaluation/fig1b_plot.py`，Mac
  canonical notebook 为 `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`
  的 cell `220739609a526f79`。另一份
  `/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/figs.ipynb` 是三 cell 的旧副本，
  从未包含当前 Fig. 1b cell。图上使用 Fig. 3a 风格 bracket，显示 paired prediction-swap
  test 的 Holm-adjusted p。旧 `1 model/fold` sensitivity 只作诊断；最终 AUPRC-only cell 使用
  每折 10-member ensemble。
- **Fig. 1b 文稿同步已完成：** Mac canonical notebook、三菌株统一 AUPRC panel、论文 Fig. 1
  caption、Results/概览和 reviewer response 已同步；Methods 按作者决定不加入详细 protocol。
  最终 TeX 完整编译为 28 页，response letter 临时转换为 25 页。修改前
  快照与前后 SHA-256 记录在
  `reproducibility/fig1b_reviewer_revision_2026-07-20.json`。

#### node002 非破坏性同步（2026-07-20）

- 旧目录 `/data1/tianang/Projects/Synergy` 保持原位且未修改；源码/config 936 个文件已做
  SHA-256，11,340 个非源码资产只登记路径、大小和 mtime。snapshot 位于
  `/data1/tianang/Projects/_legacy_manifests/Synergy_legacy_2026-07-20`，扫描错误为 0。
- 重构代码运行副本为 `/data1/tianang/Projects/Synergy_release`；reviewer 补实验输出统一写到
  `/data1/tianang/Projects/ApexOracle_revision_runs`，不得写回旧 checkpoint/data 目录。

#### Synergy 与后续 in-house 实验

- **作者确认的发布边界：** 当前仓库只保留论文汇报的 2,732-pair、strain-wise 三折、每折
  7-member ensemble 二分类路径。论文没有汇报 few-shot 结果。
- 早期 ChemBERTa/prototype、continuous-FICI、all-data、in-house evaluation、few-shot、post-paper
  guidance/regression/screening driver 及其无共享消费者的 canonical 模块已于 2026-07-19 删除。
  删除只作用于 Git 跟踪的源码、配置、专项测试和派生审计；原始数据、checkpoint、日志和结果
  未修改。逐文件哈希与恢复点见
  `reproducibility/synergy_paper_only_cleanup_2026-07-19.json`。
- 作者确认将本机完整 CV 结果作为论文高置信度复现候选后，最后一个 root classification driver
  已登记 SHA-256 并删除；canonical runner 是唯一发布入口，旧文件由 legacy tag 恢复。

#### Synergy CV 重构阶段审计（2026-07-19）

- **已由代码、数据、checkpoint 和日志验证的事实：** 候选 classification driver 的动态过滤得到 2,732 行 eligible pair；`PYTHONHASHSEED=0` 时三个 fold 的四路过滤前/后行数均逐项匹配旧日志。现存 21 个 checkpoint 构成完整 `3×7` 网格且结构一致，active fusion LoRA rank 为 1024，synergy head 为完整参数训练的 `24576→3072→128→1`，代码里构造的 rank-256 head LoRA config 未实际使用。候选 driver 加载的是 100-epoch base checkpoint，不是仓库中另存的 13-epoch checkpoint。
- **作者确认的决定：** 该 family 作为论文实现的高置信度复现候选，synergy 重构阶段完成。三个 fold 日志的未加权均值 `0.7598/0.7440` 与论文 `0.7539/0.7454` 的绝对差为 `0.0059/0.0014`；不再为追求逐文件相同的历史 checkpoint 阻塞发布。
- **已由 node002 只读核验的事实：** `node002:/data1/tianang/Projects/Synergy` 只有六个较早的
  ChemBERTa/Evo synergy prototype；不存在新的 MDLM one-base classification driver、
  `strain_wise_synergy` 目录或 `MDLM_3_fold_ensembles_1_base_model_cls` checkpoint family。
- **仍须披露但不阻塞的限制：** Methods 写 fusion LoRA rank 64 和 base training 13 epochs，并把融合维度写为 `12,294→3,073`；候选 checkpoint 分别证明 rank 1024、实际加载 100-epoch base 和真实维度 `12,288→3,072`。旧日志也未记录独立进程的 `PYTHONHASHSEED`，因此不能声称逐 bit 或精确原始 run 复现。
- **已完成的重构验收：** 1 个 base 与 21 个 member 的逐文件 SHA-256 已登记；`fold_0/ensemble_0` 在 H100 上严格加载后，genome+text 和 text-only 两路均与 inline legacy 公式逐值一致。统一 runner 的 fold 2、1 member、1 epoch 真实数据 smoke 成功写出 checkpoint、175 条预测、metrics 和 summary；临时 2.24 GB 输出已删除。该 smoke 指标不作为论文结果。
- **已由范围审计验证的事实：** 论文正文和 Methods 不包含 few-shot synergy 结果；paper-only
  清理后，共享 synergy 模块仅保留 CV 路径的消费者。全仓库为 111 passed / 4 skipped；
  post-paper 源码可从已登记 tag/commit 恢复。

#### Modality ablation 绘图与训练血缘审计（2026-07-19）

- **已由最终绘图 notebook 验证的事实：** Mac 的 `figs.ipynb` cell
  `8d84054140b51b7d` 直接硬编码了四条曲线。按 phylum/species/strain 顺序，w/o text + w/o sm
  为 `0.2382/0.3289/0.4514`，w/o genome + w/o sm 为 `0.2130/0.3441/0.4376`，w/o sm 为
  `0.2670/0.3462/0.5184`，完整 ApexOracle 为 `0.2674/0.4010/0.4890`。这些值及系列顺序已
  冻结在 `experiments/modality_ablation/paper_values.csv`；canonical 只读绘图入口为
  `scripts/reproduce/plot_modality_ablation.py`。
- **已由两台机器审计验证的事实：** 九个候选 genome-only、text-only、genome+text driver
  在本机与 node002 的 SHA-256 完全相同；checkpoint 留存却分散且不完整。代码、日志、W&B
  output 和表格中均没有检索到全部 12 个最终数值。checkpoint payload 的 `R2` 是单 member
  best held-out score，不是绘图使用的 ensemble R²。
- **根据现有证据作出的推断：** 三条 w/o sm 曲线最可能来自较早的 ChemBERTa-era modality
  driver 家族，但现有证据不足以指定每个点使用的精确 group/member、预测文件和聚合过程。
- **仍待确认但不再阻塞代码收尾的事项：** 完整训练血缘尚未恢复，因此当前支持级别是
  `paper plot reproducible / legacy training rerun unavailable`。15 个旧 driver 均会原地写过滤后的
  中间 CSV，已在不运行的前提下登记哈希并归档删除；受保护 CSV 的清理前后 SHA-256 一致。
- **已由清理后验证确认的事实：** canonical 入口成功生成 1-page PDF/PNG，冻结值、绘图配置和
  受保护 CSV 哈希未变化；全仓库为 111 passed / 4 skipped。

### 论文时期训练使用的最终数据

| 模态 | 论文来源和数量 | 最终本地训练数据 | 需要注意的区别 |
| --- | --- | --- | --- |
| AMP MIC | DBAASP 下载于 2024-09-27：16,408 个 peptide、5,630 个 strain、105,547 条 MIC；in-house：1,642 个 peptide、11 个 strain、15,718 条 MIC；论文合并后为 17,988 个 peptide、5,632 个 strain、121,265 条 MIC | 化学结构终版：`DataPrepare/Data/DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv`，121,265 行；token cache 终版：`DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv`，120,955 行 | 论文数量是 tokenizer、长度和 UNK 过滤之前的数量。训练脚本通常读取 120,955 行 token 文件，然后继续根据 genome/text embedding 可用性和 token 长度过滤。标签在单位和操作符处理后转换为 `-log10(MIC/10)`。 |
| 小分子抗生素二分类 | 共 49,331 个 molecule-strain pair：RN4220 39,312；BW25113 2,335；ATCC 17978 7,684 | `small_molecule/processed/small_molecule_Evo_binary_data.csv` 有 49,331 行；tokenized `..._SELFIES.csv` 有 49,330 行 | 三个来源的历史转换规则已从 `DataCheck.ipynb` 恢复，并由 canonical 只读 builder 逐字节复现两份冻结输出。SELFIES token filter 仅因 UNK 排除 `na_12751`。 |
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
- PepLink chemistry core 当前独立发布版本为 `PepLink==0.1.2`：仓库
  `https://github.com/DragonDescentZerotsu/PepLink`，tag `v0.1.2`，commit
  `90f627cc7fd65daaf9c5d0a973d17b79bcd097d5`，MIT license。ApexOracle 不 vendor、不使用
  submodule，只通过 `src/apexoracle/data/peplink_adapter.py` 调用公开 API。旧
  `aa_seq_to_smiles.py` 仅因仍有 legacy data driver import 而暂留，不再作为 canonical 实现。
- **已验证事实：** 独立 PepLink 0.1.2 测试 23 passed；内置 amino-acid mapping 与论文文件 SHA
  一致。0.1.2 只修改 reverse Histidine template，forward structure generation 与 0.1.1 相同。
  179 条历史 structure correction 中 177 条与 v0.1.1/0.1.2 逐字符串一致；DBAASP 19000
  和 21769 的差异仅为 v0.1.1 `FragmentParent` 移除游离 fragment，全部 179 条均为 fragment
  parent equivalent。这两个 ID 在最终 token cache 中影响 9 行。论文复现必须使用 frozen
  paper CSV 的 SHA，新数据使用 v0.1.2。
- `try.py`：为缺少 PubChem SMILES 的 DBAASP peptide 补结构的早期原型，输出缺失结构的中间 CSV。
- `correct_SMILES_offered_by_DBAASP.py`：重建 DBAASP 提供的 peptide structure 以保留 stereochemistry，并更新 merged SMILES/Evo MIC 文件。它晚于 `try.py`，属于最终化学结构清理血缘。
- `concentration_unit_transfer.py`：最早的 MIC 单位转换原型。`concentration_unit_transfer_new.py` 面向 19-task wide table；`concentration_unit_transfer_all_bact.py` 扩展到全部 bacteria 并计算 mean；历史最终 long-format 行为现由 `src/apexoracle/data/amp_mic.py` 和只读入口 `scripts/prepare_data/build_amp_mic_dataset.py` 取代。
- `APEX_in_house_to_SMILES.py`：把 in-house APEX 表转换为早期 wide SMILES 格式。`APEX_in_house_to_SMILES_merge_w_DBAASP.py` 合并该早期格式。最终 long-format、merge 和 AMP token filter 现由 `src/apexoracle/data/amp_training_data.py` 与 `scripts/prepare_data/build_amp_training_dataset.py` 取代。小分子二分类转换已迁移至 `src/apexoracle/data/small_molecule_antibiotics.py` 与 `scripts/prepare_data/build_small_molecule_antibiotic_dataset.py`；旧 converter 已由 legacy tag 保留后删除。
- **2026-07-19 已验证事实：** canonical MIC 重建得到相同 105,547 行，ID/strain/SMILES 精确一致，MIC 最大绝对误差 `4.55e-13`，没有记录超出 `1e-12` tolerance。历史 structure correction 是先用旧 SMILES 分子量换算 MIC、再原地替换展示 SMILES；新实现通过 179 条只用于分子量的 override 显式复现，不覆盖任何原始文件。frozen in-house long table 合并后的 121,265 行 CSV 和固定 IBM tokenizer revision 后的 120,955 行 token cache 均逐字节一致；310 行仅因超过 1024 tokens 排除，invalid/UNK 为 0。PepLink 新建 in-house structure 与 legacy 只差 terminal `[OH]`/canonical `O`，归一化后 15,718/15,718 行一致。
- **2026-07-20/21 已由完整数据审计与发布端验证的事实：** PepLink 0.1.1 对 16,430 个 curated DBAASP
  peptide ID 中 16,075 个成功 forward 的结构全部通过 SELFIES molecular-graph round-trip。
  reverse contract cohort 为 4,939 个，0.1.1 annotation round-trip 为 3,729/4,939；全部 1,210 个
  失败均含 Histidine，且只影响 reverse parser，不影响论文 forward data generation。修复版
  `PepLink==0.1.2` 已通过 PR #4 合并到 commit
  `90f627cc7fd65daaf9c5d0a973d17b79bcd097d5`，tag `v0.1.2`，并发布到 GitHub Release 与 PyPI；
  wheel/sdist SHA-256 已与 PyPI JSON API 核对。正式版达到 4,939/4,939，其中支持的 head-to-tail
  cyclic 为 523/523，测试为 23 passed。459 条 bundled
  residue definition 中 7 条 isolated forward 失败，其中 5 条进入 curated cohort、共 36 次
  residue occurrence。证据位于 `experiments/peplink_validation/`。
- **2026-07-20 已由血缘对齐和逐条人工审计验证的事实：** 169 条历史
  `DBAASP annotation -> ChatGPT-o1 name -> OPSIN SMILES -> final mapping` 完整对齐；105 条
  `verified`、14 条 `verified_source_annotation_typo`、20 条 `ambiguous_source_annotation`、
  22 条 `incorrect_pipeline_output`、7 条 `non_exact_polymer_proxy`、1 条
  `not_a_complete_amino_acid_definition`。OPSIN SMILES 的 RDKit validity 与 SELFIES structure
  preservation 均为 169/169，但这不证明 name-to-structure chemical identity 正确。
- **2026-07-21 已由 notebook producer、CSV lineage 与实际 model eligibility 验证的事实：** 459 条
  mapping = 39 条标准 L/D + 420 条 noncanonical；420 条的最终分支为 207 条保留 PubChem lookup、
  44 条二次 GPT-refinement+OPSIN correction、169 条无 PubChem 命中的主 ChatGPT-o1+OPSIN branch。
  旧说法“另外 251 条均未经过 GPT/OPSIN”错误；251 是初次 PubChem lookup success 数，其中 44 条
  后来进入第二条 GPT/OPSIN branch。完整血缘见
  `experiments/peplink_validation/AA_AND_PEPTIDE_LINEAGE_ZH.md`。
- **2026-07-21 已由 source-aware 交集验证的事实：** frozen MIC structure 来源为 DBAASP-linked
  PubChem CID whole-peptide 840 peptide/8,434 row、local residue builder 15,521/96,747、
  DBAASP-offered structure 69/366。whole-peptide PubChem 不经过 ChatGPT-o1、OPSIN 或 PepLink。
  PepLink forward failure 不等同于训练结构错误。
- **2026-07-22 已由 canonical loader 和逐 peptide 重算验证的事实：** 作者决定 coordination-bond
  omission 作为去金属/忽略配位预处理，不计为错误。实际 `<=512` token genome-or-text DBAASP pool
  为 15,177 peptide/74,103 row；其中 reviewer-facing 确认的本地 definition/template 错误为
  56 peptide/219 row（0.296%，报告 0.30%）。DBAASP sequence/unusual-AA annotation 内部不一致属于
  上游 source-data quality 和 historical producer 容错，不是 ChatGPT-o1/OPSIN 或 PepLink 转换错误；
  作者决定将其从 reviewer-facing 错误报告、回复和论文修改中全部排除。
  复现入口为 `scripts/audit/recalculate_reviewer_peptide_scope.py`，逐条结果位于
  `experiments/peplink_validation/recalculated_local_error_peptides.csv`。polymer proxy 不纳入范围。
- **2026-07-23 已由 canonical loader、RDKit、PubChem PUG REST 和文档渲染验证的事实：**
  `scripts/audit/build_peplink_supplementary_data.py` 生成英文期刊版和中文镜像 Supplementary Data，
  覆盖 56 个 peptide、219 条 loader row 和 18 个错误 definition；每条均记录 historical erroneous
  与 corrected structure/formula、位置、具体错误、处置和证据。16 个 direct PubChem definition 的
  名称、分子式和结构均复核一致；`NNar` 与 `D-3-OH-ASN` 因没有 exact public compound record 标为
  中等置信度且不猜测未指定 stereochemistry。manifest 登记输入/输出 SHA-256。正式
  `sn-article.tex` 已在 ChatGPT-o1/OPSIN Methods 段落加入 source-aware audit 和 56/219 数字；
  `Response to reviewers letter.docx` 已替换为完成时态精简回复。临时 TeX/DOCX 渲染分别为
  28/25 页，正式论文 PDF 未自动覆盖；被排除的 DBAASP annotation 脏数据仍未写入两份正式文档。
- **2026-07-23 reviewer 发布包边界：** 根目录 `.gitignore` 继续默认排除全部 CSV/XLSX，仅对
  `experiments/peplink_validation/` 中正式化学审计、PepLink 0.1.2 round-trip records 和
  Supplementary Data，以及 `experiments/reviewer4_unseen_targets/` 的公开候选表按精确路径放行。
  私有 in-house workbook、投稿 DOCX/PDF、PepLink 0.1.1/dev exploratory output、退役 sensitivity
  表和一次性投稿文档修改脚本不进入发布仓库。统一索引为 `experiments/README.md` 与
  `scripts/audit/README.md`。
- **根据现有证据作出的判断：** 0.30% 只能说明 prevalence 很低，不能证明 zero model impact。
  仅删除 held-out rows 的 evaluation sensitivity 不能消除 training exposure；历史 hierarchical MIC
  没有保存 ID-aligned row-level predictions，strain-wise 精确 2025 membership 也未恢复，因此当前
  不声称已做该 sensitivity。旧 forward-failure union 已退役，不进入 reviewer response。
- **仍待作者或 chemistry coauthor 确认的事项：** 解析 20 条 ambiguous source annotation、二次
  44-residue branch 的 source/site conflicts，并为 successor dataset 完成 56 个 reviewer-facing peptide 的
  corrected/excluded 处置。原 frozen paper CSV 不原地覆盖，仅用于既有模型复现。
- `DBAASP_SELFIES_Token_see.py` 和 `debug_notebook.py`：只用于 tokenizer vocabulary 检查。`debug.py` 把小分子 SELFIES 导出给外部 `mdlm` 项目。
- `bacteria_get.py`：统计 DBAASP JSON 中 strain/species 出现次数和 activity unit 变体，为 mapping 构建提供探索性支持。
- `canonical-peptide-check.py`：检查 canonical peptide 内容。`smiles_to_peptide.py` 把 peptide-like SMILES 反向转换为 D/L residue sequence，并被 peptide/non-peptide 标签脚本复用。
- `DataCheck.ipynb`：大型探索性数据检查 notebook，不是确定性的 build step。

#### 三菌株小分子二分类数据入口（2026-07-19）

- **已由代码和真实数据验证的事实：** 三个 raw source 的历史转换单元保存在
  `DataPrepare/DataCheck.ipynb`。E. coli 使用 `Activity == Active`；A. baumannii 使用
  `Mean < mean - sample std (ddof=1)`；S. aureus 直接复制 `ACTIVITY`。三个 block 的冻结顺序为
  E. coli、A. baumannii、S. aureus，分别为 2,335/120、7,684/480 和 39,312/512
  （总行数/阳性数）。
- **已由完整只读重算验证的事实：** canonical builder 生成的 49,331 行 merge CSV 与现有论文
  文件逐字节一致，SHA-256 为
  `4dabc0f8ac808d33ede3eacb47bacf7b55b2a900fcf78fd3d45a89c2037f3dc2`。固定 tokenizer
  revision 后，49,330 行 token cache 也逐字节一致，SHA-256 为
  `d8e6391bfae3c35fe8d311461565df177fc75044cbc40204bf74f7ecf1fe7f27`。
  唯一过滤记录 `na_12751` 含 UNK token；invalid SMILES 与超长记录均为 0。
- **根据现有证据作出的推断：** 当前冻结 merge 的 block 顺序来自当时 notebook 所见的目录
  遍历顺序；由于输出与完整重算逐字节相同，该顺序已被明确固化为 paper contract，但不应把
  未排序目录遍历本身视为推荐协议。
- **发布边界：** 原 notebook 会遍历 processed 目录并原地写回 merge 输出，旧 converter 也有
  硬编码绝对路径、未固定 revision 和允许覆盖输出的问题。canonical CLI 明确列出三个输入，
  拒绝输入输出同路径及覆盖既有输出。训练 runner 从 versioned config 显式读取 frozen token table；
  不在训练时重建或修改论文数据。完整 manifest 与审计见
  `configs/data_pipeline/small_molecule_antibiotics_paper.yaml` 和
  `experiments/small_molecule_antibiotics/`。

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
- in-house FICI 转换、候选 pair 枚举和 SMILES 转换脚本属于论文后 prospective workflow，已按
  paper-only 决策删除；数据文件本身未修改。

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
- `debug_notebook.py` 和 `MDLM_data.ipynb`：descriptor/schema 探索。
- `bash/node_descriptor_H100.sh` 与 `node_descriptor_node_001.sh`：机器特定的 shard launcher；其中 `/data1` 和 node 路径不可移植。

### 不属于 ApexOracle 论文复现主线的代码

- `Fangping_correlation/` 的 11 个 Bullseye/UPenn count/correlation 文件属于未发表旁支，且没有
  被任何论文入口引用；已于 2026-07-19 从工作树删除并由 legacy tag 保留。
- `e3nn_playground/` 的 4 个教程文件没有被 ApexOracle import；已于 2026-07-19 删除并由
  legacy tag 保留。
- PeptideCLM runtime 只保留在 `src/apexoracle/vendor/peptideclm_tokenizer/`：SMILES SPE tokenizer、
  vocab、merges、MIT LICENSE 和来源说明。旧教程、notebook、clustered dataset 与未使用 tokenizer
  已删除；代表性 token IDs/masks/decode 与旧实现逐项一致。
- `GPU_eye.py`、`run_full.py` 和故意占用 GPU 的 `run.py` 是非科学资源工具；已于 2026-07-19
  删除并由 legacy tag 保留。
- `bash/PeptideCLM_benchmarking.sh` 是指向已迁移 root driver 的历史 CESGA launcher，已随 Fig. 2c
  comparator 迁移删除；由 legacy tag 恢复。
- `environment.yml`：名为 `cold_base` 的完整 Anaconda 环境导出，规模过大，不是最小可复现环境。发布清理时应替换为经过筛选的 environment 或 lockfile。
- `Readme.md`：只有几行的过期说明，仍指向旧 mean-MIC 文件和 `fine_tune_on_DBAASP_SMILES_5_fold_mean_MIC.py`；必须重写，不能作为当前文档使用。

### 复制版本和第三方辅助文件的显式文件名索引

本索引用于确保以后搜索任意被通配符归类的文件名时，都能直接命中本审计；具体语义和终版判断仍以上文为准。

- **已删除并由 legacy tag 保留：** 旧 DP/in-house/SM hierarchical drivers、11/3 species
  复制版本、strain MDLM root driver 以及 `*_cls_wo_padding*`、`*_mean_wo_padding*`、`*_eval.py`
  feature 变体。canonical 替代入口是 `scripts/reproduce/run_hierarchical_mic.py`。
- **已迁移的 Fig. 2c comparator：** 四个 root driver 的不同语义没有被抹平，而是进入
  `legacy_fig2c_comparators.yaml` profiles；root 副本已删除并由 legacy tag 恢复。逐文件 SHA、
  node002 差异、checkpoint 证据和仍未解决的 PeptideCLM 血缘见
  `experiments/hierarchical_mic/strain/fig2c_comparator_migration_audit.json`。
- **已删除的 modality ablation legacy driver：** 9 个论文候选 family 文件和 6 个更早 prototype
  均由 `experiments/modality_ablation/legacy_cleanup.json` 记录文件名、SHA-256、恢复 tag 和受保护
  数据哈希；发布工作树只保留冻结数值、绘图配置、绘图入口和测试。
- **已删除的 Fig. 2b 历史副本：** `fine_tune_on_DBAASP_SMILES_5_fold_compare_pre_SSL.py`、
  `fine_tune_on_DBAASP_SMILES_5_fold_compare_SSL.py` 及同家族四个早期/all-data/in-house driver。
  五个 root `fix_*` 和 mean-pooling diagnostic。全部由 2026-07-20 清理 manifest 和 legacy tag
  恢复，主发布入口只含论文汇报的七个模型。
- **已删除的非论文 Synergy 副本：** early/prototype、continuous-FICI、all-data、in-house、
  few-shot、guidance、prospective regression/screening 代码均由 paper-only 清理 manifest 追溯；
  当前只保留论文三折 classification 候选及 canonical CV 入口。
- 生成分子的 fingerprint 变体：`DataPrepare/Morgan_fingerprint_sim_generation_SM_rediscover.py`。
- 已删除的 `compare_APEX/__init__.py`、`PeptideCLM/__init__.py` 和旧 tokenizer marker 不包含实验逻辑；
  精确文件身份记录在 cleanup manifest。

### Capsule 与审稿复现历史

- 从未上传的 `capsule/` 本地 staging 曾占 168,375,793,924 bytes、含 2,968 个派生文件；确认
  当前代码无消费者后已于 2026-07-20 删除。没有删除其来源数据或主 checkpoint，代码由 canonical
  runner 和 legacy tag 追溯。
- `capsule_fig2/` 已重构为约 256 KB 的正式共享 benchmark 审计包。`code/run` 从冻结的 35-fold
  指标重新计算七模型 mean R²/sample SD，并验证 10,886 个 molecule 的 fold 覆盖；不再携带
  feature cache、head checkpoint、mean-pooling diagnostic 或 root legacy driver。
- `capsule_fig2/code/prepare_fig2b_mic_regression_resources.py` 只打包 canonical modules、配置、
  实验 README、migration audit 和最终小型指标文件。
- 审稿阶段的 `scripts/`：
  - `reproduce_fig2b_mdlm_cached_5fold.py`：使用外部 `mdlm` cache 或评估 DLM feature。
  - `reproduce_fig2b_baselines_online_5fold.py`：在线提取 baseline feature 并训练或评估 head。
  - `cache_fig2b_molformer_fold_eval_features.py`：MolFormer cache builder。
  - `reproduce_fig2b_baselines_cached_5fold.py`：基于 cache 的 baseline evaluation。
  - `reproduce_fig2b_apex_original_5fold.py`：APEX reproduction driver。
  - `sweep_fig2b_cached_baseline_seeds.py`：用于诊断或协调 Fig. 2b 差异的 seed sweep。
- 当前 `capsule_fig2` 精确审计修订后的正式共同数据汇总；它验证已保存指标，不替代 35-fold
  训练，也不声称从模型权重重新生成预测。

### 已知复现缺口与不一致

1. 当前没有 Git 历史，因此不能确定论文最终版本对应的精确 commit。
2. DLM 训练代码、checkpoint 和 Fig. 2b DLM 源代码依赖外部 `/data2/tianang/projects/mdlm`。
3. Evo-2 genome window feature extraction 按作者决定保留为外部 producer；当前仓库只消费预计算 tensor。
4. k-mer producer/consumer 已迁入并验证，但产生论文 `0.4507` 的精确单模型训练日志、projection 状态和 checkpoint 仍未恢复；现存完整 `0.5276` 是协议不同的 post-paper 7-member reconstruction。
5. 实际 discrete guided-generation/remasking sampler 按作者决定继续位于外部仓库。
6. species-wise 唯一异常组和 phylum-wise `0.0100` aggregate 差异已量化并由作者决定暂不追查。
7. 现存 synergy CV 脚本最接近论文，但 LoRA rank 与论文描述不一致，aggregate 指标也略有差异。
8. 论文描述的不可变 2,732 行 synergy 数据没有保存；当前只有更大的原始表和动态过滤逻辑。
9. 三个已发表 small-molecule 原始数据集存在，但完整 merge/clean 脚本缺失。
10. 湿实验 MIC heatmap、toxicity、in-vivo 分析和最终候选选择记录没有形成可复现代码或数据 pipeline。
11. 大多数脚本包含绝对路径、重复模型定义、隐式全局状态，以及硬编码 device/fold；许多脚本无法从干净 checkout 直接运行。
12. 一些日志或 checkpoint 不完整、损坏或体积极大。存在 checkpoint 不代表训练已经完成；必须同时核对配套日志和预期 ensemble 数量。

### 公开发布前的安全阻塞项

- legacy snapshot 审计曾在下列文件发现明文 API 或服务凭据：
  `DataPrepare/discription_generation.py`、`discription_generation_w_ATCC.py`、
  `discription_generation_wo_ATCC.py`、`get_synergy_Evo.py`、`resistant_gene_check.py` 和已删除的
  `compare_APEX/APEX_all_data.sh`。当前 Git 历史只保留脱敏版本；任何旧凭据仍应视为已泄露并撤销。
- 所有嵌入代码的凭据都应视为已经泄露：必须撤销或轮换，从全部文件和未来 Git 历史中清除，并改为通过环境变量或有文档说明的 secret manager 读取。
- 不要把凭据值复制到 issue、日志、文档或审稿复现产物中。

### 建议的代码库清理顺序

1. 撤销并删除全部明文凭据，同时加入 secret scanning。
2. 为最终 AMP、小分子、synergy、genome、text 和 checkpoint 产物建立不可变 manifest，记录 hash 和行数。
3. 抽取一个共享 ApexOracle library，统一 dataset/mapping、split protocol、fusion block、head、metric 和 checkpoint schema；每个论文实验只保留小型 config-driven 入口。
4. 高置信度 hierarchical MIC 和 strict zero-shot 分类路径已经完成统一入口；下一步把这些
   已验证入口纳入正式 quickstart，并继续迁移 sequence similarity。
5. species/phylum、synergy rank 和 k-mer 的残余差异已经机器可读归档；不要把高置信度候选写成精确历史复现。
6. 清晰拆分外部项目：要么在许可证允许的前提下 vendoring 固定版本的 DLM/generation 代码，要么把它们声明为带版本的外部依赖。
7. 把历史副本、W&B 日志、notebook、旁支项目、巨型 checkpoint 和 reviewer capsule 移出源代码包；不要盲目删除 provenance，而应保留机器可读的 archive manifest。
8. 用最小且经过测试的安装说明、数据下载说明、实验配置、预期指标和容差、license/citation 以及清晰的支持矩阵，替换当前 `environment.yml` 和 `Readme.md`。
