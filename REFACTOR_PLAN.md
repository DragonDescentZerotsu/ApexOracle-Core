# ApexOracle / Synergy 代码库重构计划

> 建立日期：2026-07-17  
> 状态：canonical 论文/审稿路径已完成；现有 ApexOracle 已原地转换为 super-repo，MDLM 与 Generation modules 已锁定；
> legacy `DataPrepare/` 收尾仍待后续批次
> 适用范围：当前 `Synergy` 仓库，以及后续需要整合的 Evo-2 genome embedding、DLM/MDLM 和 guided generation 代码。

当前发布架构焦点（2026-08-10）：作者已冻结“复用现有 ApexOracle 作为 super-repo + 五个独立
submodule”的方案，其中
合作者的 DLM+MTR 预训练与本地 downstream MDLM 分为两个模块。下一步
按 `docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md` 第 8 节执行：MDLM 与 Generation 已关闭并作为固定 gitlink
进入现有 ApexOracle；Evo-2 已形成首个本地 clean candidate，下一步完成其 40B runtime/remote 验收，
随后处理 DLM-pretraining 与 Core，再补齐三个 pending gitlinks 和端到端 quickstarts。MDLM/Generation
clean modules 和 Hugging Face 模型已发布并验收；legacy `DLM_pretrain/` 与 Core 尚未完成 clean candidate，
因此不创建对应浮动 submodule、不移动数据或权重。

当前执行焦点（2026-07-28）：Fig. 1b 完整 10-member/fold fine-tune 与 baseline、最终统计、
图文修订，以及 PepLink 0.1.2 round-trip/ChatGPT-o1/OPSIN 审计和 Supplementary Data 均已完成。
ReMDM remasking schedule 的 36-task reviewer 补实验和 3,600-attempt clean-model evaluation
已完成，冻结协议、结果和表述边界位于 `experiments/remasking_schedule_reviewer/`。2026-07-29
structure audit 证实历史 v1 classifier-positive 不能单独解释为 generated-structure peptide
identity，随后冻结了窄结构筛选与正确 SEP-padding classifier 的联合内部口径，并重算 canonical
三面板 reviewer figure。2026-08-02 正式 TeX、Supplementary Fig. C4 和三处对应 reviewer
responses 均已落稿并独立渲染核验；对外文稿只报告 peptide/RDKit-valid yield、精确计数和
predicted MIC，不展开内部判定规则。完整证据和正式文件记录见
`experiments/remasking_schedule_reviewer/STRUCTURE_AUDIT.md` 与该实验 `README.md` 第 13 节。
2026-08-02 新增 guided-generation diversity reviewer 审计：冻结作者确认的 73-row corrected
peptide candidate pool，并对两个最终 target 的 84,226 条 guided-generation outputs 计算 exact
canonical-structure uniqueness 与可复核 Morgan/Tanimoto distributions；canonical 入口和边界见
`experiments/generated_candidate_diversity/README.md`。
作者确认的论文落稿计划已执行：在三条 representative leads 的 training-set sequence-similarity
结果后加入 generated-set diversity Results，报告 73-row peptide candidate pool 的 71 个 distinct
canonical isomeric structures（97.3% unique、median Tanimoto 0.375）和 84,226 条
MIC-guided outputs 的 83,433 个 distinct structures（99.1% unique、median Tanimoto 0.190），并
引用新增的 Supplementary Fig. C5。前面 hierarchical MIC Results 保留更直观的
`we additionally evaluated exact-peptide overlap` 及其 train-seen/train-unseen 解释；后续原
`Together with the exact-peptide-disjoint strain-wise sensitivity analysis described above` 改为
`Together with the strain-wise sensitivity analysis on peptides absent from the corresponding
training folds, ...`，继续明确它是 MIC predictor sensitivity 而非
generation-diversity test。Results 还明确 Fig. 3a 是 predicted candidate distributions 的
*in silico* comparison，prospective MIC 才是 selected candidates 的直接实验评价。
同一轮还完成了 18 条 purchased small molecules 的历史资产审计：冻结 prediction tables 对相同
44,608 个唯一 SMILES entries（39,995 个 RDKit-valid canonical structures）分别预测两个 targets，
`predicted MIC <=15 µM` 后为 1,554/395 rows、
合并 canonical 去重为 1,535 个 structures；与 MolPort exact canonical-structure matching 得到
179 个 available structures，其中 80 个未命中 PAINS/BRENK/custom structural alerts。19 个
quote-stage compounds 位于 19 个不同 Butina clusters；最终实验 18 条相对 quote 只排除了单价
显著高于其余条目的 `Molport-002-070-273`。现有 quote 不支持“因运输时间过长排除候选”，正式
文稿只写 procurement feasibility/cost。完整事实、推断和待确认边界记录在
`experiments/generated_candidate_diversity/molport_selection_audit.md`。正式 TeX 已按本轮
selection/diversity 计划修改；两条 selection reviewer replies 已在
`reviewer_response_draft.md` 中按同一分母和 manuscript revisions 补齐，并已合并进正式
`Response to reviewers letter.docx`。修改前备份为
`Response to reviewers letter_before_generated_diversity_selection_20260802.docx`；独立渲染为 30 页，
三处新回复的分页和格式已核验。
进一步核验确认 44,608 是 processed screening collection 的唯一 SMILES-entry 数，对应 39,995 个
canonical structures，并非 MolPort 全库；本地历史 MolPort snapshot 为 5,887,458 个唯一 IDs。
其上游已由逐字节/逐集合复核闭合：Fig. 1b 的三套公开 small-molecule classification data 按
2,335/7,684/39,312 rows 合并为 49,331 molecule--strain rows；`DataPrepare/debug.py` 只抽取 SMILES
并逐条转为 SELFIES，两个 unseen-target input files 均为相同的 49,331 lines 和相同 SHA-256，按
唯一 SELFIES 汇总后恰为 prediction tables 的 44,608 entries。原二分类 labels 不进入本次 MIC
scoring；44,608 不是 generated molecules。shell history 只证明运行顺序，未保存当时 mutable
producer script 的 byte-exact launch snapshot，因此不声称完整恢复原 command line。
Reviewer 4 的相似 selection/hit-rate comment 与本轮共用同一组 manuscript revisions：删除
`spanning a range of predicted MIC values`，明确 73 条均先满足 predicted MIC `<=15 µM`，再按
diversity/feasibility 选 24 条；以 intended-target experimental MIC `<=64 µM` 定义 hit 时，peptides
为 10/24（41.7%；PA5257 4/15、AR-0349 6/9；4/8/16/32/64 µM 分别 1/3/1/2/3 条），small
molecules 为 1/18（5.6%；2-fluoroadenosine 对 AR-0349 为 16 µM）。两条 reviewer response 分开
写，但引用同一 Results/Methods/Supplementary 修改和完全相同的分母。正式 TeX 的新增图为
`Fig_SI_generated_candidate_diversity.pdf`（当前三 panel SHA-256
`72c50e1649720233a3a45ba46d57d21df8fa68ffb2660aba886b3a4f491a8ab8`）；独立编译为 32 页，图号
已核验为 Supplementary Fig. C5，未覆盖正式论文 PDF。修改前 TeX 备份为
`sn-article_before_generated_diversity_selection_20260802.tex`。
正式 Supplementary Fig. B2 随后已由只展示 4 个 featured candidates 替换为全部 24 条
synthesized peptides 和 18 个 purchased small molecules 在 20 个 strains 上的 MIC heatmaps。作者
提供的 Mac v4 PDF 与正式 `Fig_SI_heatmap_re.pdf` SHA-256 均为
`5ac3bd00e52958ecb06bc066e29de4752863b0f31d851e18df531954c1ae2693`；TeX 改为双栏
`figure*`/`\textwidth`，caption 以 revision color 明确全部实验分母。独立编译为 32 页，
B2 位于第 23 页，C3--C5 编号不变；正式 `sn-article.pdf` 保持原内容。两条 selection
reviewer replies 亦已同步回指 complete B2 MIC matrices。
2026-08-03 已新增 24 条 final peptide 的 pairwise diversity audit：完全复用论文 lead
sequence-similarity 的 BLOSUM62/gap/PID/topology/cyclic-rotation 口径，并对无序 pair 做方向对称化；
结构指标继续使用同一 Morgan/Tanimoto 协议。结果为 24/24 structures unique、168 个同拓扑
sequence pairs 的 median PID `0.1719`（仅 ApexOracle-14/23 为高相似 near-neighbor）、276 个
结构 pairs 的 median Tanimoto `0.4633`。该结果可支持 selected panel 的 descriptive diversity，
但不能写成 selection 相对 random/top-predicted comparator 已证明提高 diversity；完整边界和产物见
`experiments/generated_candidate_diversity/selected_peptides_24/REPORT.md`。同一入口另生成共享
0--1 纵轴的 PID/Tanimoto 双 panel violin plot，便于直接审阅完整 pairwise distributions。
2026-08-03 同一 audit 新增不混合 target strain 的严格分层视图：按
`BAA-3197/BAA-3170 × linear/cyclic` 得到 66/3/15/3 pairs，四组 median PID 为
`0.1667/0.2500/0.2258/0.1818`，87/87 均低于 `0.5`。原 pooled high-PID pair
ApexOracle-14/23 跨 target strain，因此不进入该 selection-stratified 口径。
2026-08-03 已将这组 within-target PID 作为新 panel a 加入 Supplementary Fig. C5，原 candidate-pool
与 generation-level Tanimoto panels 顺延为 b/c。最终采用作者指定的三 panel 单行布局；Results
按 24→73→84,226 的顺序引用，Methods 新增 selected-peptide sequence-diversity protocol。Canonical
PDF SHA-256 为 `72c50e1649720233a3a45ba46d57d21df8fa68ffb2660aba886b3a4f491a8ab8`；独立编译为
32 页，C5 位于第 24 页且正式论文 PDF 未覆盖。C5 caption 已明确正式 strain 编号、四组
peptide/pair 数和 Lin./Cyc. 定义；0--100\% PID 轴与 comparison exclusions 分别只保留在图和
Methods。Reviewer diversity
回复和两条 selection 回复均加入 C5a 的 87-pair 定量证据。正式 response DOCX 最新 SHA-256 为
`5da0fc9894c861733917438b07904cb22777f800cceece7185b067b0464473bc`，修改前备份为
`Response to reviewers letter_before_selected_pid_c5_20260803.docx`，独立渲染为 30 页并完成目视核验。
2026-08-03 新增 Methods 已按作者要求重排：guided generation/remasking 留在 architecture/training；
candidate selection 与 small-molecule screen 进入独立 `Candidate prioritization and virtual screening`；
PID 在 `Sequence and structural diversity analyses` 开头统一定义，随后依次描述 lead-to-training、
selected-24 sequence 和 73/84,226 structural analyses，删除重复的 topology/cyclic-rotation 说明。
协议和结果未改变；修改前 TeX 备份为 `sn-article_before_methods_reorganization_20260803.tex`，
独立编译仍为 32 页，相关 Methods 页面已目视核验且正式论文 PDF 未覆盖。
发布前完成膨胀审计：21 MiB fingerprint cache、81 个可由外部只读输入重建的逐长度 SELFIES
provenance files、旧两 panel/selected diagnostic plots 均保持 local-only；Git capsule 只纳入
compact tables/manifests/docs、canonical 三 panel C5、四个入口、共享 module 与测试。该收紧只
改变发布白名单，不删除本地资产，也不改变任何分析功能或结果。
当前其余主要未完成事项是
corrected AA successor dataset 的重新训练、Reviewer 4 target/
multi-isolate panel 决策，以及 `DataPrepare/` legacy 脚本的迁移、去重和归档。机器职责、
代码同步、数据/权重和外部仓库位置以 `docs/COMPUTE_AND_ASSET_MAP.md` 为统一索引。

2026-08-06 genome-condition reviewer 轮次已完成并进入发布整理。正式证据仅包含两个
representation-level analyses：全部合格 bacterial embeddings 的 homologous-fragment variation，
以及 264-genome/96,716-fragment AMR/MGE linear probes。前者在 4,649 个 variable fragments 上的
pooled Spearman 为 `0.6954`（whole-genome `ANI >=99%` 子集 `0.7137`）；后者 AMR/MGE OOF
AUPRC 为 `0.2033/0.4456`，相对 prevalence `0.1667/0.1977`。Canonical 三面板图、正式命令、
输出和解释边界见 `experiments/genome_condition_reviewer/README.md`。

发布前维护审计将 annotation-manifest 与 linear-probe 逻辑从弃用的 swap scripts 解耦到
`src/apexoracle/evaluation/genome_fragment_validation.py`，并移除了正式入口对本机绝对 producer
路径的依赖。Genome/text swaps、162-genome probe 和 strain-wise fragment pilot 只保留本地内部
取证，不进入公共 capsule；公共代码只保留正式 pair preparation、fragment analysis、annotation
probes、figure 和 tests。正式 manuscript、Supplementary Fig. C6、三条新增引用和 reviewer response
均已落稿并独立编译/渲染验证；`REVIEWER_RESPONSE_AND_MANUSCRIPT_DRAFT.md` 现为实施记录，不再是
待确认草稿。

## 1. 重构目标

本次重构的目标不是简单删除旧文件，而是把当前以实验脚本副本为主的研究代码整理成一个可审计、可复用、可复现且适合公开发布的代码库：

1. 保留一份经过脱敏的原始源码快照，作为历史依据。
2. 清除非论文最终版、重复、调试、机器专用和无关旁支代码。
3. 把论文实验从大型单文件脚本迁移为共享模块、明确配置和稳定命令行入口。
4. 为每项论文结果记录数据版本、划分、模型、checkpoint 和指标血缘。
5. 按 reviewer 的要求，重新构建使用完全相同样本和 fold 的 molecule encoder benchmark。
6. 明确当前仓库能够完整复现、只能部分复现以及仍依赖外部仓库的部分。

仓库名称目前不作为阻塞项。最终完整发布物预计还会整合 Evo-2 genome embedding、MDLM 和 generation 等其他仓库，届时再统一仓库名、Python package 名和论文链接。

## 2. 已确认的关键决策

### 2.1 历史源码快照

- 初始 Git 快照包含：全部源码、capsule 中的源码、配置和清空输出后的 notebook。
- 初始 Git 快照完全排除：训练/评估数据、checkpoint、实验结果、缓存、日志、大型二进制文件和密钥。
- 在脱敏快照提交后创建 Git tag：`legacy-code-snapshot-2026-07-17`。
- 用户已经额外保存当前 legacy 代码，因此清理后的主代码树不再保留 `legacy/` 文件夹；旧实现仅通过 Git tag 追溯。

### 2.2 Fig. 2b 公平 benchmark

- 按 reviewer 要求，将完全共享数据和 fold 的 benchmark 作为新的正式结果。
- 如果新结果与当前论文图不同，应更新论文图、正文数值和 reviewer response。
- 所有 encoder 使用同一份样本清单和同一份 molecule-level 五折划分，不允许各模型在划分后静默丢弃样本。
- APEX 不因输入限制而缩小公共数据集：noncanonical residue 映射为 `X`，cyclic peptide 使用线性化序列表示。
- APEX 的上述表示属于有损投影，必须在数据 manifest、Methods 和结果说明中明确记录。

### 2.3 DLM/MDLM 模型范围

正式 benchmark 计划包含：

- DLM MTR + DLM；
- DLM MLM-only；
- ChemBERTa MTR；
- ChemBERTa MLM（first-token）；
- MolFormer；
- PeptideCLM；
- APEX。

ChemBERTa MLM mean-pooling 作为可选消融，不与论文主 comparator 混淆。DLM MLM-only 的代码和权重
从 `/data2/tianang/projects/mdlm` 及保存的 checkpoint 血缘中定位和核验；合作者 joint DLM+MTR
预训练 producer 则以 public legacy `ApexOracle/DLM_pretrain/` 为来源，二者不得再混写成同一个仓库职责。

### 2.4 统一公开仓库架构（2026-08-09 作者确认）

最终发布固定采用轻量 `DragonDescentZerotsu/ApexOracle` super-repo，通过 Git submodule 组合
`ApexOracle-Core`、`ApexOracle-DLM-Pretraining`、`ApexOracle-MDLM`、`ApexOracle-Evo2` 和
`ApexOracle-Generation` 五个独立模块。`ApexOracle-DLM-Pretraining` 只记录合作者的 DLM+MTR
预训练 producer；`ApexOracle-MDLM` 只记录 downstream checkpoint loading、molecule embedding、
guidance heads 和 candidate scoring。各模块保留自己的内部目录、依赖和环境；不再计划把 MDLM、
Evo-2 或 generation 大规模重排进当前 `src/apexoracle/`。PepLink 继续以 `PepLink==0.1.2`
版本化依赖接入，不作为 submodule。

Super-repo 只负责根 README、recursive-clone 指引、固定 module SHA、资产 manifests、分离环境、
bootstrap、MIC prediction quickstart、guided-generation quickstart 和展开 submodules 的完整 source
release archive。数据、权重、embedding、raw outputs 和 cache 不进入 Git。

当前 public `ApexOracle` 不再改名或由新 repo 替代；先以 tag/branch 保存 legacy 状态，再在同一
`DragonDescentZerotsu/ApexOracle` repository 原地转换默认分支。当前 `Synergy` 也不复制，发布前直接
把同一个 GitHub repository 重命名为 `ApexOracle-Core`。执行阶段、
模块边界和验收标准以 `docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md` 为详细操作计划。本决策已经冻结；
改变 module 数量、两个 MDLM 相关模块的职责边界、submodule 策略、目标 remote 或 canonical URL
切换方式必须再次取得作者确认。详细 MDLM 源码审计见 `docs/MDLM_MODULE_SPLIT_AUDIT.md`。

## 3. 目标目录结构

```text
.
├── src/apexoracle/
│   ├── data/                       # 数据 schema、读取、过滤、映射和划分
│   ├── features/                   # molecule、genome、strain text feature 接口
│   ├── models/                     # fusion、attention、prediction heads
│   ├── evaluation/                 # 指标、ensemble、预测导出
│   └── benchmarks/
│       └── molecule_encoders/      # Fig. 2b 公平 benchmark
├── experiments/
│   ├── fig1a_hierarchical/
│   ├── fig1b_zero_shot/
│   ├── fig2b_molecule_encoders/
│   ├── hierarchical_mic/           # strain/species/phylum 共用 runner + split adapters
│   └── synergy/
├── configs/                        # 可版本化实验配置
├── scripts/
│   ├── prepare_data/
│   └── reproduce/
├── reproducibility/
│   └── code_ocean/                 # 整理后的 capsule/Code Ocean 入口
├── quickstarts/
├── tests/
├── AGENTS.md
├── REFACTOR_PLAN.md
└── README.md
```

目录结构可以随迁移细化，但不得重新形成大量复制粘贴的单文件模型实现。

上述结构是当前 `ApexOracle-Core`（即本仓库）的内部目标结构，不是最终 super-repo 将所有源码合并
后的单体目录。最终公开入口的固定结构为：

```text
ApexOracle/
├── modules/
│   ├── core/          # ApexOracle-Core / 当前 Synergy
│   ├── dlm_pretrain/  # ApexOracle-DLM-Pretraining / 合作者预训练 producer
│   ├── mdlm/          # ApexOracle-MDLM / downstream embedding 与 guidance support
│   ├── evo2/          # ApexOracle-Evo2
│   └── generation/    # ApexOracle-Generation
├── quickstarts/
├── environments/
├── manifests/
├── scripts/
├── .gitmodules
├── README.md
├── LICENSE
└── CITATION.cff
```

五个 modules 均固定到 clean commit；super-repo 不复制其实现。完整目录、职责和 release contract
见 `docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md`。

## 4. 分阶段执行计划

### 阶段 0：建立审计基线

状态：已完成。

- [x] 在 `AGENTS.md` 中记录论文、代码、数据与 checkpoint 血缘。
- [x] 区分最终版、可能最终版、历史版本、论文后代码和缺失代码。
- [x] 确认 reviewer 对 Fig. 2b 的核心要求是相同数据和相同划分。
- [x] 核验当前 Git 仓库状态和 GitHub 远程仓库状态。
- [x] 定位 MDLM 仓库中的 DLM MLM-only 代码、配置和候选权重；精确论文 checkpoint 仍按证据等级记录。
- [x] 生成待保留、待迁移和待删除的机器可读清单：
  `reproducibility/migration_inventory.yaml`。清单中的删除权限默认关闭，只有替代入口和
  checkpoint 等价验证完成后才允许改变。

验收标准：任何删除或迁移都能从审计记录、Git tag 或清单中解释其原因。

### 阶段 1：创建脱敏的 legacy 源码快照

- [x] 从源码中移除硬编码 API key、W&B 凭据和机器私有认证信息，改为环境变量或不可用占位符。
- [x] 清空所有纳入版本控制的 notebook 输出和执行计数。
- [x] 建立 `.gitignore`，排除数据、checkpoint、结果、日志、缓存和大型二进制文件。
- [x] 检查 GitHub 单文件 100 MB 限制以及总体提交体积。
- [x] 显式检查 staged 文件，确保只包含源码、文档、配置和清空输出后的 notebook。
- [x] 扫描 staged 内容中的疑似密钥和大型文件；旧绝对路径保留在 legacy tag 中，重构分支再统一迁移为配置。
- [x] 创建本地初始提交和 `legacy-code-snapshot-2026-07-17` tag。
- [x] 将 legacy `main` 和重构分支同步到 `DragonDescentZerotsu/Synergy.git`。
- [x] 远程 tag `legacy-code-snapshot-2026-07-17` 已通过恢复后的 GitHub HTTPS 凭据推送；`archive/legacy-code-snapshot-2026-07-17` branch 继续作为额外恢复点。
- [x] 从该 tag 创建 `agent/paper-release-refactor` 重构分支，后续清理不直接破坏历史快照。

验收标准：从 tag 可以查看原始源码血缘，但仓库中不存在数据、checkpoint、结果或可用密钥。

执行记录：当前 Codex 工作区将 `.git` 作为只读保护挂载，因此 Git metadata 位于被忽略的 `.git-state/`，并通过 `--git-dir=.git-state --work-tree=.` 操作同一工作树。本地脱敏快照提交为 `a68707c`。最初本机没有可用 `gh`、SSH public key 或 HTTPS credential，因此使用已授权的 GitHub App Git object API 重建远程提交链；上传的 237 个去重 blob 和五个版本 tree 均逐一与本地 SHA 校验一致。远程提交因额外的仓库初始化 parent 而拥有不同 commit SHA，但每个科学代码版本的 tree 与本地完全一致。2026-07-18 已完成 `gh` 2.96.0 网页认证，Git protocol 和 `origin` 均切换为 HTTPS；随后成功 fetch 合并后的 `main` 并推送 annotated tag `legacy-code-snapshot-2026-07-17`。PR #2 已通过 merge commit `24d975c` 合入远程 `main`。

### 阶段 2：建立共享核心模块

- [x] 建立可安装的 `src/` package 和最小依赖定义。
- [x] 为已迁移的 canonical 论文路径抽取数据 schema、label transform、mask、strain mapping 和
  filtering；未迁移的 `DataPrepare/` legacy 脚本不包含在此完成声明中。
- [x] 为 canonical runner 抽取 genome/text/molecule feature 接口。
- [x] 为 hierarchical MIC、Fig. 1b 和 synergy canonical runner 抽取
  cross-attention/fusion、LoRA、regression/classification head。
- [x] 为已迁移主路径抽取指标、ensemble、checkpoint loading、seed 和设备选择逻辑。
- [x] 建立 `configs/model_weights.yaml` 统一登记权重当前位置、SHA-256、消费实验和计划迁移路径；
  `src/apexoracle/resources/model_weights.py` 已实现 manifest ID 解析及 size/SHA-256 验证。
  未完成的是其余历史权重的集中搬迁和再分发 URI/许可确认。
- [x] canonical runner 的运行时路径已迁移到 CLI 参数或 YAML 配置；`DataPrepare/` legacy
  脚本中的绝对路径留待后续归档清理。
- [ ] 统一预测输出格式，至少包含 sample ID、fold、label、prediction 和模型元数据。

执行进度（2026-07-19）：hierarchical MIC 路径已完成统一。strain-wise、论文中的
species-wise（11 clusters）和 phylum-wise（3 clusters）共用同一份 Dataset/collate、
precomputed feature loader、cross-attention、prediction heads、四路训练、评估、scheduler、
best-metric selection、checkpoint payload 和 ensemble runner；三个实验只保留 split adapter、
group 名称与输出路径差异。canonical 入口为 `scripts/reproduce/run_hierarchical_mic.py`，
配置为 `configs/hierarchical_mic/legacy_mdlm.yaml`。三个真实数据 dry-run、保存日志计数对照、
CPU 等价测试和 H100 一轮四路训练 smoke 均已通过；被替代的 DP/in-house/SM 和 pooling/eval
复制脚本已删除，完整历史由 `legacy-code-snapshot-2026-07-17` tag 保留。

验收标准：论文主实验不再复制共享模型和数据逻辑；公共模块具备单元测试。

### 阶段 3：优先完成 Fig. 2b molecule encoder benchmark

这是本轮重构中优先级最高、且会生成 reviewer 要求的新正式结果的模块。

#### 3.1 冻结数据协议

建立下列不可变产物：

- `common_molecule_ids.csv`：所有 comparator 共享的样本/分子标识；
- `folds.csv`：按 molecule ID 生成的一份共享五折划分；
- `exclusions.csv`：进入公共集合前的每条排除记录及原因；
- `dataset_manifest.json`：源数据校验和、过滤规则、label transform、随机种子、fold 统计和 APEX 投影规则。

APEX 输入转换规则：

1. canonical linear peptide：保留标准单字母序列；
2. noncanonical residue：映射为 `X`；
3. cyclic peptide：去除环连接语义，使用线性化 residue 序列；
4. 转换失败必须在构建公共集合前显式报错或记录，不允许训练/评估时静默跳过。

状态：**已实现并通过真实数据验证。** 当前源表 11,401 个 molecule。按各论文脚本的原生输入限制审计后，ChemBERTa-MTR、ChemBERTa-MLM 和 MolFormer 各可处理 10,889 个，PeptideCLM 可处理 11,377 个，两个 DLM 版本各可处理 11,082 个，APEX 可处理 11,321 个；最终公共交集为 10,886 个，五折大小为 2,178、2,177、2,177、2,177、2,177。APEX 输入中的 noncanonical residue 写为 `X`，但继续使用原始 23-token vocabulary；按原 `onehot_encoding` 行为，`X` 留在 index 0。不得修改 APEX 的 AAindex embedding、encoder、checkpoint 或 regression head。

#### 3.2 统一训练与评估协议

- 所有模型严格读取同一 `folds.csv`。
- 保留每个模型原论文实现中的 encoder、prediction head、训练参数和 checkpoint-selection 行为；本轮只统一 molecule IDs 和 folds。
- frozen encoder 与 fine-tuned encoder 必须明确分组，不能在同一表中无说明混比。
- 每个 fold 保存预测和指标；汇总报告 mean ± SD。
- 输出逐模型处理成功率和任何异常，但正式指标必须基于完全相同的 test IDs。

状态：**已完成。** 共享 native-processability 审计、公共 ID 交集、outer fold 校验、MIC label transform、指标实现和严格 ID 对齐的 `.npz` feature-cache 契约已实现并通过测试。曾新增的 10% validation、统一 prediction head 和统一 head runner 超出了 reviewer 要求，已经撤回。APEX adapter 已严格加载完整原 checkpoint，并为 10,886 个公共 molecule 生成及回读 `(10886, 128)` 审计 cache。各原始训练实现的共享 IDs/folds 薄入口和两个 DLM checkpoint 接入已经完成；正式 7-model × 5-fold 训练于 2026-07-18 全部完成，35 个 fold 无缺失且每个模型的 test IDs 均完整覆盖公共集合。训练保持原 200 epochs、batch size 200、Adam、`1e-4`、模型特有 head 和 train/eval mode；只缓存 held-out fold 上确定性的 frozen-backbone eval feature，以避免每个 epoch 重复相同计算。正式结果记录在 `experiments/fig2b_molecule_encoders/results_shared_5fold.md`。

权重审计补充：**已完成 node002 核验。** 本机、node002、W&B 和公开 Hugging Face 权重中均未发现 24-layer/1024 纯 DLM checkpoint，因此当前正式表中的 12-layer DLM-only 与 24-layer joint 只能表述为模型版本 benchmark。随后在 node002 原始 Fangping run 中确认了 12-layer/768 joint `best.ckpt`（step 650032），可与 `best_2.ckpt` 做容量匹配的新五折比较；但二者预训练 learning rate、global batch size 和最佳 step 不完全一致，不能称为严格单变量 objective ablation。该候选尚未运行，不能提前替换正式结果。

#### 3.3 capsule 迁移

- 将 `capsule_fig2/` 的可复用源码迁移到 `src/apexoracle/benchmarks/molecule_encoders/` 和 `experiments/fig2b_molecule_encoders/`。
- 将 Code Ocean 专用入口整理到 `reproducibility/code_ocean/fig2b/`。
- 删除 capsule 内重复源码和历史结果，不保留第二套 canonical 实现。

验收标准：**已满足。** 一条数据准备命令生成共享 manifest/folds；benchmark runner 能选择各 encoder；最终汇总包含全部 35 个 fold 指标和 mean ± SD。

### 阶段 4：迁移论文其余最终实验

#### 4.1 优先迁移的高置信度最终版

- Fig. 1a / Fig. 2c strain-wise、species-wise 和 phylum-wise DLM ensemble：统一入口
  `scripts/reproduce/run_hierarchical_mic.py`，旧 root drivers 由 legacy tag 追溯。
- Fig. 1b 三菌株分类：统一入口
  `scripts/reproduce/run_antibiotic_classification.py`，旧 root drivers 由 legacy tag 追溯。
- AMP/PepLink 最终数据处理血缘。
- ApexOracle-3/12/23 sequence similarity 流程。
- reviewer 的 Evo-2 embedding scaling 分析脚本。

状态（2026-07-19）：三个 hierarchical holdout 和 Fig. 1b 三菌株分类均已完成单一 runner
的行为保持迁移。

- 共享实现位于 `src/apexoracle/{data,features,models,training,evaluation}`；唯一入口为
  `scripts/reproduce/run_hierarchical_mic.py`，实验契约和审计材料位于
  `experiments/hierarchical_mic/`。
- 完整 outer loop 已脱离 root legacy driver。测试覆盖 logits、loss、gradient、Adam 更新、
  epoch-0/逐 epoch evaluation、prediction 分区、loader `zip_longest`、scheduler、strict
  best-metric selection、七键 checkpoint payload 和统一 runner 的 H100 一轮集成 smoke。
- 21 个历史 checkpoint 的 `3 × 7` 网格和实际消费的 fusion/head contract 已全部扫描。
  group 0/2 的 14 个文件只保存 fusion/head；group 1 的 7 个文件额外保存一个名称错误的
  `ChemBERTa_state_dict`，其结构实际是 12-layer/768 MDLM backbone。三个 group 的 optimizer
  均不包含该 backbone 参数；是否在 group 1 的 forward 中在线使用仍待旧源码确认。
- 历史 split 同时依赖无序 `set` 和原地 taxonomy-alias list mutation；日志没有记录三个独立
  Python 进程的 `PYTHONHASHSEED`。本机与 node002 的 driver、核心数据和文件名清单已经核验
  一致，因此当前只能提供 `PYTHONHASHSEED=0` 的确定性候选 manifest，不能声称恢复了 2025 年
  checkpoint 的精确 strain membership。过渡入口要求显式确认这一限制。
- group 0 / ensemble 0 已在 H100 上完成两次独立固定批次严格加载与推理，结果一致；其余
  checkpoint 已完成结构扫描，但剩余 20 个 SHA-256 和逐文件推理仍待补齐。

因此 hierarchical MIC 项当前状态是 `unified runner complete / legacy duplicates removed`。

Fig. 1b 三菌株分类也已完成统一迁移。canonical 入口为
`scripts/reproduce/run_antibiotic_classification.py`，通过 `--mode strict-zero-shot`、
`fine-tune` 或 `molecule-only` 显式选择三种旧协议。共享数据/fold adapter、feature loader、
fusion/head、四路训练、metrics、best-AUROC selection、checkpoint schema 和 ensemble runner
已经抽取；旧 root drivers 和两份机器专用 launcher 已删除并由 legacy tag 保留。

行为验证包括：真实数据 dry-run 与旧日志计数逐项一致；11 项 CPU 测试；H100 上四路合成
batch 的一轮训练/评估 smoke；以及 strict zero-shot group 0 / ensemble 0 的 2,335 条真实
checkpoint logit 与 capsule 在 batch size 70 下逐值完全一致。30 个 strict checkpoint 和
150 个 molecule-only checkpoint 网格完整。最初本机单独审计只看到 77/150 个 fine-tune
checkpoint；合并 node002 后恢复 104 个历史 checkpoint，随后补训 RN4220 fold 4 member 0，
本轮开始前为 105/150。剩余 45 个 member 已于 2026-07-21 补齐；最终 `150/150`
checkpoint 网格、15 个 10-member fold prediction、matched baseline 和 paired statistics
均已验证完成。
详见 `experiments/fig1b_antibiotic_classification/`。

Sequence similarity 已于 2026-07-19 完成迁移。canonical 入口为
`scripts/reproduce/run_sequence_similarity.py`，配置为
`configs/sequence_similarity/paper_leads.yaml`。训练 cache 可从当前源数据逐字节重建；
ApexOracle-3/23 的四份核心历史 CSV 也与 canonical 全量重算逐字节相同。三条 lead 的
最大 PID 均与论文一致。历史 cache 的 training sequence 使用 uppercase normalization；
直接保留当前 DBAASP JSON 大小写会改变 ApexOracle-23 结果，因此只作为 sensitivity，不能
替代正式 paper contract。ApexOracle-12 的旧 full CSV 未保存，且其 35.7% 有四个 complete
ties；论文展示 15510，而稳定输入顺序选择 9800，数值不变。旧 `DataPrepare/get_similarity`
tracked drivers/manifests 已删除并由 legacy tag 保留。详见
`experiments/sequence_similarity/`。

AMP/PepLink、synergy、modality ablation、k-mer 和其余 generation 前路径已经完成发布清理与
证据冻结。Fig. 1b reviewer 补实验、完整 ensemble paired statistics 和最终文稿已完成；
generation 外部仓库仍为只读/外部边界，不在本阶段修改。

Reviewer 4 的 unseen-species 初筛已于 2026-07-20 插入 Fig. 1b 与 guidance 外部重构之间。
已从 Mac 复制私有 in-house AMP workbook 到 Git ignored 目录并核验两端 SHA-256；canonical
审计入口为 `scripts/audit/audit_reviewer4_inhouse_species_coverage.py`，执行 notebook 和 aggregate
CSV/JSON 均留在 `DataPrepare/Data/private_inhouse_amp/`，不得上传 GitHub。producer 过滤路径确认
guidance regressor 实际训练 exposure 为 1,599 个标准化 strain ID / 389 个 producer-era species；
表格 35 个 normalized species 中有 11 个未进入实际训练，10 个具有 MIC 测量。

本项当前状态是 `target discovery complete / generation assets blocked`。已验证 11 个候选均缺少
完整 exact-target genome+text embedding；9 个有 exact accession 但现存资产无匹配，另 2 个未给
exact strain ID。现有表格每个 unseen species 最多只有一个有数据的 strain，因此只能支持
species-level zero-shot target discovery，不能支持 reviewer 要求的 broad species/genus efficacy。
下一步必须先由作者和 microbiology 团队选择临床 target 与多-isolate panel，再由外部 Evo-2/text
producer 生成并登记输入资产；在此之前不启动 guided sampler，也不修改 reviewer response 的结果段。

PepLink 外部依赖边界始建于 2026-07-19，并于 2026-07-21 升级到当前版本。作者维护的独立仓库
`DragonDescentZerotsu/PepLink` 已发布 PyPI `PepLink==0.1.2`、tag `v0.1.2` 和 commit
`90f627cc7fd65daaf9c5d0a973d17b79bcd097d5`。ApexOracle 不使用 submodule，也不复制其
chemistry core；仅保留 optional dependency、公开 API adapter 和版本/data SHA manifest。
独立 PepLink 0.1.2 测试为 23 passed。0.1.1 仍作为历史兼容性审计版本：179 条历史
structure correction 中 177 条输出逐字符串相同，另 2 条由其明确移除 legacy 游离 fragment，
全部 179 条均为 fragment-parent equivalent。论文复现继续消费 frozen paper CSV；新数据使用
0.1.2，且其 forward structure generation 与 0.1.1 相同。
AMP MIC parsing、in-house merge 与 SELFIES/token filtering 也已于 2026-07-19 完成。
canonical CLI 对原数据只读并拒绝覆盖输入；`paper_legacy` 显式保留 inhibition、censor 和
structure-correction 执行顺序。105,547 行 DBAASP MIC 的 ID/strain/SMILES 精确一致，MIC
最大绝对误差 `4.55e-13`；121,265 行合并表与 120,955 行 token cache 均逐字节一致。
tokenizer revision 已固定。PepLink 重建 15,718 条 in-house row 与 legacy 的唯一差异是
terminal `[OH]` 对 canonical `O`，归一化后全部一致；论文复现继续读取 frozen in-house
long table。旧 `DataPrepare/aa_seq_to_smiles.py` 的清理明确列为后续未完成事项：它仍被
`try.py`、`correct_SMILES_offered_by_DBAASP.py`、`APEX_in_house_to_SMILES.py` 和
`APEX_in_house_to_SMILES_merge_w_DBAASP.py` 四个 tracked legacy driver import，本批不删除。
`discription_generation.py` 与 `discription_generation_w_ATCC.py` 的重复清理同样推迟到
调用者迁移/归档之后。

PepLink round-trip 与历史 ChatGPT-o1/OPSIN chemistry 审计已于 2026-07-20 完成，当前状态为
`PepLink 0.1.2 released / reviewer Supplementary Data complete / corrected-data retraining pending`。
历史 0.1.1 在 16,430 个 curated
DBAASP peptide ID 中成功 forward 16,075 个，全部 16,075 个通过 SELFIES molecular-graph
round-trip；reverse contract cohort 的 annotation round-trip 为 3,729/4,939，全部 1,210 个失败
均由仅影响 reverse parser 的 Histidine tautomer template 不一致导致，不影响论文 forward data
generation。修复版 `PepLink==0.1.2` 已于 2026-07-21 通过 PR #4 合并，commit
`90f627cc7fd65daaf9c5d0a973d17b79bcd097d5`、tag `v0.1.2`，并发布到 GitHub Release 与 PyPI；
wheel/sdist SHA-256 已登记于 `configs/data_pipeline/peplink_v0.1.2.yaml`。23 项测试和正式版本完整
数据审计达到 4,939/4,939，其中支持的 head-to-tail cyclic peptide 为 523/523。reviewer response
直接汇报修复版结果，不展开 0.1.1 的历史 reverse-only mismatch。

AA 与 peptide source-aware 血缘已于 2026-07-21 重新核定。459 条 mapping = 39 条标准 L/D +
420 条 noncanonical；420 条的实际最终分支为 207 条保留 PubChem name lookup、44 条二次
GPT-refinement+OPSIN correction、169 条无 PubChem 命中的主 ChatGPT-o1+OPSIN branch。因此旧表述
“另外 251 条均未经过 GPT/OPSIN”已撤回；251 只是初次 PubChem lookup success 数，其中 44 条后来
进入第二条 GPT/OPSIN correction branch。169 条主 branch 的完整人工判定仍为 105 verified、14
verified/source formula typo、20 source ambiguous、22 pipeline output 错误、7 non-exact polymer
proxy、1 非完整 amino-acid definition；这份完整审计不能跳过 whole-peptide source lineage 而直接
当作训练错误范围。

frozen MIC 的完整结构来源为：DBAASP-linked PubChem CID whole-peptide 840 peptide/8,434 row，
local residue-based builder 15,521/96,747，DBAASP-offered structure branch 69/366；最终 union 为
16,430/105,547。PubChem 支路按 DBAASP record 已给定 CID 直接查询，不经过 ChatGPT-o1、OPSIN 或
PepLink。作者于 2026-07-22 确认 coordination omission 作为去金属/忽略配位预处理，不计为错误。
DBAASP sequence/unusual-residue annotation 内部不一致属于上游源数据质量和 historical producer
容错问题，不是 ChatGPT-o1/OPSIN 或 PepLink 转换错误，因而不进入 reviewer response 或论文修改。
应用实际 `<=512` token loader filter 后，reviewer-facing 确认错误为 56/15,177 peptide 与
219/74,103 MIC row（0.296%，reviewer 报告 0.30%）。该范围不包含 PubChem whole-peptide、仅因
当前 PepLink API unsupported 的记录或 polymer proxy；20 个
source-ambiguous definition 和二次 44 branch 的其他 source/site conflicts 继续单列为 unresolved。

旧 355-forward-failure 与 617/4,095 strict sensitivity 口径已退役，不进入 reviewer response。
原建议的 evaluation-only sensitivity 也不再作为当前交付：hierarchical MIC 历史结果没有保存带
DBAASP ID 的逐行 prediction，strain-wise 2025 精确 membership 未恢复；只删 held-out row 还无法
消除 training exposure。当前 reviewer 策略是报告 round-trip、完整血缘、0.30% 实测 prevalence
与 limitation，并在 successor dataset 修正或排除标记问题；不得由 0.30% 推导“no model effect”。
canonical 记录为 `experiments/peplink_validation/AA_AND_PEPTIDE_LINEAGE_ZH.md`、
`reviewer_response_scope_summary.json`、`revised_training_impact_scope.csv` 和
`reviewer_response_draft.md`。原 frozen paper CSV 保持不变用于复现。

2026-07-23 已完成 reviewer 承诺的 record-level Supplementary Data：英文
`experiments/peplink_validation/supplementary_data/Supplementary_Data_AA_conversion_errors.xlsx`
与中文镜像各含 56-row affected-peptide sheet 和 18-row definition summary；每行记录 historical
erroneous/corrected structure、formula、位置、错误理由、处置和证据。生成入口
`scripts/audit/build_peplink_supplementary_data.py` 会以 canonical loader 强制断言
56/15,177 peptide、219/74,103 row 和 18 definition，manifest 登记全部输入/输出 SHA-256。16 个
direct PubChem definition 已通过 PUG REST 再核对；`NNar` 与 `D-3-OH-ASN` 明确标为基于 DBAASP
name/formula 的中等置信度，未指定 stereochemistry 不作猜测。

正式 `sn-article.tex` 已在 ChatGPT-o1/OPSIN Methods 段落加入 source-aware audit 和
56/219 数字；`Response to reviewers letter.docx` 已用完成时态的两层验证精简回复替换旧 future-tense
草稿。TeX 临时编译成功为 28 页，DOCX 临时渲染成功为 25 页；正式论文 PDF 未自动覆盖。回复和论文
仍不包含 DBAASP sequence/annotation 数量或位置冲突、Leucine fallback 等已排除的上游脏数据内容。

发布仓库已按 reviewer-facing 边界整理：`.gitignore` 继续全局排除 CSV/XLSX，只精确放行正式
PepLink 0.1.2、化学审计、56-peptide Supplementary Data 和公开 Reviewer 4 unseen-target 表。
投稿 DOCX/PDF、私有 in-house workbook、0.1.1/dev exploratory output、退役 sensitivity 表及
一次性文档修改脚本不发布。`experiments/README.md` 与 `scripts/audit/README.md` 是统一入口。

作者于 2026-07-19 确认 Evo-2 producer 边界：当前仓库不重构 genome embedding extraction，
训练和评估直接消费预计算 tensor；未来整合后的 ApexOracle 主仓库可在 `external/evo2` 使用
固定 commit 的 Git submodule。该 consumer contract 已完成：567 个文件的逐文件 SHA-256、
字节数和消费状态已登记，三份论文数据匹配 563 个；只读 safe loader 和 reviewer scaling
完整重算通过，CSV/PNG 与现有结果逐字节一致。当前 Evo-2 checkout 为 dirty，因此本阶段只
记录 commit candidate，不把它加入当前仓库，也不声称它是已确认的精确 producer。下一阶段
进入 synergy CV 的代码迁移与历史协议核验。

Synergy CV 迁移已于 2026-07-19 完成，当前状态为
`paper high-confidence reproduction candidate / exact checkpoint identity unresolved`。候选
legacy driver 曾为 `synergy_Evo_train_new_reg_MDLM_one_base_model_classification.py`，输入
`synergistic_pairs_Evo.csv` 的 SHA-256 为
`ff57e2152159be950a9823f5d94f24c0771b18465bd17cfd11d9d0318db393be`。只读 adapter 已恢复
2,263 行 genome+text 与 469 行 text-only、合计 2,732 行的动态 eligible 数据。由于旧 split
依赖无序 `set` 和 taxonomy alias list 原地修改，候选 manifest 要求显式设置
`PYTHONHASHSEED`；在 `PYTHONHASHSEED=0` 下三个 fold 的四路原始行数与 2025 年日志完全一致：
`289/1974/727/31`、`2179/84/2436/212`、`2250/13/2538/181`。该不均衡划分属于历史协议，
本次重构保留并披露，不静默修正。tokenizer revision 已固定，pair Dataset/collate、对称双分子
forward、LoRA/base 初始化、held-out AUROC selection、checkpoint schema 和统一 runner 已抽取。
21 个 member 与 1 个 base 的 SHA-256 已全部登记；代表性真实 checkpoint 在 H100 上完成
genome+text 与 text-only 两路 inline legacy 公式逐值一致验证，fold 2 的真实数据 1-member/
1-epoch 完整 runner smoke 也已通过。

作者于 2026-07-19 进一步确认发布范围只覆盖论文中实际汇报的结果。论文 synergy 部分只汇报
2,732 个 eligible pair、strain-wise 三折、每折 7-member ensemble 的二分类结果（mean AUROC
`0.7539`、mean AUPRC `0.7454`）；正文和 Methods 没有 few-shot 结果。因此本阶段停止维护并从
工作树删除 early prototype、continuous-FICI、in-house evaluation、few-shot、all-data guidance、
prospective regression 和 BAA-3170 screening 的 root/canonical 源码、配置、专项测试与审计副本。
共享模块同时删除仅由这些路径消费的 tokenized dataset、regression target、online MDLM encoder、
guidance step 和 screening loader，避免留下无消费者的发布 API。

这次 paper-only 清理不删除或改写任何原始数据、checkpoint、日志或已有结果。legacy/root 源码可
从 `legacy-code-snapshot-2026-07-17` 恢复；此前完成的 post-paper canonical 迁移可从 merge commit
`e9a68d57e4b0b235906530d3ed389c66771141ce` 恢复。逐文件 SHA-256 和恢复命令登记于
`reproducibility/synergy_paper_only_cleanup_2026-07-19.json`。
清理后全仓库测试为 111 passed / 4 skipped；减少的测试仅对应已删除的 post-paper guidance、
regression 和 screening 路径。JSON/YAML 解析、核心 synergy 模块编译和引用扫描均通过。

作者随后确认将本机完整候选的 `0.7598/0.7440` 作为论文实现的高置信度复现候选；它与论文
`0.7539/0.7454` 的绝对差为 `0.0059/0.0014`。Methods 中 rank 64/13 epochs 与候选
rank 1024/100-epoch base 的差异继续作为披露边界，但不再阻塞重构完成。最后一个 root legacy
driver 已登记 SHA-256 后删除，由 legacy tag 恢复；synergy 发布代码只保留 canonical CV 入口。
node002 的已知 Synergy 工作目录也已只读复核：只保留六个早期 prototype，没有该 MDLM
classification driver 或对应 `strain_wise_synergy` checkpoint family。

Modality ablation 的绘图血缘已于 2026-07-19 冻结。最终 Mac notebook cell 中硬编码的四条
曲线、三个 holdout 粒度和 12 个精确 R² 已迁移到
`experiments/modality_ablation/paper_values.csv`，canonical 只读绘图入口为
`scripts/reproduce/plot_modality_ablation.py`。本机与 node002 的九个候选 driver SHA-256
完全一致，但两台机器的 checkpoint 仅为互补残片；代码、日志和 W&B output 均未找到完整
最终指标表，checkpoint 内 `R2` 也只是单成员 best score，不能反推出 ensemble 曲线。因此本项
状态为 `paper plot reproducible / legacy training rerun unavailable`。15 个旧训练 driver 均无
运行时消费者且会原地覆盖过滤后的中间 CSV；它们已在不运行的前提下登记 SHA-256 并归档删除。
受保护的 91 MB CSV 清理前 SHA-256 已冻结，原始数据、checkpoint 和结果未修改。
清理后 canonical 入口成功生成 1-page PDF 和 PNG，冻结值/配置/受保护 CSV 的 SHA-256 均未变化；
全仓库测试为 111 passed / 4 skipped。

2026-07-19 追加核验纠正了此前过于宽泛的“训练血缘未恢复”表述：四条 modality 曲线的
分子 encoder 均已由本机/node002 同源 driver 和真实 checkpoint 验证为
`DeepChem/ChemBERTa-77M-MTR`。三条无 small-molecule auxiliary task 的曲线依次来自
genome-only、text-only 和 genome+text driver；完整曲线来自对应 `DP_inhouse_SM_*`
driver。抽查的四类 checkpoint 均包含 55-key、`600 x 384` word embedding、3-layer 的真实
ChemBERTa state。仍未恢复的是“最终 12 个绘图点 -> 精确 member checkpoint -> held-out
prediction -> ensemble 聚合”的逐点链路；因此后续应写为
`encoder/driver family identified; exact plotted ensemble lineage unresolved`，不得再写成
模型家族未知。旧 full driver 可从 node002 和 `legacy-code-snapshot-2026-07-17` 恢复。

Synergy 关闭后的执行顺序如下：

1. **Modality ablation 收尾与 root 清理（已完成）。** 以已冻结的 12 个论文绘图值和 ChemBERTa-MTR
   driver-family 审计作为发布证据；先登记全部残留 genome/text/genome+text root driver 的哈希，
   再删除会原地覆盖中间 CSV 的旧训练入口，不伪造缺失的逐 checkpoint 血缘。清理 manifest 为
   `experiments/modality_ablation/legacy_cleanup.json`。
2. **Small-molecule classification 数据入口（已完成）。** 三个来源的历史转换规则已从
   `DataPrepare/DataCheck.ipynb` 恢复。canonical builder 明确固定 E. coli → A. baumannii →
   S. aureus 的 block 顺序，拒绝覆盖输入或既有输出；49,331 行 merge 与 49,330 行 token cache
   均完成逐字节复现，所有 raw/processed 文件 SHA-256 保持不变。tokenizer revision、唯一 UNK
   排除记录和完整数据 manifest 已冻结；旧 converter 由 legacy tag 保留后删除。classification
   runner 现在从 versioned config 显式读取 frozen token table。
3. **Guided generation 外部边界。** 对 `/data2/tianang/projects/discrete-diffusion-guidance`
   建立固定 commit、输入/输出和 checkpoint manifest；只在该外部仓库存在可验证 paper sampler
   时迁移或以 submodule/链接接入，不把论文后 synergy screening 重新引入本仓库。
4. **发布收口。** 建立论文 panel→命令/config/data/checkpoint/status 映射，重写主 README，明确
   `fully supported`、`high-confidence candidate`、`partially supported` 和 `missing/external`，
   然后完成 license、secret、绝对路径和大文件扫描。

三菌株小分子数据入口已于 2026-07-19 完成。已确认的三个标签规则分别是 E. coli
`Activity == Active`、A. baumannii `Mean < mean - sample std (ddof=1)` 和 S. aureus 直接复制
`ACTIVITY`。只读全量重建得到 49,331 行、1,112 阳性，输出 SHA-256
`4dabc0f8ac808d33ede3eacb47bacf7b55b2a900fcf78fd3d45a89c2037f3dc2`，与冻结论文 merge
逐字节一致。固定 IBM SELFIES tokenizer revision 后，49,330 行 token cache 也逐字节一致；唯一
排除的是含 UNK 的 `na_12751`。原 notebook 的未排序目录遍历/原地写回和旧 converter 的硬编码
路径/未固定 revision 不再作为发布入口。作者随后要求在 guided generation 前先完成 k-mer 消融迁移。
本阶段全仓库回归为 115 passed / 4 skipped；strict zero-shot group 0 的完整只读 dry-run 也以
`dry_run_ok` 结束，目标集在 512-token 过滤前后均为 2,335 行。

#### 4.1.1 Fig. 2c molecular encoder comparator 统一迁移（已完成）

状态（2026-07-19）：**代码迁移与验证完成。** 迁移范围为 strain-wise ChemBERTa-MTR、
ChemBERTa-MLM、MolFormer 和 PeptideCLM 四个复制 driver。已由源码验证的差异必须进入配置：

- 四者均使用 online tokenizer/model 和 first-token pooling，不能误接到 MDLM 预计算 feature；
- ChemBERTa-MLM 与 MolFormer 默认保持 backbone train mode，冻结 3 epochs 后解冻；
- ChemBERTa-MTR 与 PeptideCLM 显式保持 backbone eval mode，`freeze_epochs` 分别为 3000 和
  5000，在默认 25 epochs 内不会解冻；
- legacy 默认 ensemble 数分别为 1、7、7、1，batch size 均为 70，backbone optimizer group
  均使用 `3e-6` 和 `0.1 x weight_decay`；
- tokenizer、model revision、hidden size、checkpoint 目录和 payload key 必须由 encoder profile
  明确声明，不能用统一默认值覆盖。

上述差异现已通过 `configs/hierarchical_mic/legacy_fig2c_comparators.yaml` 接入现有 hierarchical
MIC split/fusion/evaluation outer runner。新增测试覆盖 raw-SMILES Dataset/collate、first-token
forward/logit/loss、optimizer group 顺序、freeze/unfreeze、八键 online checkpoint payload 与 loader
识别。四个固定 revision 均已在 H100 上完成真实 CUDA forward；MLM 与 MolFormer 还完成解冻后
backward。MTR、MolFormer、PeptideCLM 的现存 checkpoint encoder payload 已严格加载，node002
MLM checkpoint schema 与固定 revision 精确一致；真实数据 dry-run 成功。

四个 root 复制 driver 已从工作树删除，由 `legacy-code-snapshot-2026-07-17` 恢复。迁移只读取
原始 CSV、embedding 与 checkpoint，迁移前后四个源数据 SHA-256 保持一致。PeptideCLM 仍有明确
证据边界：本机 fixed 单 member 与 node002 早期 7-member driver 行为不同，且后者没有完整
checkpoint 网格；canonical 采用有严格 checkpoint 证据的 fixed profile，不声称论文精确 ensemble
血缘已恢复。完整审计见
`experiments/hierarchical_mic/strain/fig2c_comparator_migration_audit.json`。

#### 4.1.2 Fig. 2c strain-wise k-mer 消融（已完成）

状态（2026-07-19）：**代码迁移、真实 tensor 等价和血缘冻结完成。** 权限开放后确认
`/data/fangping/kmer_baseline` 是 2026 年 post-paper reconstruction，而不是论文精确历史源码。
论文最终绘图中的单模型 k-mer 为 R²/Spearman/Pearson `0.4507/0.6688/0.6793`；现存完整
global reconstruction 明确改为 25 epochs 和 7 members，三组 R² `0.3660/0.6308/0.5860`，
均值 `0.5276`，不能替代论文值。

新增 `apexoracle.features.kmer` 统一 global/windowed k=`4,5,6` producer，并提供拒绝覆盖非空
输出的 CLI；现存 reconstruction 的一次性 frozen random `5376→8192→8192` projection 已通过
可选 adapter 接入共享 hierarchical runner，新 checkpoint 会保存 projection state。567 个 global
和 567 个 windowed tensor 均已生成 SHA-256/shape/dtype manifest。E. coli ATCC 25922 的两种
canonical 重算与现存 tensor 均逐值相同。真实 group 0 dry-run 的 71,419 条 held-out 样本与历史
metrics 完全一致；group 1 当前动态 hash split 多 5 条，继续按既有 runner 契约标注为不恢复精确
历史 membership。仅迁移论文相关 strain-wise 路径，不迁移额外 phylum/species k-mer 实验。

完整限制和命令见 `experiments/kmer_ablation/README.md`。
本阶段新增 6 项 k-mer 测试后，全仓库回归为 125 passed；既有论文路径没有回归失败。

#### 4.2 迁移前需要作者或原始结果进一步核验

以下 Fig. 1b worker/ETA 条目是 2026-07-20--21 的历史运行记录；所有对应任务已经完成，
最终状态以本节后续 2026-07-22 完成记录和 `experiments/fig1b_antibiotic_classification/`
为准，不得据此重启 worker。

- phylum-wise 候选均值 `0.3844` 与论文 `0.3744` 相差 `+0.0100`；作者决定暂不追查；
- 11-cluster 合并两台机器后覆盖全部组，只有异常值 `-0.3467` 与绘图 `-0.1467` 不同；作者决定暂不追查；
- fine-tuned Fig. 1b 合并两机后为 104/150 个历史 checkpoint；作者于 2026-07-20 改为要求最终
  结果使用每个 outer fold 的完整 10-member ensemble。RN4220 fold 4 member 0 已按冻结协议补训，
  其余 45 个缺失成员最初分派到本机 3 张空闲 H100 和 node002 8 张空闲 A100，随后又由
  node001 8 张 A100 接管 node002 各队列的后排 member；新产物写入独立
  `results/fig1b_revision/full_ensemble_reconstruction/`，不覆盖历史 checkpoint；
  运行一小时后的持续采样确认 11 张卡均正常计算，node002 单次 0% utilization 是验证和 9 GB
  checkpoint 写盘间隙。由于 H100 约完成 7 epochs、A100 约完成 3 epochs，node002 八条队列的
  最后一个未启动任务（group 1 fold 4 members 1--8）已追加到 H100 后续队列；完成 prediction
  会同步到 node002，使原队列跳过对应任务，从而缩短尾部耗时且不移动大型权重；
- 2026-07-20 17:00 再次尝试启用本机 GPU1，但正式训练开始约 2 分钟后达到 90°C、显存温度
  88°C，并同时出现 NVIDIA hardware/software thermal slowdown。当前账号没有权限把 350W power
  limit 下调，因此该 worker 在首个 epoch 完成前终止；未完成目录已明确标记，node002 原队列仍是
  这四个成员的权威执行者。GPU1 不再计入可用训练资源；
- 新增只读监控入口 `scripts/reproduce/monitor_fig1b_revision.py`。它合并本机、node001 与 node002
  的 45 个
  缺失 member、当前 epoch、15 个 Chemprop fold、GPU 温度/利用率/热降频、checkpoint 占用和
  磁盘余量，并按活跃 worker 的实测 epoch 时间动态估计 Apex 阶段 ETA。17:00 快照为
  `62/1125` epoch units、吞吐量外推约 15.6 小时；这是 Apex 阶段的动态粗估，不包含尚未开始且
  尚无实测速率的完整 baseline 阶段；
- 2026-07-20 17:15 确认 node001 与 node002 共同挂载 `bright91:/data1`，release driver 的 inode、
  文件大小和 SHA-256 一致；node001 的 8 张 A100 80GB、Torch 2.7.1+cu126 和 Chemprop 1.5.2
  环境均通过验证。node002 每条四任务队列的第 3 个 member 已各迁 1 个到 node001；node001
  正式训练时 8 张卡利用率为 86--100%、温度 56--68°C。共享 `/data1` 只扫描一次，避免监控
  重复计算同一任务；
- node001 的 8 个 Apex worker 完成后分别继续执行 baseline folds
  `2/3, 2/4, 1/0, 1/1, 1/2, 1/3, 1/4, 0/4`。node002 尚未启动的 baseline session 已增加
  `metrics.json` 完成保护，后续 historical evaluation 等待链原样恢复。扩容后 17:15 动态 Apex
  ETA 从约 15.6 小时降至约 9.3 小时；完整实验 ETA 当时仍需首个 baseline fold 完成后
  才能可靠外推；
- 2026-07-21 14:11，Apex 补训已完成 `45/45`，完整 fine-tune 网格达到 `150/150`。停止本机
  两个已经由 node002 产出有效 prediction 的重复 Apex worker后，最后 5 个 RN4220/Wong baseline
  fold 已改为本机 GPU0/2/3 运行 fold 0/1/2、node001 GPU0/1 运行 fold 3/4。node002 后启动且
  会与 node001 并发写共享 fold 3/4 的 duplicate worker 已停止；node002 GPU0--7 改为并行完成
  已验证 checkpoint 存在的历史 ensemble inference。本机 Chemprop venv 已按 node 环境精确补齐
  `descriptastorus 2.7.0.3`，此前未生成 checkpoint 的 fold 2 已安全重启；原始数据与历史权重均
  未修改；
- Fig. 1b reviewer 修订已重新开启为独立补实验阶段：三个 strict zero-shot ensemble 的样本级
  预测和三个 Chemprop baseline 的 15 个共同-fold 运行均已完成。strict zero-shot 的 5,000 次
  paired bootstrap/randomization/Holm 结果显示只在 E. coli AUROC 上显著优于 baseline；
  A. baumannii AUPRC 相当而 AUROC 较低，RN4220 两项均显著较低。fine-tune sensitivity 已统一为
  每折单个 `ensemble_0`：14/15 个 fold 复用历史 checkpoint，RN4220 fold 4 已按旧 25-epoch
  协议补训并完成确定性推理。fine-tune 只在 E. coli AUPRC/AUROC 上显著优于 baseline；
  A. baumannii 和 RN4220 均无显著差异。作者随后确认这只能作为 sensitivity，不能作为最终
  结果；完整 ensemble 及对应 paired statistics 完成前，不再更新论文数值。
- Fig. 1b 旧柱子与当前 sensitivity 的指标口径诊断已完成：旧值来自不完整历史
  10-member ensemble 的 fold-level 汇总，而当前是 5-fold single-member pooled OOF；两者不得
  作为行为等价复现互相替代。单成员图仅作为诊断，最终图等待完整 ensemble。
- baseline 发布资产已核验：Stokes/Wong 各为 20-member ensemble，Liu 为 10-fold checkpoint
  directory。作者于 2026-07-21 决定最终共同-fold 公平比较不再照搬不同的发布 ensemble 大小，
  而是与 ApexOracle 对齐为三个 target 均使用固定 `model_0`--`model_9` 的 10-member ensemble；
  已完成的额外 checkpoint 保留但不进入最终 prediction。Liu 继续使用论文汇报的
  no-RDKit-feature ablation，不采用带 RDKit2D descriptor 的更强主模型。
- ApexOracle 15 个 fine-tune fold 的确定性 prediction 已全部完成，并由固定 assembly manifest
  验证每折恰好 10 members。共同 cohort 的 fold-mean AUPRC/AUROC 为 E. coli
  `0.71205/0.95884`、A. baumannii `0.43436/0.82200`、RN4220 `0.40127/0.95309`；正式
  paired significance 已在三个 10-member baseline 的 `15/15` folds 完成后重算。fine-tune 的
  三株 AUPRC/AUROC 共 6/6 比较均高于 baseline 且 Holm-adjusted `p < 0.05`；Fig. 1b 补实验
  计算阶段完成。Mac notebook 已于 2026-07-22 新增独立双 panel cell
  `fig1b-final-10member-dual-metric-20260722`：左 AUPRC、右 AUROC，显示五折 sample s.d. 和
  Holm-adjusted paired `p`；原 cell 未执行或修改，论文总图未覆盖。后续只需作者决定是否手工
  纳入论文总图并同步文稿。
- 作者随后新增 AUPRC-only cell 区域；已填写并执行独立 cell
  `fig1b-final-10member-auprc-only-20260722`，输出新的 PDF/PNG，原始与双指标 cell 均保持不变。
  reviewer response、建议图注、Results 替换文本和论文修改清单已写入
  `experiments/fig1b_antibiotic_classification/reviewer_response_auprc_final.md`。论文总图由作者手工
  整合。
- AUPRC-only 初版留白已按作者反馈压缩；最终 PDF 页面尺寸与旧论文 panel 精确一致为
  `741.12 × 380.724 pt`，同时保持所有 bracket/labels 完整且不修改其他 cell。
- 作者确认 `Methods / Data` 应只描述数据，并进一步删除先前迁入 `Implementation Details` 的
  Fig. 1b 细碎 protocol；Methods 最终不新增该实验的实现说明。TeX 仅更新结果概览、Fig. 1b 图注
  和 Results，实际 response-letter docx 已同步最终数值并保留指定的 Stokes/Liu/Wong common-fold
  说明。TeX 临时完整编译通过 28 页，Word 临时转换通过 25 页；未覆盖论文 PDF、figure 或总图。
- 作者明确禁止自动覆盖论文总图。已将 `Fig1.pdf` 恢复为更新前备份并删除自动拼接脚本；Mac
  原始 cell `220739609a526f79` 已从修改前备份逐字节恢复。reviewer sensitivity 已迁到新 cell
  `fig1b-reviewer-sensitivity-20260720`，并改为输出独立 panel。
- synergy CV 已作为论文高置信度复现候选完成；rank/base epoch 差异保留为 provenance 限制，
  不再是代码迁移阻塞项；
- modality ablation：最终绘图值和图形入口已冻结，但缺少能够把这些值精确连接到 legacy
  checkpoint、held-out predictions 和 ensemble 聚合过程的最终指标表。

除已由作者接受为高置信度候选的 synergy 外，其余证据不完整实验继续标为 partially supported，
不声称已找回精确历史结果。

#### 4.3 明确外部或缺失部分

- Evo-2 genome embedding 提取；
- DLM 预训练及部分 checkpoint；
- guided generation/remasking sampler（作者确认继续作为外部仓库，未来固定 clean commit 后再作为 ApexOracle submodule）。

在对应外部仓库完成重构前，仅提供接口、数据契约和缺失说明，不复制未经核验的大型代码树。

### 阶段 5：删除 legacy 和无关代码

完成迁移与最小验证后，从重构分支删除：

- [x] 已被 canonical 入口取代的 hierarchical MIC、Fig. 1b、sequence similarity、早期
  Fig. 2b 和 APEX 支持代码；
- [x] synergy、few-shot 与其余 in-house 副本：论文 CV 已迁入 canonical runner；作者接受本机
  完整结果为高置信度复现候选；所有 root、post-paper 和无共享消费者副本均已归档删除；
- [x] 已确认无消费者的 `_old.py`、debug、临时 notebook 和部分机器专用 launcher；
- [x] `Fangping_correlation/`、`e3nn_playground/`；
- [x] `GPU_eye.py`、`run.py`、`run_full.py` 等资源占用工具；
- [x] tracked capsule 重复源码；W&B 本地日志、缓存和旧结果继续由 `.gitignore` 排除；
- [x] 根目录 `train_on_all_data.py` 和五个 `fix_*` Fig. 2b driver：逐文件 SHA、用途和恢复点已
  冻结，canonical consumers 验证后删除；
- [x] `compare_APEX/`：最佳 checkpoint 与 AAindex 已迁到 canonical ignored 资源位置，其余
  checkpoint/W&B 完整外移到 legacy archive；
- [x] `PeptideCLM/`：精简为 `src/apexoracle/vendor/peptideclm_tokenizer/` 并完成 token 等价验证；
- [x] 未上传的 168 GB `capsule/` 派生 staging 删除；`capsule_fig2/` 改为正式 35-fold 轻量审计包；
- [ ] 仍服务于外部 DLM preprocessing 的机器专用 shell，以及证据未冻结的 launcher。

执行进度（2026-07-19）：已完成第一批发布清理，共删除 59 个 tracked 文件、未删除任何数据、
checkpoint 或结果。范围包括 11 个 `Fangping_correlation` 旁支文件、4 个 e3nn 教程、3 个 GPU
占卡/监控工具、11 个被取代的 `compare_APEX` Python/launcher、6 个早期 Fig. 2b driver 和
24 个可由 builder 重建的 `capsule*/data/source` 副本。精确清单和恢复点位于
`reproducibility/release_cleanup_2026-07-19.json`。

删除 `compare_APEX` 源码前，实际消费的 APEX encoder、AAindex loader、23-token adapter、masked
loss 和 per-task R² 已迁入 `src/apexoracle/benchmarks/molecule_encoders/`。真实 checkpoint
`strict=True` 加载，固定四序列的 legacy/canonical `(4,128)` feature SHA-256 完全相同。2026-07-20
最佳 checkpoint 已迁到统一 `weights/` 根并通过 manifest ID 解析，AAindex 改为 ignored reference
asset；其余历史 checkpoint 与 W&B 完整外移，未删除。synergy、modality 和 Fig. 2c root driver
均已在证据冻结和恢复点登记后删除。

执行进度（2026-07-20）：上述 pre-generation root cleanup 已完成，精确 SHA、迁移路径、删除边界
和恢复点见 `reproducibility/pre_generation_cleanup_2026-07-20.json`。本阶段没有修改论文原始数据，
没有删除训练 checkpoint。PeptideCLM 只 vendor 必需 tokenizer 资源并保留 MIT LICENSE；APEX
AAindex 因非营利研究许可边界保持为 ignored 本地 reference asset。

验收标准：根目录不再堆积实验副本；每个保留脚本都能对应公开文档中的明确任务。

### 阶段 6：验证

- [x] Python compile/import 检查。
- [x] 数据单位转换、label transform、strain mapping、APEX 序列投影测试。
- [x] 共享交集和 fold 无泄漏测试。
- [x] mask、tensor shape、pooling 和指标测试。
- [x] 已迁移 canonical 主入口的小规模 smoke test。
- [x] 对可用旧 checkpoint 运行小样本等价性检查。
- [x] staged 密钥、绝对路径、超大文件和未跟踪实验产物扫描。

本阶段默认不重新训练全部论文实验。Fig. 2b 公平 benchmark 是例外：完成代码和数据协议验证后，需要正式重新训练并将其作为 reviewer 要求的新结果。

执行记录（2026-07-19）：hierarchical MIC 统一迁移新增了 legacy/shared 逐项等价测试，覆盖四种
collate、Dataset lookup、MIC label transform、feature filename parsing/loading、strain mapping、
cross-attention、regression head、split mutation 语义和两类 checkpoint 顶层 payload。代表性
9.08 GB checkpoint 已通过 H100 固定批次验证。后续单批次训练测试又覆盖四种 modality/task
组合，并逐参数验证 gradient、历史 clipping 目标和 Adam step 后的参数一致性；H100 上另以
有限合成 GradScaler scale 验证 CUDA autocast 路径逐位一致。正式 driver 的默认动态 scaler
保持不变。这里仅表示
strain/species/phylum 的共享子路径通过本批验收，
不等同于阶段 6 的全仓库验证已经完成。

同日 Fig. 1b 三菌株分类新增 11 项 CPU 测试和 1 项 H100 集成测试，覆盖目标 frame、
512-token 后 KFold、molecule ID 导出、full-fusion/molecule-only forward、optimizer step、
module mode、AUROC/AUPRC、best tracker 的 AUPRC 保存顺序、两种 checkpoint schema 和文件名。
strict group 0 / ensemble 0 的真实 checkpoint 又与 capsule 在 batch size 70 下完成 2,335 条
logit 逐值一致验证。

Synergy CV 本批新增 8 项 CPU 测试，覆盖 FICI label、双分子 token filter、Dataset/collate、
分子顺序对称、BCE forward、LoRA-only checkpoint schema 以及 base/member round-trip。真实
H100 checkpoint 两路 forward 与 1-member/1-epoch runner smoke 另行通过。当时全仓库为
103 passed / 3 skipped；三个 skipped 均为沙箱内不可见 CUDA
时跳过的测试，其中本批新增 CUDA test 已在宿主 H100 单独通过。该计数是 2026-07-19 的
历史快照，不是当前测试总数。

同日 APEX support migration 新增 3 项测试，覆盖 inline legacy forward、masked loss/R² 公式、
真实 AAindex checksum、真实 checkpoint strict load 和固定 feature hash；另以 100-row、5-fold、
1-epoch CPU cached runner 完成端到端 smoke。10-row 诊断会因部分 fold 没有 finite task R² 触发
旧 runner 的 missing-best-summary 边界，本阶段为保持 selection 行为没有修改该语义。

同日 all-data synergy guidance 新增行为测试，覆盖两条 route 的 tokenized Dataset/collate、
alias merge 与完整 row membership、`g1→t1→g2→t2` / `g1→g2→t1→t2` 调用顺序、dead CPU RNG
消耗、对称 forward、optimizer step 和七键 full-state checkpoint contract。真实数据 dry-run
再次得到 `2320→2213` 与 `2789→2635`，输入及两份历史 checkpoint SHA-256 未变化；两个 4.1 GB
checkpoint 的 H100 strict-load/forward 均通过，40-epoch fixed batch 与独立 inline legacy 公式
逐值一致。完整训练默认写到 `results/`，不会覆盖历史产物，并要求显式确认 post-paper metric
边界与动态 legacy row order。该批次收尾时全仓库回归为 119 passed / 5 skipped；本批 CUDA
optimizer-step 已在宿主 H100 单独通过，两份 guidance checkpoint 的 independent inline legacy
差异均为 0。

最终状态（2026-07-23）：当前 tracked canonical 范围为 `147 passed`；PepLink 0.1.2 独立仓库
为 `23 passed`。本阶段的完成声明覆盖已迁移 canonical 论文/审稿入口，不表示
`DataPrepare/` 中每个 legacy 或探索脚本都已具备 smoke/checkpoint 验证。

### 阶段 7：文档和发布

- [x] 更新 README：研究目标、安装边界、数据准备、模型资源、复现命令和当前状态。
- [x] 在 `AGENTS.md`、`experiments/` README 和 manifest 中提供论文图表到
  command/config/checkpoint/data 的映射。
- [x] 在 README、`AGENTS.md` 和各实验审计中区分 fully supported、partially supported 与
  missing/external。
- [ ] 为 MIC prediction 提供最小 quickstart。
- [x] generation 在外部 sampler 整合完成前明确标注不可端到端复现。
- [x] Fig. 2b 当前正式修订已完成：正文、图注和 reviewer response 已更新公平 benchmark 数值；完整 `Fig2_2.pdf` 已换入 10,886 个共享分子的七模型结果和五折 sample s.d. error bars，并经渲染核对；最新 TeX 已完整编译为 28 页。当前图文对应 24-layer joint 正式结果。12-layer joint 容量匹配实验属于后续核验；若采用其结果，仍须同步更新图、正文、回复信和相对提升。
- [ ] 确认 license、第三方模型许可、数据再分发条件和 citation。
- [x] 持续用中文维护 `AGENTS.md`，记录新的审计结论和迁移关系。

#### 2026-08-02 ReMDM reviewer capsule 收束状态

- [x] 完成 peptide-guidance/remasking 补实验、结构审计、canonical reviewer figure、正式 Methods/
  Supplementary Fig. C4 和三处 reviewer response。
- [x] 建立 `experiments/remasking_schedule_reviewer/PUBLICATION_HANDOFF.md`，登记四个相关项目路径、
  GitHub remotes、白名单/禁止项和防膨胀策略。
- [x] 核验本轮 canonical 代码/测试约 142 KB，当前未忽略 reviewer capsule 为 414,113 bytes，
  约 0.41 MB；raw runs、
  日志、逐 attempt 表、权重和重复图稿不进入 Git。
- [x] 按作者决定在当前 `Synergy/main` 上用显式路径分为四个主题提交并直接推送；没有使用
  `git add -A` 或创建 PR，推送前全仓库测试为 `164 passed`。
- [x] ApexOracle-specific Generation clean fork 已发布到
  `DragonDescentZerotsu/ApexOracle-Generation`；默认 `main` 固定候选 `de6c1e5`，source-only recovery tag、
  paper GPU smoke、remote fresh-clone release audit、14 tests 和双 strain dry-run 通过，上游未被推送。
- [x] 公开 `ApexOracle-MDLM` module-level clean release 已进入默认 `master`，固定候选 `c9d17c7`；发布只使用
  `custom`，未误推上游。全仓 118 tests、14 项跨仓库 source contracts
  passed；五个正式 MIC-guidance checkpoints 另已完成 schema/strict-head audit，Generation 正式
  padding-preserved regression GPU/bfloat16 output exact；
  checkpoint/embedding、heads、candidate scoring、interpretability、Fig. 3a、通用 screens/embedding producer
  已迁移。Hugging Face 18-file clean release 已完成，固定 revision
  `77694f08c1d0664fdb24c5a7bab130c8a3bc2eda` 已通过 fresh-cache strict-load/GPU smoke。旧 HF duplicates 与
  peptide-classifier trainers 已完成 clean migration，source archive fresh install/import/CLI smoke 已通过；
  六个 MIC guidance trainers 已归并为五 profiles 并删除 root copies；11 个 hierarchical drivers 已移交
  Core 后删除；两个 chemistry utilities 已迁为通用 API 并通过 11,401-row byte parity 与 5,887,458-row
  catalogue full-scan parity。三个 synergy-guidance root producers 已归并为两个 experimental profiles，
  正式 encoder/candidate GPU parity 与 checkpoint schema 通过并删除旧 copies。远端 shallow clone、wheel/
  install/import/CLI、118 tests、显式资产 20 checks 和 recovery tag fetch 已通过；MDLM module candidate 已
  就绪；full Generation integration 与 Core compatibility bridge 删除均已完成。
- [ ] 从 public legacy `ApexOracle/DLM_pretrain/` 准备
  `DragonDescentZerotsu/ApexOracle-DLM-Pretraining` 独立历史；作者要求原则上不改代码。先原样归档并做
  synthetic train/save/load smoke；只有 blocking portability failure 才允许最小化修补路径/文档/config，
  不改模型结构、objective 或训练行为。
- [ ] 从官方 Evo-2 基线准备 `DragonDescentZerotsu/ApexOracle-Evo2` clean fork commit：upstream
  `53f1959` 上的本地 candidate `ccdbfbe` 已加入通用 extraction CLI，9 CPU tests、567-FASTA plan-only、
  clean build 与 wheel fresh CLI 通过；40B runtime、public remote/fresh clone 尚待完成。
- [ ] 将当前 Synergy 原 repository 整理并重命名为 `DragonDescentZerotsu/ApexOracle-Core`，
  保留现有 history/内部结构并完成公开边界审计；不得复制第二个 Core repo。
- [x] 作者已决定直接把现有 public `DragonDescentZerotsu/ApexOracle` 原地转换为 super-repo，以五个
  固定 commit 的 Git submodule 组合 Core/DLM-Pretraining/MDLM/Evo2/Generation；不再新建 super-repo，
  legacy 状态由同仓 tag/branch 恢复。详细计划见 `docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md`。

#### 2026-08-09 统一 ApexOracle super-repo 固定计划

- [x] 固定目标 repository topology、module 职责、submodule 策略、canonical URL 切换方式和
  PepLink 独立依赖边界。
- [ ] R0：冻结四个来源 checkout 加 public legacy `DLM_pretrain/` 的 source inventory、SHA-256、
  科学角色和非破坏恢复点。Downstream MDLM 与 Evo-2 的 source-only 恢复点已完成，其余来源仍待执行。
- [ ] R1：完成 Core/DLM-Pretraining/MDLM/Evo2/Generation 五个可独立安装和 smoke-tested 的
  clean commits。MDLM 与 Generation 两项已完成；Evo2 已有 CPU-validated local candidate，
  Core/DLM-Pretraining 待执行。
- [ ] R2：现有 ApexOracle 已原地转换，恢复 branch/tag、`.gitmodules`、module/asset manifests、bootstrap、
  CI 与 MDLM/Generation 固定 gitlinks 已完成；Core/DLM-Pretraining/Evo2 三个 gitlinks 待各自 clean candidate。
- [ ] R3：完成新 molecule × known strain 的 MIC prediction end-to-end quickstart。
- [ ] R4：完成 target strain guided generation 的 smoke/paper-preset end-to-end quickstart。
- [ ] R5：完成 model-ready data、strain texts、许可、完整 source archive 和 fresh-clone QA。
- [ ] R6：在同仓保留 legacy tag/branch、原地转换默认分支、发布 tag/Release 并同步论文/HF/Zenodo 链接。

阶段验收、禁止项和 reviewer-facing 命令固定记录于
`docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md`；执行时必须逐项更新这里的状态。

#### 2026-08-07 hierarchical MIC censor-multiplier sensitivity

- [x] 新增共享 raw-censor lineage 与 frozen-prediction metric 重算逻辑
  `src/apexoracle/evaluation/hierarchical_mic_censor_sensitivity.py`，不复制训练 runner 或模型实现。
- [x] 新增单一 CPU/只读入口
  `scripts/audit/analyze_hierarchical_mic_censor_sensitivity.py`；默认冻结普通右删失
  `1×/2×/4×`、删除右删失、删除全部删失五种 sensitivity，并记录输入/输出 SHA-256。
- [x] 使用现有 fixed strain-wise 和 canonical phylum-wise 七成员 ensemble predictions 完成
  172,182条 held-out measurement instances 的分析。mean-across-groups R² 在 strain
  `1×/2×/4×` 下为`0.5785/0.5813/0.5634`，phylum 为`0.3804/0.3879/0.3748`；删除右删失为
  `0.5699/0.3491`。
- [x] 从42 MB逐行 lineage/prediction 表独立复算60个 metric rows；R²/MAE/RMSE/Pearson 最大差
  `<=2.2e-16`，Spearman 最大 CSV round-trip 差`5.03e-10`；focused tests 为28 passed，全仓为
  196 passed（14条既有 warnings）。
- [x] 将逐行表设为 local-only，只发布 compact counts/metrics/deltas/manifest；入口、参数、输出、
  claim boundary 已同步到 experiment README、`scripts/audit/README.md`、根`AGENTS.md`和计算资产图。
- [x] 完成R²机制诊断：量化17--18%局部log-label平移以及SSE/TSS同向变化的抵消；exact SQLite
  decomposition 的精确数值与解释已保留在 experiment README。2026-08-09 维护审计确认原
  466 KB portable HTML/JSON 没有 canonical 生成入口且未被正式文稿消费，已移出工作区至
  `/data2/tianang/.codex-trash/20260809_synergy_mic_multiplier_diagnostics/` 可恢复归档，不再将其
  列为正式产物。
- [x] 作者已确认按英文 plan 草稿正式落稿：在 DBAASP MIC preprocessing table 前后紧凑加入
  representative `>V` sensitivity 的方法、分母、`V/2V/4V`、exclusion 指标与 heuristic 结论，并将
  reviewer DOCX 的 future-tense 占位回复替换为三段完成时结果。正式 TeX/DOCX 修改前均已创建
  `before_mic_multiplier_sensitivity_20260808` 备份；TeX 独立编译为35页且无 undefined
  citation/reference，Methods 位于第10--11页；DOCX独立渲染为32页，回复位于第24--25页并已目视
  核验。正式 `sn-article.pdf` 未覆盖。对外表述仍限定为 evaluation-label sensitivity without
  retraining，不声称 censor-aware training robustness。
- [x] 2026-08-09 完成代码/文件系统维护审计：将623行 canonical audit script 收缩为118行 CLI，
  新增 `src/apexoracle/evaluation/hierarchical_mic_censor_workflow.py` 统一重建、prediction 对齐、
  output contract 与 manifest，保留纯 label/metric module；删除未使用的 `censor_counts`。补充
  multiplier/output contract、in-memory MIC audit columns 和 auxiliary-column compatibility tests。
  同一冻结输入 canonical `--overwrite` 重跑后六个CSV与manifest的SHA-256全部与重构前一致，31项
  focused tests和全仓205 tests通过（14条既有warnings）；42 MB逐行表因承担独立复算与重复记录
  审计而继续local-only保留。

## 5. 计划提交序列

建议使用小而可审查的提交：

1. `chore: import sanitized legacy source snapshot`
2. `chore: add release-oriented project scaffold`
3. `refactor: extract shared data and evaluation modules`
4. `feat: add shared molecule encoder benchmark protocol`
5. `refactor: migrate final paper experiment entrypoints`
6. `test: cover data splits metrics and model interfaces`
7. `chore: remove superseded and unrelated legacy code`
8. `docs: add reproducibility and release documentation`

每个提交完成后都更新本文件的复选框和 `AGENTS.md` 中受影响的血缘说明。

## 6. 风险与停止条件

出现以下情况时不得自行猜测并继续发布：

- staged 内容中发现无法确认是否已撤销的真实密钥；
- GitHub 远程仓库包含用户已有且本地不存在的提交；
- 准备删除的实现无法从 Git tag 或用户备份中恢复；
- 某个“最终版”与论文数值、Methods 或 checkpoint 结构出现新的实质冲突；
- 公平 benchmark 无法为某个 comparator 构造确定且可审计的输入；
- 数据或模型的许可不允许公开再分发。

这些情况应记录为“仍待作者确认的事项”，并在执行相关发布或删除动作前与作者确认。
