# 计算节点、代码版本与资源位置

> 最后核验：2026-07-21。动态运行状态以
> `python scripts/reproduce/monitor_fig1b_revision.py` 为准；本文记录稳定职责和路径，不把瞬时
> GPU utilization 当作实验完成证据。

## 1. Canonical 代码与同步规则

| 位置 | 工作树 | 用途 | Git 操作 |
| --- | --- | --- | --- |
| 本机 `sn4622119311` | `/data2/tianang/projects/Synergy` | canonical 开发、测试、文档和本机 H100 运行 | `.git` 是只读保护挂载，使用 `git --git-dir=.git-state --work-tree=.` |
| node001 / node002 | `/data1/tianang/Projects/Synergy_release` | 共享 release runner 和 reviewer 补实验 | 两台机器看到同一个 `bright91:/data1` 工作树，只能在其中一台执行一次 Git 同步 |
| node002 历史目录 | `/data1/tianang/Projects/Synergy` | 2025 代码、checkpoint、日志和数据来源 | 只读审计；不得为了同步 release 代码而修改或覆盖 |
| GitHub | `DragonDescentZerotsu/Synergy` 的 `main` | canonical 远程版本 | 每个大阶段完成后 push；节点工作树只允许 fast-forward 到 `origin/main` |

node001 与 node002 上 `Synergy_release` 的 inode/文件内容已经交叉核验一致。因为它们共享 NFS，
不要在两台机器上同时运行 `git pull`。节点 release 工作树中的 `DataPrepare/Data` 是有意保留的
运行时 symlink，不是 Git 源码：

```text
/data1/tianang/Projects/Synergy_release/DataPrepare/Data
  -> /data1/tianang/Projects/Synergy/DataPrepare/Data
```

版本核验命令：

```bash
# 本机
git --git-dir=.git-state --work-tree=. rev-parse HEAD
git --git-dir=.git-state --work-tree=. rev-parse origin/main

# node001 和 node002 共享同一结果，只需运行一次
ssh node002 'git -C /data1/tianang/Projects/Synergy_release rev-parse HEAD'
```

三个值必须一致。`results/`、checkpoint、数据 symlink 和运行日志不参与 Git 同步。

## 2. 机器职责与环境

| 机器 | GPU / 当前约束 | 应在此处运行的任务 | 环境 |
| --- | --- | --- | --- |
| 本机 `sn4622119311` | 4×H100 80GB；GPU0/2/3 可用于当前训练；GPU1 达到 90°C 并触发 thermal slowdown，本轮禁用 | canonical 单元测试、代码重构、数据/血缘审计；Fig. 1b H100 队列；`mdlm` 本地任务 | 默认 `/home/tianang/anaconda3` 的 `base`；`/data2/tianang/projects/mdlm` 必须用 `mdlm` conda env |
| `node001` | 8×A100-SXM4-80GB | node002 后排 Fig. 1b member、随后 8 个完整 baseline fold | `/data1/tianang/anaconda3` 的 `base`；Chemprop venv `/data1/tianang/Projects/.venvs/fig1b-chemprop-v1` |
| `node002` | 8×A100-SXM4-80GB | 原始 Fig. 1b 队列、剩余 baseline、历史 checkpoint 确定性推理；历史资产只读审计 | 与 node001 相同；不要复制共享 `/data1` 产物 |
| SSH alias `Mac` | 论文绘图 | 只在新 notebook cell 生成独立 panel；论文总图由作者手工编辑 | `/Users/kirianozan/Documents/anaconda/anaconda3` 的 `base`；canonical notebook `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb` |

本机 Fig. 1b Chemprop venv 是
`/data2/tianang/projects/.venvs/fig1b-chemprop-v1`。两套 Chemprop 环境均固定为 Chemprop
1.5.2、Torch 2.7.1+cu126、NumPy 1.26.4、pandas 2.2.2、scikit-learn 1.8.0 和
RDKit 2025.03.5。

## 3. 当前 Fig. 1b 补实验

### 已验证协议

- fine-tune 网格是 `3 strains × 5 outer folds × 10 members = 150 checkpoints`；104 个历史
  checkpoint 已恢复，RN4220 fold 4 member 0 已补训，因此本轮只补 45 个 member。
- 每个成功 member完整运行 25 epochs，没有 early stopping。目标菌株的四个 folds 进入训练；
  第五个 held fold 每个 epoch 都用于 strict highest-AUROC checkpoint selection。没有独立
  validation set，这是为保持论文时期行为而冻结的已知局限。
- 每个 fold 的 10 个 member 使用相同 fold membership、不同配置种子。保存最佳 checkpoint 后，
  当前流程另以全 `eval()` 模式加载 checkpoint 并导出确定性 prediction。
- 当前 ApexOracle 不在线运行 molecule encoder。启动时一次性加载 768 维 float32 cache：
  - peptide：`DataPrepare/Data/Pep_emb_dict_cls_wo_pad_eval.pt`，18,029 entries，60,959,487 bytes，
    SHA-256 `c08630860efb87e97fa9955b08db526499b16208f7be1a89e7fa0b2346a9ff7b`；
  - small molecule：`DataPrepare/Data/SM_emb_dict_cls_wo_pad_eval.pt`，49,330 entries，
    166,844,850 bytes，SHA-256
    `5d2e2f4d7b9c7287764e44f84dc7977b60cd1c0c6a7317fd91189e91bc5ede83`。
- `fine-tune` 在这里指训练 strain-aware attention、regression/classification heads 和 missing-genome
  parameter，不更新 DLM/MDLM molecular encoder。Genome 与 strain-text embedding 也使用预计算
  tensor。

### 运行位置与产物

| 内容 | 位置 / 说明 |
| --- | --- |
| 本机新 Apex member | `/data2/tianang/projects/Synergy/results/fig1b_revision/full_ensemble_reconstruction` |
| node001/node002 新 Apex member | `/data1/tianang/Projects/Synergy_release/results/fig1b_revision/full_ensemble_reconstruction` |
| 完整 no-RDKit baseline | 共享 release 工作树的 `results/fig1b_revision/baselines_full_ensemble_no_rdkit`；本机分到的 folds 在本机同名目录 |
| GPU1 热记录 | 本机 `results/fig1b_revision/gpu1_thermal.csv` |
| 机器可读审计 | `reproducibility/fig1b_reviewer_revision_2026-07-20.json` |

node001 的 Apex session 为 `fig1b_apex_node001_gpu0..7`，完成后对应
`fig1b_baseline_node001_gpu0..7` 自动接续。node002 原队列是 `fig1b_apex_node_gpu0..7`；其
baseline session 已设置 `metrics.json` 完成保护，在共享文件系统中发现 node001 已完成 fold 时
直接跳过。不要手工删除仍在运行目录中的 checkpoint 或 `driver.log`。

### 2026-07-21 负载重排快照

- **已验证事实：** Apex fine-tune 网格已达到 `150/150`，本轮缺失的 `45/45` member 均已有
  prediction。因任务启动时只检查本机目录，本机 GPU0/2 上仍各有一个已被 node002 完成的重复
  member；核验 node002 对应 JSON 可正常解析后，已停止这两个重复 worker，未删除本机中间
  checkpoint。
- **已验证事实：** 最后 5 个 Chemprop fold 为 RN4220/Wong 2024 profile。node001 已为
  fold 3/4 生成额外的 20-member 权重，最终 prediction 只消费前 10 members。fold 0/1/2
  当前分别由 node001 GPU0/1/2 从干净输出目录训练 10 members。node002 曾在共享 `/data1` 上
  重复启动 fold 3/4；发现后已停止 node002 的后启动进程，只保留更早的 node001 owner，避免继续
  并发写同一目录。作者随后决定三个 baseline 均与 ApexOracle 对齐为 10-member ensemble；
  fold 0/1/2 已按 10 members 重启，fold 3/4 和 E. coli 的现存目录只选择
  `model_0`--`model_9` 做最终 prediction，额外权重保留但不参与指标。
- **已验证事实：** 本机 Chemprop venv 已从 node 环境的 producer commit
  `9a190343bcd3cfd35142d378d952613bcac40797` 补齐 `descriptastorus 2.7.0.3`；此前因缺少该依赖
  失败且没有生成 model checkpoint 的 fold 2 已在 GPU3 重新启动。这里的 RDKit2D 是 Wong 2024
  profile；作者要求的 Liu 2023 no-RDKit ablation 仍为 `features_generator: null`。
- **已验证事实：** 历史 checkpoint 实际分散在本机和 node002；最初基于宽松文件名 pattern 的
  node002 可用性检查产生了误判。精确文件名核验后，node002 只负责其实际拥有的 E. coli folds
  2--4 和 RN4220 fold 2，其余推理已放回本机。失败尝试没有改写权重，最终输出只接受成功返回且
  JSON 明确登记 ensemble indices 的运行。
- **根据现有证据作出的推断：** node001 三个 fold 当前首先进行 CPU/RDKit feature 预计算，所以
  短时 GPU utilization 为 0% 不代表 worker 空闲；node002 历史推理在 embedding 加载和分批推理
  间也会低利用率。当前已经分配完所有互不冲突的独立任务，继续拆分单个 Chemprop ensemble 会
  改变现有 runner/聚合方式，本轮不这样做。
- **2026-07-22 已验证事实：** RN4220 folds 0--2 已在 node001 完成，三个 baseline 达到
  `15/15` folds，所有 Fig. 1b GPU/tmux worker 已退出。canonical baseline summaries 位于共享
  `/data1/tianang/Projects/Synergy_release/results/fig1b_revision/baselines_full_ensemble_no_rdkit/`；
  三份 `summary.json` 与 OOF prediction 已同步到本机同名 ignored 目录用于最终 paired statistics。
  不再重启任何 Fig. 1b 训练 worker。
- **2026-07-22 已验证事实：** Mac canonical notebook 新增 cell
  `fig1b-final-10member-dual-metric-20260722`，左侧绘制 AUPRC、右侧绘制 AUROC；输入冻结为
  `experiments/fig1b_antibiotic_classification/final_10member_dual_metric.csv`。输出使用新的
  `3-strain-antibiotics-final-10member-dual-metric.{pdf,png}` 文件名，原 cell、旧 panel 和论文总图
  均未改写。
- **2026-07-22 已验证事实：** 作者新增的 AUPRC-only 区域已填写为
  `fig1b-final-10member-auprc-only-20260722`，输出
  `3-strain-antibiotics-final-10member-auprc-only.{pdf,png}`。该 cell 只消费冻结 CSV 的 AUPRC
  行；原始 cell、双指标 cell、旧 panel 和论文总图均未改写。
- **2026-07-22 已验证事实：** AUPRC-only layout 已按作者反馈压缩 legend 下方留白；新版 PDF
  页面尺寸为 `741.12 × 380.724 pt`，与旧论文 panel 精确一致。该修订只改 AUPRC-only cell。
- **2026-07-22 已验证事实：** 作者最终决定 Methods 不加入 Fig. 1b 的细碎实验流程；small-molecule
  数据段保持纯数据说明，先前迁入 `Implementation Details` 的详细 protocol 已删除。TeX 只更新
  结果概览、Fig. 1b 图注和 Results，response-letter docx 已同步最终 10-member AUPRC 数值及
  Stokes/Liu/Wong common-fold 说明。TeX/Word 分别在 `/tmp` 编译或转换核验，没有覆盖论文 PDF、
  figure 或总图。

统一监控：

```bash
cd /data2/tianang/projects/Synergy
watch -n 30 python scripts/reproduce/monitor_fig1b_revision.py
```

监控中的 `105/150` 是本轮开始前可用 checkpoint 网格；`0/45` 一类数字只表示新补 member 的
完成数。短暂 0% GPU utilization 常见于 held-fold evaluation 或约 9GB checkpoint 写盘，必须结合
进程、日志和连续采样判断。

## 4. 数据、权重和外部仓库

| 资源 | Canonical 位置 | 边界 |
| --- | --- | --- |
| 论文原始/冻结数据 | 本机 `DataPrepare/Data`；节点通过上述 symlink 读取历史目录 | 只读消费；重构输出写 `results/`，不得原地覆盖 |
| Reviewer 4 私有 in-house AMP 表 | `DataPrepare/Data/private_inhouse_amp/Master_List_Peptides_Antimicrobial_Activity.xlsx` | Git ignored；SHA-256 `956ff3d60364a113c2149a63b74b65d4f81ec03741fefbac1841558ed54744a4`；不得上传 GitHub |
| 本仓库权重登记 | `configs/model_weights.yaml`、`MODEL_WEIGHTS.md` | 二进制不进 Git；新路径使用 `${APEXORACLE_WEIGHTS_DIR:-weights}` |
| DLM/MDLM producer | `/data2/tianang/projects/mdlm` | 必须用 `mdlm` conda env；当前 checkout 有本地修改，不能直接作为 clean submodule |
| Evo-2 producer | `/data2/tianang/projects/evo2` | 当前仓库只消费 `DataPrepare/Data/Genome_embs`；不重跑 extraction |
| guided generation | `/data2/tianang/projects/discrete-diffusion-guidance` | sampler、DLM pretraining 同属外部边界；当前只读审计，不并入 Synergy |
| PepLink | `/data2/tianang/projects/PepLink` | clean 独立仓库；`PepLink==0.1.2` 已发布到 GitHub/PyPI，ApexOracle 新数据入口固定消费 0.1.2，论文复现仍读取 frozen CSV，不复制源码 |
| AA/peptide reviewer 审计 | `experiments/peplink_validation/` | canonical 中文血缘为 `AA_AND_PEPTIDE_LINEAGE_ZH.md`；机器可读范围为 `reviewer_response_scope_summary.json`；56-peptide Supplementary Data 位于 `supplementary_data/`；不得用已退役的 forward-failure union 代替 source-aware scope |
| 论文 TeX / reviewer response | `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech` | Codex 不自动替换论文 PDF；作者决定何时采用新 panel |

发布 Git 仓库只包含 reviewer-facing 审计代码、公开派生表和 manifest。投稿 TeX/DOCX/PDF 继续位于
独立 manuscript checkout；私有 in-house workbook 及其派生 assay 表继续受 `.gitignore` 保护。
`experiments/README.md` 与 `scripts/audit/README.md` 分别是结果和审计入口索引。

截至本次核验，外部 checkout 为：`mdlm` HEAD `7a6a7d1`（dirty）、guidance HEAD `edb0f8c`
（dirty）、Evo-2 HEAD `afd0dae`（dirty）、PepLink HEAD `90f627c`（clean；tag `v0.1.2`，
GitHub Release 与 PyPI 均已发布）。这些 SHA 只记录当前
候选状态，不反向证明 2025 论文运行的精确 producer commit。

### AA/SELFIES reviewer 当前任务

- **已验证事实：** PepLink 0.1.2 已完成 16,075/16,075 structural SELFIES round-trip、
  4,939/4,939 annotation reverse contract 和 523/523 supported head-to-tail cyclic round-trip。
- **已验证事实：** frozen MIC structure 分为 DBAASP-linked PubChem CID whole-peptide、local
  residue builder 和 DBAASP-offered structure 三条来源。PubChem whole-peptide 不经过
  ChatGPT-o1/OPSIN/PepLink，因此当前 forward reconstruction failure 不能作为其训练结构错误证据。
- **已验证事实：** 作者确认 coordination omission 是去金属/忽略配位预处理，不计错。实际
  `<=512` token genome-or-text DBAASP pool 中 reviewer-facing 确认错误为 56/15,177 peptide 与
  219/74,103 row（0.296%，报告 0.30%）。DBAASP sequence/unusual-residue annotation 内部不一致是
  上游 source-data quality，不属于 ChatGPT-o1/OPSIN/PepLink 转换错误，也不进入回复或论文修改。
  polymer proxy 不纳入该口径；旧 forward-failure union 已退役。
- **根据现有证据作出的判断：** 0.30% 可描述为低 prevalence，但不能证明 zero model impact。
  evaluation-only deletion 不能清除 training exposure。
- **2026-07-23 已验证事实：** 英文/中文 Supplementary Data 已生成，覆盖 56 个 peptide 和
  18 个错误 definition，并逐条给出 historical erroneous/corrected structure、formula、位置、错误理由、
  处置和证据。生成脚本重新调用 canonical loader 并断言 219/74,103 row；manifest 登记输入/输出
  SHA-256。正式 TeX 已在 ChatGPT-o1/OPSIN 段落加入审计结果，正式 reviewer-response DOCX 已替换为
  完成时态精简回复；临时渲染分别为 28 页和 25 页，论文 PDF 未自动覆盖。
- **仍待作者/chemistry coauthor 确认：** 20 个 source-ambiguous definition、第二条 44-residue
  branch 的 source/site conflicts，以及 successor dataset 的最终 corrected/excluded 处置。

### Reviewer 4 unseen-species 当前任务

- **已验证事实：** `scripts/audit/audit_reviewer4_inhouse_species_coverage.py` 只读上述私有 Excel 和
  canonical training assets；aggregate 输出与执行 notebook 位于同一 ignored 目录。审计得到 35 个
  normalized workbook species、11 个实际训练未见 species、其中 10 个有 MIC 数据；guidance
  producer 的实际 exposure 为 1,599 个 strain ID / 389 个 producer-era species 名。
- **已验证事实：** 11 个候选中没有任何一个已同时具备 exact-target genome 与 text embedding。
  9 个候选有表头 accession 但两个 embedding 目录均无匹配；`[Clostridium] symbiosum` 和
  `Staphylococcus capitis` 没有 exact strain ID，后者还没有任何 MIC 测量。
- **根据现有证据作出的推断：** 当前 in-house 表可用于选 target 和已有 peptide counter-screen，
  但每个 unseen species 只有一个有数据的 strain，尚不能构成 broad species/genus efficacy panel。
- **仍待作者确认：** 选择临床 target 和多-isolate panel；随后才由外部 Evo-2/text producer 创建
  exact-target embeddings，并在外部 guidance repo 重构完成后启动 generation。

## 5. 发生变化时必须同步更新

以下任一变化都必须同时更新本文件、根 `AGENTS.md` 和 `REFACTOR_PLAN.md`：

1. 新增/停用 GPU 或更换机器；
2. 数据、embedding、checkpoint 或外部仓库迁移；
3. GitHub `main` 与节点 release 工作树不再 fast-forward 对齐；
4. Fig. 1b 队列、ensemble 数、fold、epoch、selection 或 baseline feature 协议改变；
5. Mac notebook、论文图编辑边界或 submodule 决策改变。

无法由代码、hash、日志或实际 mount 验证的内容必须标成“推断”或“待作者确认”，不能写成事实。
