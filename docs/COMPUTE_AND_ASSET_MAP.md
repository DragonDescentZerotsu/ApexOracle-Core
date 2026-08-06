# 计算节点、代码版本与资源位置

> 最后核验：2026-08-02。Fig. 1b 与 ReMDM remasking schedule reviewer 补实验均已完成；
> node002 原 GPU guard 已恢复。
> `python scripts/reproduce/monitor_fig1b_revision.py` 仅用于只读核验历史产物与节点状态；
> 本文记录稳定职责和路径，不把瞬时 GPU utilization 当作实验完成证据。

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
| 本机 `sn4622119311` | 4×H100 80GB；GPU1 有 90°C thermal-slowdown 历史，重新使用前必须复核 | canonical 单元测试、代码重构、数据/血缘审计；经确认后的新 H100 任务；`mdlm` 本地任务 | 默认 `/home/tianang/anaconda3` 的 `base`；`/data2/tianang/projects/mdlm` 必须用 `mdlm` conda env |
| `node001` | 8×A100-SXM4-80GB | 后续经确认的 release runner/GPU 任务；Fig. 1b 历史任务已完成 | `/data1/tianang/anaconda3` 的 `base`；Chemprop venv `/data1/tianang/Projects/.venvs/fig1b-chemprop-v1` |
| `node002` | 8×A100-SXM4-80GB | 历史 checkpoint 确定性推理和只读资产审计；Fig. 1b 历史任务已完成 | 与 node001 相同；不要复制共享 `/data1` 产物 |
| SSH alias `Mac` | 论文绘图 | 只在新 notebook cell 生成独立 panel；论文总图由作者手工编辑 | `/Users/kirianozan/Documents/anaconda/anaconda3` 的 `base`；canonical notebook `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb` |

本机 Fig. 1b Chemprop venv 是
`/data2/tianang/projects/.venvs/fig1b-chemprop-v1`。两套 Chemprop 环境均固定为 Chemprop
1.5.2、Torch 2.7.1+cu126、NumPy 1.26.4、pandas 2.2.2、scikit-learn 1.8.0 和
RDKit 2025.03.5。

### node002 非科学 GPU guard（2026-07-27）

- **已由进程、日志和连续 GPU 采样验证的事实：** 作者授权在 node002 历史工作树中运行非科学
  资源工具。owner 为 `tianang`，tmux 会话为 `synergy_gpu_guard_20260727`，命令为
  `/data1/tianang/anaconda3/bin/conda run --no-capture-output -n base python -u run_full.py
  --gpus 0,1,2,3,4,5,6,7`，工作目录为 `/data1/tianang/Projects/Synergy`，日志为同目录
  `gpu_guard_node002_20260727.log`。
- `run_full.py` 只在单卡显存低于250MB时启动一个独占该卡可见性的 `run.py`。当前 `run.py`
  默认保留90%显存，并以4096方阵 BF16 GEMM、约2ms高频 burst 实现目标8% compute duty。
  连续15次每秒采样为每卡约7--8% utilization、约73.3GB显存和30--33°C；这只能标记为
  resource guard，不能作为模型训练或实验运行证据。
- 历史目录通常只用于只读审计；本次是作者明确授权的单文件运行例外。原文件保存在
  `run.py.backup_20260727_memory_only`，原/当前 SHA-256 分别为
  `b1372f5d82234e92856f837989b76f4036aea9c868e1c5041e2631ceea375a3b` 和
  `4c217c488bc41f5a4dc238e79c285adb24c8ca8677dbc38e2dff7d0134933993`。不得把该修改当作
  2025 producer 源码，也不得同步到 `/data1/tianang/Projects/Synergy_release`。
- 查看与停止：

  ```bash
  ssh node002 'tmux attach -t synergy_gpu_guard_20260727'
  ssh node002 'tmux kill-session -t synergy_gpu_guard_20260727'
  ```

### ReMDM remasking schedule reviewer 实验（2026-07-28--29，已完成）

- **已由 GPU 查询和 task manifest 验证的事实：** 本机 4 张 H100 与 node002 8 张 A100
  共同承担 `experiments/remasking_schedule_reviewer/task_manifest.json` 中 36 个独立任务。
  每卡唯一 owner 为一个 host-local orchestrator queue，每卡顺序运行 3 个 task；本机负责
  12 tasks，node002 负责 24 tasks。每个 task 写唯一
  `experiments/remasking_schedule_reviewer/runs/<task_id>/`，以 `completed.json` 和逐 batch
  SHA-256 为完成条件，不存在跨 worker 共写 batch 文件。
- **本机 owner：** tmux session `remasking_reviewer_local_20260728` 已正常完成并退出；
  12/12 tasks、48/48 batches、1,200/1,200 raw attempts 均有 completion marker，4 张 H100
  已释放。运行环境为 `/home/tianang/anaconda3/envs/mdlm`，外部 sampler
  `/data2/tianang/projects/discrete-diffusion-guidance` 只读导入。
- **node002 staging：** 运行资产隔离在
  `/data1/tianang/Projects/remasking_schedule_reviewer_assets/`，包括从本机按文件复制的
  DLM、v1 peptide classifier、guidance regressor 与 `conda-pack` 环境；正式启动前必须完成
  size/SHA-256、`conda-unpack`、CUDA import 和单卡 smoke。代码只同步本次新增的 runner、
  orchestrator 和 manifest 到历史 Synergy 工作树，不执行 Git 更新。
- **已验证的 node002 staging 身份：** DLM、v1 classifier、guidance regressor SHA-256 分别为
  `a509b94e...2615`、`40f638ca...45b`、`f24faf67...3a4`，与本机逐项相同；环境 archive 为
  `26db6a7e...f740`，解包后版本为 Torch 2.3.1+cu121、flash-attn 2.6.3、Hydra 1.3.2、
  Lightning 2.2.1、Transformers 4.49.0 和 RDKit 2024.09.6。两菌株的 genome/text embedding
  四份文件也与本机逐项同 hash。
- **producer 一致性修正：** 两台机器原外部 guidance checkout 虽然 HEAD 相同，但 dirty
  `diffusion.py`、`models/dit.py` 和 `configs/config.yaml` 内容不同。没有修改 node002 历史
  checkout；正式 node002 任务改用
  `/data1/tianang/Projects/remasking_schedule_reviewer_assets/producer/` 中本机 smoke-tested
  只读快照。其关键文件 SHA-256 与本机逐项相同：
  `diffusion.py=929a2e41...df5f`、`classifier.py=5b7b5b65...1129`、
  `models/dit.py=058d97e8...48a3`。
- **guard 边界：** 资产 staging 期间保持 `synergy_gpu_guard_20260727` 运行；完成上述
  CPU/文件核验后才按作者授权停止 guard、确认 8 卡显存释放、运行 smoke 并启动正式队列。
  正式队列全部完成或失败退出后必须恢复原 guard session，并重新核验 8 张卡约 7--8%
  utilization 与约 73.3GB 显存。停止/恢复的实际时间和核验结果记录如下。
- **实际启动状态：** guard 于 `2026-07-28T23:23:17-04:00` 停止；停止前每卡
  73,277--73,373 MiB、7--8%，8 秒后降至 18--114 MiB、0%。随后 GPU7 上的
  BAA-3197 batch-1 smoke 完成，输出为 complete 且 RDKit-valid。正式 node002 owner 为 tmux
  `remasking_reviewer_node002_20260728`，8 queues 已启动，共负责 24 tasks；状态文件为
  `/data1/tianang/Projects/Synergy/experiments/remasking_schedule_reviewer/runs/queue_status_node002.json`。
- **实际完成与恢复：** node002 24/24 tasks、96/96 batches、2,400/2,400 attempts 完成后，
  结果无覆盖同步到本机并与本机 1,200 attempts 合并；36 个任务、144 个 batch 的 size/SHA-256
  均通过 completion-marker 复核。正式 evaluator 完成 3,600 个 v1 peptide scores 和 2,355 个
  finite clean-MIC predictions。guard 于 `2026-07-29T00:04:21-04:00` 恢复；复核为每卡
  73,277--73,373 MiB、7--8% utilization、29--32°C，tmux
  `synergy_gpu_guard_20260727` 正常运行。
- **冻结协议：** 6 conditions × 2 strains × 3 seeds × 100 raw attempts，共 3,600；
  windows 为 `0.75--0.65`、`0.55--0.45`、`0.35--0.25`、`0.525--0.475` 和
  `0.55--0.25`，另有 current-window `gamma_peptide=0` direct control。正式结果不得加入
  manifest 外的 exploratory ablation。
- **2026-07-29 structure audit GPU 记录：** 本机只读核验确认4张H100均空闲后，临时使用
  GPU0 和 `mdlm` 环境复算历史 v1 full-token/first-`[SEP]`-padded probabilities，并应用两个
  reviewer-retrained heads；运行结束后 GPU 释放，无 tmux、无常驻 worker、无 checkpoint
  写入。canonical 命令登记在
  `scripts/audit/audit_remasking_peptide_classifier_structure.py`，compact 输出位于
  `experiments/remasking_schedule_reviewer/analysis/peptide_structure_audit/summary.json`。
  本次只读审计没有停止或修改 node002 guard。

### ReMDM reviewer 正式文稿资产（2026-08-02，已完成）

- 正式 TeX 和 response DOCX 位于
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/`；TeX 已加入
  Supplementary Fig. C4 及对应 Methods 修改，response 已更新三处 remasking 相关回答。作者最终
  决定不在 Results 保留 window/effectiveness 两段细节；相关结果位于 Supplementary Fig. C4/
  caption 和 reviewer response。
- 正式图文件为 `Fig_SI_remasking_schedule.pdf`，与 Synergy canonical 源图
  `experiments/remasking_schedule_reviewer/figures/remasking_structure_qualified_peptides_with_mic_control.pdf`
  的 SHA-256 均为
  `23ce3a58f82b82f1fb1f458efd08152fbf1284f9b2e2f6b97cd2e1030e9bb847`。
- TeX 只在 `/tmp` 独立编译核验为 31 页，没有覆盖正式论文 PDF；response DOCX 修改前备份为
  `Response to reviewers letter_before_remasking_revision_20260802.docx`，独立渲染核验为 29 页。
- 本次只有文稿和 figure asset 变更，没有新增 GPU 任务、移动 checkpoint/data、修改 node002
  guard 或改写外部 sampler。

### Guided-generation diversity reviewer 审计（2026-08-02）

- CPU-only canonical 入口为
  `MPLBACKEND=Agg PYTHONPATH=src /home/tianang/anaconda3/bin/python
  scripts/audit/analyze_generated_candidate_diversity.py`，在本机 base 环境运行，不占用 GPU，
  不影响 node002 guard。
- 外部只读输入位于
  `/data2/tianang/projects/discrete-diffusion-guidance/outputs/{generated_mol_SELFIES_w_mic-new,generated_mol_SELFIES-new}`、
  `/data2/tianang/projects/mdlm/smiles_to_peptide.py`、`/data2/tianang/projects/PepLink` 和
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/ApexOracle_MIC_data/Summary_pathogens.xlsx`。
- Canonical corrected 73-row candidate copy、compact pairwise/plotted data、figure、summary 与
  SHA-256 manifest 写入 `experiments/generated_candidate_diversity/`；不改写外部 sampler 历史
  output。Generation-level scope 显式限制为 `BAA-3170/3197`，不纳入同目录的 107 条
  `BAA-1556` 历史输出。
- Exact generation all-pairs owner 为 node002 CPU task，隔离目录为
  `/data1/tianang/Projects/generated_candidate_diversity_20260802/`。输入 fingerprint cache 为
  84,226 × 2,048-bit、约 21 MiB，SHA-256
  `a2b0a9789670057665ad124cb5c0fd9105c70030a809909c1a0b907074c764bf`；64 workers 穷举
  3,546,967,425 pairs 的 compute/total time 为 21.72/22.62 秒，实际吞吐约 163.3 million
  pairs/s。输出 histogram/summary 已按 SHA-256 同步回本机实验目录。该任务只使用 CPU，未停止、
  修改或占用 node002 GPU guard。
- **2026-08-03 final-peptide pairwise diversity：** 本机 base 环境 CPU-only 运行
  `scripts/audit/analyze_selected_peptide_diversity.py`；无 GPU、node002 或外部 producer 写入。
  输入为 `experiments/generated_candidate_diversity/canonical_candidates/` 下冻结的 mapping 和
  73-row structures，compact 输出为 `selected_peptides_24/` 下 168-row sequence pairs、276-row
  Tanimoto pairs、两份 24 × 24 matrices、nearest-neighbor CSV、PID/Tanimoto 双 panel violin
  plot、`target strain × topology` stratified 87-pair summary/plot、summary/manifest。已验证 24 个
  distinct structures、median PID `0.1719`、median Tanimoto `0.4633`；解释边界见该目录
  `REPORT.md`。
- **2026-08-03 Supplementary Fig. C5 三 panel 组图：** 本机 base CPU-only 入口为
  `scripts/audit/plot_generated_candidate_diversity_figure.py`，只读消费冻结的 selected-peptide
  PID CSV 和 Tanimoto plotted-data CSV，不重算 generation fingerprints。输出单行 a/b/c 图到
  `experiments/generated_candidate_diversity/generated_candidate_diversity_three_panel.*`；canonical
  PDF 与正式 `Fig_SI_generated_candidate_diversity.pdf` SHA-256 均为
  `72c50e1649720233a3a45ba46d57d21df8fa68ffb2660aba886b3a4f491a8ab8`。
- **正式文稿落稿状态：** 正式 TeX 已加入 selection denominator、peptide/small-molecule
  prioritization、intended-target hit rate、generated-set diversity Results/Methods，以及将 Fig. 3a
  限定为 predicted distributions 的 *in silico* comparison、将 prospective MIC 定义为 selected
  candidates 的直接实验评价。Canonical 图已复制为
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Fig_SI_generated_candidate_diversity.pdf`
  （当前三 panel SHA-256 `72c50e1649720233a3a45ba46d57d21df8fa68ffb2660aba886b3a4f491a8ab8`）。修改前
  TeX 备份为 `sn-article_before_generated_diversity_selection_20260802.tex`；`/tmp` 独立编译为
  32 页且图号为 Supplementary Fig. C5，没有覆盖正式 `sn-article.pdf`。正式 response DOCX
  已合并本轮 selection/diversity replies，并回指完整 Supplementary Fig. B2 MIC matrices；
  2026-08-03 又同步 C5a 的 87-pair PID 结果到 diversity 与两条 selection 回复，整体口径为
  selected-panel/candidate-pool/generation-output 三层；
  最终 SHA-256 为
  `5da0fc9894c861733917438b07904cb22777f800cceece7185b067b0464473bc`，最新修改前备份为
  `Response to reviewers letter_before_selected_pid_c5_20260803.docx`，独立渲染为 30 页并完成相关
  页面核验。当前 canonical 回复草稿位于
  `experiments/generated_candidate_diversity/reviewer_response_draft.md`。
  2026-08-03 Methods 进一步按作者要求重排为独立的 candidate-prioritization 与
  sequence/structural-diversity subsections，PID 定义移至首次使用之前并合并重复 topology 说明；
  修改前备份为 `sn-article_before_methods_reorganization_20260803.tex`。独立编译仍为 32 页，
  第 16--17 页已目视核验，正式 `sn-article.pdf` 未覆盖。
  发布白名单审计后，21 MiB fingerprint cache、81 个可重建的逐长度 SELFIES provenance files
  和 superseded diagnostic plots 均保持 local-only；canonical compact tables/manifests/docs、最终
  三 panel C5、入口代码与测试可进入 Git。此调整不移动或删除本地/外部资产。
- **Supplementary Fig. B2 全部实验候选 heatmap（2026-08-02）：** 作者提供的 Mac
  `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/Fig_SI_heatmap_re_v4.pdf` 已只读导入正式
  `Fig_SI_heatmap_re.pdf`，两者 SHA-256 均为
  `5ac3bd00e52958ecb06bc066e29de4752863b0f31d851e18df531954c1ae2693`。旧 4-candidate 图备份为
  `Fig_SI_heatmap_re_before_all_candidates_20260802.pdf`。正式 TeX 使用双栏 `figure*` 和
  `\textwidth`，展示全部 24 peptides 与 18 small molecules 的 20-strain MIC matrices；修改前
  TeX 备份为 `sn-article_before_all_candidate_heatmap_20260802.tex`。`/tmp` 独立编译为
  32 页，B2 位于第 23 页且 C3--C5 编号不变；正式 `sn-article.pdf` 未覆盖。

### ReMDM reviewer 发布交接与 remote 审计（2026-08-02）

- 本轮 canonical reviewer 代码、紧凑结果和内部说明位于 Synergy；8 个脚本/测试约 142 KB，
  当前未忽略 reviewer capsule 共 414,113 bytes，约 0.41 MB。实验目录约 67 MB 的主要占用为本地
  raw runs、日志和图片，均受 `.gitignore` 保护，不需要删除取证资产，也不得进入 staged set。
- 完整路径清单、提交白名单和统一公共仓库方案见
  `experiments/remasking_schedule_reviewer/PUBLICATION_HANDOFF.md`。用户决定把当前 reviewer 轮次
  按主题拆分后直接推送 `Synergy/main`；已完成四个主题提交，没有创建 fork 或 PR。
- `Synergy` remote 为 private `DragonDescentZerotsu/Synergy`；发布时使用显式路径分组 stage，
  没有使用 `git add -A`。推送前全仓库测试为 `164 passed`，本地与远程 `main` 已对齐。
- `mdlm` 同时有上游 `kuleshov-group/mdlm` 和公开自有 remote
  `DragonDescentZerotsu/ApexOracle-MDLM`；本轮没有修改该 checkout。其当前 branch 跟踪上游，
  未来发布必须显式推向自有 remote。
- `discrete-diffusion-guidance` 目前只有上游 remote，不存在自有同名 GitHub fork；当前 dirty
  producer 不得直接推送。应先建立 clean fork/独立 repo、参数化绝对路径并固定可复现 commit。
- public `DragonDescentZerotsu/ApexOracle` 的现有 legacy history 约 235 MiB，且已混入数据、模型和
  外部仓库副本，不适合作为继续复制本轮 raw 资产的目标。未来统一发布应采用清理后的 release
  history/layout，并以固定 commit/submodule 或版本化依赖连接 MDLM 与 guidance producer。

### 当前 reviewer 轮次 GitHub 防膨胀检查（2026-08-02）

- 在准备按主题提交当前 reviewer 轮次时，发现 peptide-classifier split 的五个本地 memmap
  (`*.u1`/`*.u8`) 合计约 911 MB，其中 `molecule_hashes.u8` 约 662 MB。这些文件是
  `prepare_peptide_classifier_split.py` 的确定性中间产物，不是 reviewer-facing 结果，并已按目录
  加入 `.gitignore`。
- Git 只保留 `split_manifest.json`、`split_audit.json`、紧凑训练/评估结果、代码和测试；manifest
  继续记录本地 memmap 的 size/SHA-256。该边界避免可重建缓存进入永久 Git history，同时保留
  split 复核能力。

### 新 reviewer GPU 任务调度约束（2026-07-26）

- 作者要求需要 GPU 的新实验尽量同时使用本机当前可用的 3 张 GPU、node001 的 8 张 A100 和
  node002 的 8 张 A100；启动前仍须逐机核验进程、显存、温度和环境，实时不可用设备不得强行
  加入。
- 科学协议保持不变，只按可独立重现的 `protocol/group/fold/ensemble` 单元并行。每个任务必须
  登记 machine、GPU、owner、完整命令、输入版本、输出目录和完成条件。
- node001 与 node002 共享 `/data1`。两台机器可以并行消费同一只读代码和数据，但不得写同一个
  checkpoint、日志、临时文件或汇总文件；输出路径必须包含唯一任务标识，最终汇总由单一 owner
  在所有原子完成标记就绪后执行一次。
- 本机与 `/data1` 不共享结果目录。跨机器汇总前必须按 manifest 核对任务键、文件大小和
  SHA-256，不能只按文件名或数量判断完成。
- 当前拟开展的 hierarchical MIC peptide-overlap audit 是 CPU/只读任务；后续
  molecule-disjoint checkpoint replay 或 deterministic companion rerun 才进入上述 GPU 调度。

### Genome-condition reviewer 实验（2026-08-04，已完成）

- **已由实时只读查询验证的资源事实：** 本机 4 张 H100 PCIe 均约 81 GiB free、0% utilization；
  node001 的 A100 GPU 0/3 有其他 Python workload，1/2/4/5/6/7 空闲；node002 8 张 A100 均已有
  vLLM service 占用约 76 GiB，因此本轮不抢占 node002 GPU。当前实现和全部数据位于本机 checkout，
  swap replay 按唯一 owner 分配给本机 GPU0--3；node001/node002 只在各自 `/local` 上用 CPU 从
  原 fixed strain checkpoint 提取 inference-only payload，不写共享 checkpoint 或 replay 结果。
- **GPU owner：** `experiments/genome_condition_reviewer/task_manifest.json` 冻结 GPU0=group0
  ensembles 0--6、GPU1=group1 ensembles 0--6、GPU2=group2 ensembles 0--3、GPU3=group2
  ensembles 4--6。每个 member 写唯一 CSV/JSON 到
  `experiments/genome_condition_reviewer/replay/`；21/21 completion 后仅由本机 root task 汇总一次。
  实际执行中 group1/2 的 14 members 先完成；原 GPU0 group0 串行 worker 在写出任何 member 前停止，
  随后 group0 按 `0--1/2--3/4--5/6` 重新分配到 GPU0--3。该调度变更、未产生输出的 stopped
  worker 和最终命令均登记在 task manifest，科学协议与 member ownership 未改变。
- **权重与工具位置：** inference-only 权重统一中转到
  `/data2/tianang/tmp/genome_condition_reviewer_inference_checkpoints/`，是带原 source size/SHA-256
  的可重建大文件，不进入 Git；`skani 0.3.1` 安装在
  `.external/envs/genome_condition_reviewer/`，base 环境未修改。
- **完成事实：** 21/21 swap members 与两个五折 probes 均已完成，GPU worker 已退出，本机四张
  H100 已释放。Swap donor cohort 为 292 个 target-fold pairs、42 species、67,794
  measurements，ANI median/range 为 99.57%/95--100%；effective train/test 无重叠。Probe 精确
  兼容子集为 162 bacterial genomes、58,717 fragments，AMR/MGE-associated positives 为
  722/4,520。Correct/swapped pooled R² 为 `0.419880/0.420045`，paired MAE delta bootstrap
  95% CI 为 `[-0.000142, 0.000158]`，是近零效应；AMR/MGE OOF AUPRC 为
  `0.2010/0.3863`，对应 sampled prevalence 为 `0.1667/0.1866`。完整边界见
  本地忽略的内部分析目录。最终 focused tests 为 5 passed，全仓为
  176 passed（14 条既有 warning），artifact/format/`git diff --check` 均通过。
- **文稿决策：** 作者已决定 swap replay 仅保留为内部诊断历史，不用于 reviewer response 或论文；
  上述 162-genome probe 也被后续 264-genome saved-window probe 取代。正式证据使用新 probe
  与全 embedding homologous-fragment analysis。

### Bacterial genome/text 四条件 replay（2026-08-05，已完成）

- **实时资源事实：** 启动前本机 GPU0--3 均为约 81 GiB free、0% utilization；因此没有跨节点复制
  数据或占用 node001/node002。推理阶段四卡各约使用 27.6 GiB。
- **cohort 与条件：** 268 bacterial target strains、36 species、64,646 measurements；target 与
  nearest same-species donor 均未出现在对应 fold 的训练 frame。四条件为 correct/correct、
  donor-genome/correct-text、correct-genome/donor-text、donor/donor，不重训任何参数。
- **GPU owner：** GPU0/1/2 分别负责 group0 ensembles `0--1`/`2--3`/`4--5`；GPU3 先负责
  group0 ensemble 6，再串行完成 group1/2 全部 ensembles。唯一输出目录为
  `experiments/genome_condition_reviewer/condition_replay/`，完整命令和 completion contract 见
  `condition_task_manifest.json`；21/21 完成后仅由 root task 汇总到 `condition_analysis/`。
- **完成事实：** 21 CSV + 21 JSON completion markers 构成完整 3×7 grid；汇总器验证每行七成员、
  metadata 一致且预测有限。Correct/genome-only/text-only/whole-condition pooled R² 为
  `0.429559/0.429742/0.410544/0.410647`，MAE 为
  `0.550197/0.550100/0.561588/0.561517`；genome-only paired ΔMAE CI 跨 0，而 text-only 与
  whole-condition CI 均为正。四张 H100 已全部释放，最终数值与解释边界见
  本地忽略的 `condition_analysis/`。最终 focused tests 为 7 passed，全仓为
  178 passed（14 条既有 warnings），artifact/format/JSON/`git diff --check` 均通过。
- **文稿决策：** 该四条件 replay 同样不用于 reviewer response 或论文。

### Evo-2 homologous-fragment variation（2026-08-05，已完成）

- **计算与工具：** 本机 CPU 16 workers；未占用 GPU。`minimap2 2.31-r1302` 位于项目隔离 prefix
  `.external/envs/genome_condition_reviewer/`，`edlib 1.3.9.post1` 位于
  `.external/python/genome_condition_reviewer/`。Canonical 命令和 owner 见
  `experiments/genome_condition_reviewer/fragment_variation_task_manifest.json`。
- **Saved-tensor compatibility：** 受测试的 window reconstruction 与 567/567 tensor shapes 一致，
  其中 370 个为 multi-record FASTA。结果只代表 saved fragment condition，不外推到其余 sequence。
- **pair preparation：** `prepare_all_genome_fragment_variation_pairs.py --threads 64` 使用本机 CPU
  与项目隔离的 `skani 0.3.1`。567 个 embedding/FASTA/species 交集中有 539 个 bacterial assets；
  379 个来自 71 个 multi-strain species。360 个 genomes 通过 ANI/coverage 最近邻阈值，去重为
  255 个 unordered pairs、67 species，ANI median/range 为 `99.36%/95.05--100%`。
- **完成事实：** 255 个 pairs 全部处理；166 pairs/53 species 产生 6,625 个分析 homologous
  fragments，其中 4,649 个含变异。Variable-fragment pooled sequence-divergence/cosine-distance Spearman 为
  `0.6954`，ANI `>=99%` 子集为 `0.7137`；99.94% homologous fragments 比同 donor genome 的
  deterministic random fragment 更近。24,605 raw rows 与 6,625 analysis rows 的 finite/schema/
  uniqueness/summary contract 已通过。原 185-pair strain-wise pilot 已移动到
  `fragment_variation/strainwise_pilot/`。最终 focused tests 为 10 passed、全仓为 181 passed
  （14 条既有 warnings）；完整表述边界见
  `experiments/genome_condition_reviewer/RESULTS.md`。
- **Historical-window probes：** 本机CPU依次运行
  `prepare_historical_genome_annotation_probes.py`和`run_historical_genome_annotation_probes.py`，
  未占用GPU。563个paper-matched embedding IDs中264个bacterial genomes、96,716 fragments通过
  saved-window精确兼容。AMR/MGE positives为
  1,217/8,843，OOF AUPRC为`0.2033/0.4456`，OOF AUROC为`0.5775/0.7415`。输出和owner见
  `experiments/genome_condition_reviewer/historical_probe/task_manifest.json`。最终focused 11 passed、
  全仓182 passed（14条既有warnings）；fragment/OOF artifact contract与独立metric重算通过。
- **论文候选图：** 本机 CPU 只读运行
  `MPLBACKEND=Agg PYTHONPATH=src python scripts/audit/plot_genome_representation_validation.py`；
  不重训 probe、不重算 embedding。三面板 PDF/SVG/PNG、caption、plotted-data 与 SHA-256 manifest
  位于 `experiments/genome_condition_reviewer/figures/`。Panel a 按作者要求排除 identical
  fragments，仅绘制 4,649 个连续 variable fragments，不使用bins或拟合趋势线，cosine distance
  使用 log scale；panel b/c 使用正式 264-genome saved-window probes 展示 5-fold performance 与基线。
  canonical 图已目视 QA；focused 13 passed、全仓 184 passed（14 条既有 warnings）。Panel a 的
  log y 轴下限为 `1e-8`（最小观测值 `5.86e-8`）；ANI `>=99%` 的4,156个 fragments为蓝色圆点，
  其余493个为灰色叉号。作者决定删除 linear-scale 对照及入口。
- **正式文稿资产（2026-08-06，已完成）：** canonical figure 已复制到
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Fig_SI_genome_representation_validation.pdf`
  并在正式 TeX 中编号为 Supplementary Fig. C6；其 SHA-256 为
  `0c1cf2307a93c8c9cc1b53d2d29e4f04efdc2e65df1d5bcb136acb0165abf68f`。同目录的
  `sn-article.tex`、`sn-bibliography.bib` 和 `Response to reviewers letter.docx` 已按冻结草稿
  更新，修改前均保留 `before_genome_representation_revision_20260806` 备份。独立编译/渲染产物
  只位于 `/tmp/apexoracle_genome_revision_*`，正式 `sn-article.pdf` 未覆盖。论文编译为 34 页，
  C6 位于第 26 页且无 undefined citation/reference；回复渲染为 31 页，新增回复位于第 7--9 页。
- **公共发布边界（2026-08-06）：** annotation manifest 与 L2 probe 的共享逻辑已独立到
  `src/apexoracle/evaluation/genome_fragment_validation.py`；正式 entrypoints 不再读取本机绝对路径
  下的外部 producer 源码。Genome/text swaps、旧 162-genome probe 和 strain-wise pilot 仅保留本地
  取证，不发布其 scripts、逐 member completion markers 或 superseded summaries。公共 capsule 只含
  正式 scripts/shared modules/tests、compact formal manifests/summaries、文稿记录和 canonical figure。

## 3. Fig. 1b 补实验（已完成）

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
  当时分别由 node001 GPU0/1/2 从干净输出目录训练 10 members，随后均已完成。node002 曾在共享 `/data1` 上
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

### 2026-07-26 hierarchical MIC molecule-disjoint sensitivity

- **已验证事实：** 原始 21 个 strain-wise checkpoint 合计约 181 GB，单文件约
  9.1--9.5 GB，包含本次推理不消费的 optimizer/classification payload。canonical producer
  `scripts/reproduce/prepare_hierarchical_mic_inference_checkpoints.py` 已生成 21 个
  inference-only 副本（单文件约 2.88 GB）；manifest 逐文件登记 source/inference size 和
  SHA-256。派生文件不替代历史训练权重。
- **任务分配：** 本机 H100 GPU 0/2/3 完成 strain group 1 ensembles 0/1/2，随后按
  `0,3,6`、`1,4`、`2,5` 复用 feature load 运行 group 2；node001 A100 GPU 0/1/2/3 运行
  group 0 ensembles 0/1/2/5，GPU 4 运行 group 1 ensemble 6；node002 A100 GPU 0/1/2
  先运行 group 0 ensembles 3/4/6，再运行 group 1 ensembles 3/4/5。没有使用本机 GPU 1。
- **共享写入边界：** node001/node002 只写共享 release 下唯一的
  `experiments/hierarchical_mic/molecule_disjoint/predictions/strain_group_G_ensemble_E.{csv,json}`
  和对应唯一日志；同一 `(G,E)` 没有双 owner。本机输出写当前 checkout 同名目录，完成后只读
  rsync 汇总。每个任务以同时存在且 JSON `status=completed` 为完成条件。
- **科学边界：** strain membership 来自冻结的 `PYTHONHASHSEED=0` candidate manifest，不是
  未恢复的 2025 精确 membership；replay 使用确定性 `eval()`，不能把结果写成论文原始
  train-mode-dropout evaluation 的逐值复现。
- **正式 sensitivity 补充：** 为消除上述 strain membership 不确定性，另对 membership
  可确定的 phylum-wise 三组 final MDLM checkpoint 运行 exact-molecule-disjoint replay。
  node001 GPU 0/1 分别负责 Fungi/Pseudomonadati，node002 GPU 0 负责 Bacillati；每个进程复用
  一次 feature load 后连续评估 7 members。源 checkpoint 位于共享历史资产
  `/data1/tianang/Projects/Synergy/Checkpoints/genome_text_learnable_emb/3_species_w_SM/MDLM_MTR_fix_cls_wo_pad_7_fold_ensembles`，
  唯一输出位于 shared release 的
  `experiments/hierarchical_mic/molecule_disjoint/phylum_predictions/`。
- **2026-07-26 完成事实：** phylum `3 × 7` replay 已全部完成并汇总回本机。
  exact-peptide-unseen 共 6,515 条 / 3,491 个 molecules，low-MIC<=16 micromolar 为
  58.20%；pooled R2/Spearman/Pearson 为 `0.0135/0.3326/0.3398`。三组 R2 分别为
  `0.0109/0.0752/-0.1589`，因此必须披露 Bacillati 的负 R2。2,000 次 exact-molecule
  cluster bootstrap 的 R2 CI 跨零，但模型相对 group-specific train-mean baseline 的 paired
  delta CI 为正。compact 结果位于
  `experiments/hierarchical_mic/molecule_disjoint/phylum_analysis/`。

### 2026-07-26 fixed strain-wise reviewer retraining

- **实时资源核验：** 启动前本机4张 H100 PCIe、node001 8张 A100 80GB、node002 8张
  A100 80GB 均为0% utilization且无 hierarchical MIC worker；节点 base 环境均为
  Torch `2.7.1+cu126`、CUDA可用。
- **固定协议：** 读取
  `experiments/hierarchical_mic/strain/legacy_protocol_manifest.json`，不动态重建 split；
  模型、四路训练、batch size 80、25 epochs、7个原 seeds和 held-out R2 selection均保持
  `configs/hierarchical_mic/legacy_mdlm.yaml` 契约。该实验是新 fixed-split reconstruction，
  不是2025精确 membership。
- **任务 ownership：** 完整 `3 × 7` 网格和唯一 owner 位于
  `experiments/hierarchical_mic/fixed_strain_retrain/task_manifest.json`。本机 GPU0--3
  负责 group0 members 0--3，GPU0随后串行 member6；node001 GPU0--6负责 group1全7个，
  GPU7负责 group0 member4；node002 GPU0--6负责 group2全7个，GPU7负责 group0 member5。
- **节点本地写入边界：** 首轮节点任务同时向共享NFS更新16个约9GB checkpoint，实测单epoch
  被拉长到23--32分钟；该首轮在1--3 epochs后精确终止，已有共享文件保留但明确排除于分析。
  node001/node002随后从相同seed的epoch 0重新启动，仅把输出分别改到
  `/local/tianang/Synergy_fixed_strain_retrain/node001/checkpoints` 与
  `/local/tianang/Synergy_fixed_strain_retrain/node002/checkpoints`。两台节点各有约1.6TB本地
  NVMe空闲；模型、split和训练协议未改变。完成后在producer节点直接做推理，只同步compact
  predictions/analysis，不搬运大型checkpoint。
- **完成状态与结果：** 21/21训练任务和21/21 deterministic replay均已完成。group1的七个
  8.5GB checkpoint 在node001提取为每个约2.7GB、带源size/SHA-256血缘的inference-only权重，
  经本机临时中转到node002；group1/2 replay按作者指示与node002现有任务共存于GPU7，未发生
  OOM或错误。compact结果已回收到
  `experiments/hierarchical_mic/fixed_strain_retrain/analysis/`。full/seen/unseen pooled R2为
  `0.4638/0.5672/0.0942`；unseen为26,272条 / 8,259个pooled distinct exact peptides，
  Spearman/Pearson为`0.4070/0.4130`，R2 cluster-bootstrap 95% CI为
  `[0.0687, 0.1191]`。论文式 mean-across-folds full/seen/unseen R2 为
  `0.5814/0.6283/0.1089`；full 与历史论文 `0.5793` 一致。本机临时中转副本位于
  `/data2/tianang/tmp/fixed_strain_group1_inference/`，属于可删除的派生缓存，不是canonical资产。
- **2026-07-27 manuscript handoff：** 正式TeX已消费上述fixed strain结果：主文strain-wise
  ensemble mean R2改为`0.5814`，Results/Methods分别增加sensitivity结果与split scope，
  appendices增加逐fold sensitivity table；加入MIC distribution section后其编译编号为
  Appendix Table D1。generation段在三条lead及低training-set identity
  已介绍后再回扣joint unseen-peptide/unseen-strain证据。临时编译29页且cross-reference通过，
  未覆盖正式`sn-article.pdf`或任何figure。已只读渲染核验`Fig1.pdf`与`Fig2_2.pdf`：
  Fig. 1a/2c均未直接印出`0.5793`，无需为本次`0.5814`更新修改figure。
- **2026-07-27 metric Methods handoff：** 作者要求撤销扩展版split/imbalance小节。正式TeX已
  恢复既有split段落，仅新增一个简短粗体metric段落，涵盖transformed-label R2、
  Pearson/Spearman、AUPRC/AUROC、constant predictor、accuracy未使用和五折mean/sample s.d.；
  未新增独立subsection，也未搬入完整prevalence与resampling说明。response-letter docx仍有一个
  待同步旧值：strain-wise R2 `0.5793`应改为`0.5814`。
- **2026-07-28 MIC distribution Supplementary Figure：** CPU入口为
  `python scripts/audit/plot_hierarchical_mic_test_distribution.py`，只读消费fixed strain-wise
  ensemble replay逐测量CSV，输出到`experiments/hierarchical_mic/mic_distribution/`，不使用GPU、
  不重训。实际response-letter docx与markdown draft中的strain-wise R2已从`0.5793`同步为
  `0.5814`。输出核验为pooled 86,358条、MIC<=16 micromolar 47.01%；
  三fold分别47.25%/41.59%/51.02%。分布宽但非严格均匀，512 micromolar附近存在明显峰。
  最终图移除总标题和source脚注，panel标题居中且不加粗，并增加`a/b`标记。正式文稿资产为
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Fig_SI_MIC_distribution.pdf`，
  与canonical输出SHA-256
  `2e35f86464a343ab21a7cf8df8ef605e8f9aa4fd4d60b9d1a188b70b5d4fc7a5`
  相同；TeX已将其作为Supplementary Fig. C3并在Methods引用。
  reviewer response DOCX与markdown draft已同步fixed strain-wise composition和Fig. C3引用。
  新增Fig. C3和Table D1的section标题/caption已使用`\rev{}`，Table D1正文使用局部
  `\color{red}`；figure内部坐标文字保持原图配色。临时独立目录编译29页且cross-reference
  通过，未覆盖正式`sn-article.pdf`。

| 资源 | Canonical 位置 | 边界 |
| --- | --- | --- |
| 论文原始/冻结数据 | 本机 `DataPrepare/Data`；节点通过上述 symlink 读取历史目录 | 只读消费；重构输出写 `results/`，不得原地覆盖 |
| Reviewer 4 私有 in-house AMP 表 | `DataPrepare/Data/private_inhouse_amp/Master_List_Peptides_Antimicrobial_Activity.xlsx` | Git ignored；SHA-256 `956ff3d60364a113c2149a63b74b65d4f81ec03741fefbac1841558ed54744a4`；不得上传 GitHub |
| 本仓库权重登记 | `configs/model_weights.yaml`、`MODEL_WEIGHTS.md` | 二进制不进 Git；新路径使用 `${APEXORACLE_WEIGHTS_DIR:-weights}` |
| DLM/MDLM producer | `/data2/tianang/projects/mdlm` | 必须用 `mdlm` conda env；当前 checkout 有本地修改，不能直接作为 clean submodule |
| Evo-2 producer | `/data2/tianang/projects/evo2` | 当前仓库只消费 `DataPrepare/Data/Genome_embs`；不重跑 extraction |
| guided generation | `/data2/tianang/projects/discrete-diffusion-guidance` | sampler、DLM pretraining 同属外部边界；当前只读审计，不并入 Synergy |
| Reviewer 2 peptide classifier | `experiments/peptide_classifier/README.md` | v1 数据血缘与论文 checkpoint 的 canonical 审计记录；node002 历史 producer 只读，v2 parser-label 数据不得反向解释 v1 checkpoint |
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

### 2026-07-26 Reviewer 2 peptide classifier retrain

- **只读输入：** 本机 v1 Arrow 数据
  `DataPrepare/MDLM/Data/hf_pep_SM_cls_1024`、raw labeled CSV、论文 classifier checkpoint；
  node001 的训练副本固定写到本地 NVMe
  `/local/tianang/peptide_classifier/hf_pep_SM_cls_1024`，复制完成后必须以
  dataset state、shard 数和总 bytes 核验。本机/节点 checkpoint SHA-256 均为
  `40f638ca5668f20a641a538035015b1741ab69cded300cba27f7148cc291945b`。
  **已验证：** node001 本地副本为 515 个 `data-*.arrow` shards；本机与节点的
  `filename + size` manifest SHA-256 均为
  `fe38e00b54e6e61de276c54cbf4e93e4c008b57f5a8089664d732f339f117a95`，`state.json` 与
  `dataset_info.json` 的 SHA-256 也逐项一致。
- **node001 环境：** 该节点没有预存 `mdlm` conda env；本机环境用 `conda-pack` 固定为
  `/local/tianang/peptide_classifier/mdlm-env`，archive SHA-256 为
  `7fc5bed8113037953258a849081b3535961056e90ff087a9210ddf39f305307b`。解包后执行一次
  `bin/conda-unpack`，并核验 Torch/flash-attn/OmegaConf/datasets 版本和 CUDA forward。
- **CPU owner：** node001 负责 sequence extraction/MMseqs clustering，并在本地 NVMe 上完成
  raw SMILES canonicalization、final Arrow 顺序联结与 split vector；node001 和本机分别执行
  独立 split audit，本机保存正式副本。MMseqs binary
  version 为 `17b688d21dda57fc5f5b7286ecba7ec003d4717f`，archive SHA-256 为
  `9c4c946cae9c9213a5d85c1381a32d2eb41f47cafdee8e42a730bcfbd64c348b`。
- **GPU ownership：** 本机 GPU0--3 唯一负责 seed 0；
  node001 GPU0--3 唯一负责 seed 1，GPU4--7 唯一负责 seed 2。三个输出分别为
  本机 `reviewer_retrain/runs/seed_0` 与 node001 本地 NVMe
  `/local/tianang/peptide_classifier/runs/seed_{1,2}`，不得交叉写 checkpoint。node002 启动时
  8 张卡均被既有任务占用，因此本轮不抢占；后续即使空闲也不重复启动已有 seed。
- **训练完成条件：** 每 seed 完成 2 epochs，并在每 45,000 steps 与 epoch 结束时执行
  validation；同时存在 `best.pt`、`last.pt`、`history.json` 和
  `best_validation_metrics.json`；最终 test 必须从每 seed 的 `best.pt` 单独执行并产生
  `test_metrics.json`。训练固定 2 epochs，不以接近时间预算作为完成条件。
- **正式 split 与启动状态：** 两台机器的独立 audit 均通过；split 为
  `81,137,648 / 830,022 / 827,381`（train/validation/test），manifest SHA-256 为
  `410d73abda7ece85015a700ecb69c947973f07766dd167686ec9517df0f675cb`。三个任务于
  2026-07-26 22:32 EDT 启动；首 1,000 steps 用时分别为 368.2、444.4、445.1 秒。
- **2026-07-27 seed 1 失败边界与作者决定：** seed 1 在完成 epoch 0 的 90,152 train steps
  后，于 epoch-end validation 的单元素 metric broadcast 触发 600 秒 NCCL watchdog timeout；
  GPU0--3 随后释放。作者决定不重跑，正式汇总仅纳入完整 seed 0/2。失败 run 保留在
  `/local/tianang/peptide_classifier/runs/seed_1`，不得将其写成低指标淘汰或完整 run。
- **接力入口：** `python scripts/reproduce/orchestrate_peptide_classifier_reviewer.py` 只读取上述
  冻结 task，等待 `2 × 90,152` train steps 完成后在原 GPU owner 上启动 clean/`t=0.5`
  评估；node-local 结果同步到本机后执行 1,000 次 molecule bootstrap。运行状态写入
  `experiments/peptide_classifier/reviewer_retrain/pipeline_status.json`。

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
