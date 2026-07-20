# 计算节点、代码版本与资源位置

> 最后核验：2026-07-20。动态运行状态以
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
| 本仓库权重登记 | `configs/model_weights.yaml`、`MODEL_WEIGHTS.md` | 二进制不进 Git；新路径使用 `${APEXORACLE_WEIGHTS_DIR:-weights}` |
| DLM/MDLM producer | `/data2/tianang/projects/mdlm` | 必须用 `mdlm` conda env；当前 checkout 有本地修改，不能直接作为 clean submodule |
| Evo-2 producer | `/data2/tianang/projects/evo2` | 当前仓库只消费 `DataPrepare/Data/Genome_embs`；不重跑 extraction |
| guided generation | `/data2/tianang/projects/discrete-diffusion-guidance` | sampler、DLM pretraining 同属外部边界；当前只读审计，不并入 Synergy |
| PepLink | `/data2/tianang/projects/PepLink` | clean 独立仓库，未来使用固定 release/submodule 或链接，不复制源码 |
| 论文 TeX / reviewer response | `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech` | Codex 不自动替换论文 PDF；作者决定何时采用新 panel |

截至本次核验，外部 checkout 为：`mdlm` HEAD `7a6a7d1`（dirty）、guidance HEAD `edb0f8c`
（dirty）、Evo-2 HEAD `afd0dae`（dirty）、PepLink HEAD `cec2a02`（clean）。这些 SHA 只记录当前
候选状态，不反向证明 2025 论文运行的精确 producer commit。

## 5. 发生变化时必须同步更新

以下任一变化都必须同时更新本文件、根 `AGENTS.md` 和 `REFACTOR_PLAN.md`：

1. 新增/停用 GPU 或更换机器；
2. 数据、embedding、checkpoint 或外部仓库迁移；
3. GitHub `main` 与节点 release 工作树不再 fast-forward 对齐；
4. Fig. 1b 队列、ensemble 数、fold、epoch、selection 或 baseline feature 协议改变；
5. Mac notebook、论文图编辑边界或 submodule 决策改变。

无法由代码、hash、日志或实际 mount 验证的内容必须标成“推断”或“待作者确认”，不能写成事实。
