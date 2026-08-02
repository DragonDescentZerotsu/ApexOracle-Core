# Reviewer 2 peptide classifier 数据血缘与评估边界

> 最后核验：2026-07-26。本文记录论文 guided generation 实际使用的 v1 peptide classifier，
> 不把 2025 年 6 月生成的 v2 parser-label 数据反向描述成 2025 年 5 月 checkpoint 的训练标签。

## Canonical 资产

| 资产 | 位置 | 状态 |
| --- | --- | --- |
| 论文实际 classifier checkpoint | `/data2/tianang/projects/mdlm/cls-guide-pad-no-mask-checkpoints/epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt` | v1，epoch 1 / step 134000 |
| v1 最终训练数据 | `DataPrepare/MDLM/Data/hf_pep_SM_cls_1024` | 82,795,051 rows |
| v1 tokenization 前 Hugging Face 数据 | `DataPrepare/MDLM/Data/hf_pep_SM_cls` | 保存了 `mol_ids`、token IDs 与标签，可逐来源审计 |
| 精确历史 trainer | `node002:/data1/tianang/Projects/mdlm/guaidance_classifier_all_data_pad_no_mask.py` | 2025-05-14；node002 历史目录只读 |
| 当前本机同名 trainer | `/data2/tianang/projects/mdlm/guaidance_classifier_all_data_pad_no_mask.py` | 已改为 v2 数据，不是上述 checkpoint 的精确 producer |

外部 `/data2/tianang/projects/mdlm` 中的代码必须在 `mdlm` conda 环境运行。当前审计只读消费
node002 历史目录和外部 guidance 仓库，没有改写原始数据或 checkpoint。

## 已由最终 CSV、Arrow shards 和 producer 路径验证的 v1 血缘

1. `DataPrepare/MDLM/MDLM_data.ipynb` 汇集 SmProt2、UniProt 和 UniRef peptide sequence。
2. node002 的 `DataPrepare/MDLM/uniprot_uniref_to_smiles.py` 将标准氨基酸序列转换为 SMILES。
3. `all_smiles.csv` 汇合四种可由 ID 前缀区分的来源：
   `SmProt2_*`、`uni_*`、`Generated_pep_CLM_*` 和 `pubchem_*`。
4. v1 按来源赋标签：前三种来源为 peptide `1`，PubChem 为 non-peptide `0`。
5. 数据随后转换为 SELFIES、分片、tokenize，并过滤/填充到 1,024 tokens，最终写入
   `hf_pep_SM_cls_1024`。

最终进入 v1 Hugging Face 数据的逐来源计数为：

| 来源 | v1 标签 | Rows |
| --- | ---: | ---: |
| SmProt2 | 1 | 677,323 |
| UniProt/UniRef | 1 | 3,105,732 |
| PeptideCLM generated | 1 | 6,654,492 |
| PubChem | 0 | 72,357,504 |
| **总计** |  | **82,795,051** |

正类共 10,437,547 条，占 12.6065%；这与历史 trainer 固定的 `pos_weight=7` 一致。这里的
“peptide”是来源标签，不等价于“每个结构都经独立 parser 验证为 peptide”。

## v1 历史训练 wall-clock

- **已由 checkpoint payload 验证的事实：** 论文采用的 step 134,000 checkpoint 位于 epoch 1；
  当前 epoch 已完成 42,925 steps，因此第一个完整 epoch 为 91,075 train steps。旧 1% validation
  对应 920 batches。
- **已由 checkpoint 文件时间验证的事实：** step 1,000 保存于 2025-05-13 17:16:35，论文采用的
  step 134,000 保存于 2025-05-14 11:08:28，最后的 step 163,000 / `last.ckpt` 保存于
  2025-05-14 23:04:52。step 1,000 之前的启动时间按稳定 step 间隔外推约为 17:08:39。
- **已由时间序列验证的事实：** step 1,000--134,000 之间每 1,000 steps 的中位耗时为
  475.692 秒，即 0.475692 秒/step。一个完整 epoch 约 12.0 小时；论文 checkpoint 约在启动后
  18.0 小时生成。现存运行继续到 step 163,000，包含后段减速和停顿的总 wall-clock 约 30.0 小时，
  并未跑完 trainer 配置中的 10 epochs。
- **根据现有证据作出的高置信度推断：** 数据规模、每卡 batch size 300、91,075 steps/epoch 和
  同期 trainer 副本共同指向 3-GPU DDP、global batch size 900。node002 当前保留的同名文件在
  2025-05-14 18:16 后已改为 `devices=1`，因此它能验证架构和数据路径，但不能单独证明最初启动
  进程的 GPU 数；精确 launch command 仍未恢复。

## v2 不能反向解释 v1 checkpoint

2025 年 6 月的 v2 使用 `/data2/tianang/projects/mdlm/smiles_to_peptide.py` 将结构解析为 peptide
sequence；解析失败或包含 `X` 时标为 0。该 parser 明确只覆盖线性、无环、无复杂修饰的标准
氨基酸肽。v2 原始 125,300,511 个结构中的来源级标签为：

| 来源 | v2 正类 | v2 负类 |
| --- | ---: | ---: |
| SmProt2 | 816,379 | 195 |
| UniProt/UniRef | 1,396,180 | 1,709,552 |
| PeptideCLM generated | 5,269 | 9,994,730 |
| PubChem | 87 | 111,378,119 |

因此，v2 更接近“parser 可识别的标准线性 peptide”，但 parser failure 不能自动解释为真实
non-peptide。除非重新构建标签并重训 checkpoint，Methods 和 reviewer response 必须按 v1
来源标签描述论文 classifier。

## 已恢复行为与仍缺失文件的边界

- **已验证事实：** 四类来源、v1 标签规则、最终逐来源数量、dataset 路径、trainer 路径和
  checkpoint 身份均可由保存的 CSV、Arrow shards、文件时间和 checkpoint payload 交叉验证。
- **仍未恢复：** 负责把四个来源拼接并写出 `all_smiles_pep_SM_cls.csv` 标签列的临时脚本原文件。
  本机工作树、`legacy-code-snapshot-2026-07-17`、node002 项目目录和 shell history 均未找到该
  文件；它可能来自未保存的交互式 notebook cell。
- **表述边界：** 可以说“v1 实际数据装配行为已恢复”，不能说“标签写入脚本源码已恢复”。

## 当前确认的最小 reviewer 评估协议

- primary：canonical-molecule-disjoint、peptide-sequence-cluster-disjoint 的 clean
  AUROC/AUPRC，并在固定 probability 0.5 阈值报告 peptide recall 与 non-peptide specificity；
- noisy robustness：只评估实际 guidance 中点 `t=0.5`，使用固定 mask seeds，并先在 molecule
  内聚合预测；
- bootstrap 和置信区间以 molecule 为独立单位；
- 当前不把 peptide-like hard negatives 设为必要条件；
- clean 输入对应 `sigma=0`。`t=0.5` 在 log-linear schedule 下对应约 `sigma=0.69215`；
  论文 checkpoint 训练时 `time_conditioning=False` 会把传入 backbone 的 sigma 清零，而实际
  generation 配置 `time_conditioning=True` 会保留该非零 sigma。部署一致的 `t=0.5` 主结果应走
  generation 路径；可在相同 masks 上额外比较一次 zero-sigma 作为配置敏感性检查。

## 2026-07-26 reviewer retrain 冻结协议与入口

> **发布边界：** `labels.u1`、`sources.u1`、`split_codes.u1`、`molecule_hashes.u8` 和
> `real_sequence_roots.u8` 是由下述 split producer 确定性重建的本地 memmap 中间件，总计约
> 911 MB，不进入 Git。公共仓库只保留 `split_manifest.json`、`split_audit.json`、紧凑结果、
> 生成脚本和测试；manifest 中的 size/SHA-256 用于核验本地重建结果。这样不能从 Git checkout
> 直接开始训练，须先执行 split preparation，但可避免把可重建缓存永久写入仓库历史。

- 标签仍为上述 v1 四来源标签，不混入 v2 parser-label 数据。
- 每个 RDKit 可解析结构先经 canonical isomeric SMILES 归一化；同一 canonical molecule 只能
  位于一个 split。当年 SELFIES producer 接受、但当前 RDKit 无法解析的 raw row 不会被静默删除，
  而是进入独立 `raw:SMILES` identity namespace；最终 manifest 必须报告 fallback 数量，且不得把
  这部分写成已 canonicalize。
- 最终 Arrow 的 row order 经实查不保持 raw CSV 顺序（例如首个 1,000-row record batch 从
  `SmProt2_2526` 开始并在 batch 内回跳到 `SmProt2_1912`）。split producer 因此先按四种
  structured ID 的整数部分构建 raw canonical lookup，再按 Arrow 的实际顺序回填，不能改回
  双指针顺序 join。
- `SmProt2_i` 已验证对应 `SmProt2_All.csv` 第 i 个数据行；`uni_i` 已验证对应
  `unique_peptide_sequences.txt` 第 i 行。两类真实 peptide 共 3,954,076 条 sequence，使用
  MMseqs2 `easy-linclust` 以 40% identity、80% 双向最短 coverage 聚类；共享 canonical
  molecule 的 sequence cluster 再合并为一个 connected component。
- PeptideCLM generated 正类没有可验证的原始 sequence，因此只按 canonical molecule 分组；
  sequence-OOD 解释限定于 SmProt2 和 UniProt/UniRef。
- group 通过固定 seed `20260726` 确定性映射为 98% train / 1% validation / 1% test；完整 group
  不为追求精确比例而拆分。若同一 canonical molecule 同时携带正负 v1 来源标签，仍保留在同一
  split，但从 molecule-level 主指标排除并单独报告。
- 模型直接从论文 v1 classifier checkpoint 提取 12-layer/768 DDiT backbone，backbone 固定为
  `eval()` 且冻结；重新初始化 `768→384→128→1` head。训练随机采样 t 并固定
  `time_conditioning=True`；原计划 3 seeds、每 seed 4 GPUs、global batch 900、2 epochs。每 45,000
  steps 与 epoch 结束时按 validation clean AUPRC 选最佳 checkpoint，因此覆盖约 0.5、1.0、1.5
  和 2.0 epochs；epoch 结束时另算一次 `t=0.5` validation robustness。CosineAnnealingLR 仍固定
  历史 `T_max=10`，不会因 reviewer run 只执行两个 epochs 而把 scheduler 人为压缩到 2 epochs。
- 最终 test 只解封一次：clean 使用 1 次确定性输入；`t=0.5` 使用 10 个 stateless molecule mask
  replicates，先在 row/replicate 内聚合、再按 canonical molecule 聚合。
- **已通过两台机器独立审计的正式 split：** train 81,137,648、validation 830,022、test
  827,381；其中 test 含 103,150 positive rows 和 724,231 negative rows。split manifest
  SHA-256 为 `410d73abda7ece85015a700ecb69c947973f07766dd167686ec9517df0f675cb`。
  3,954,076 条真实 peptide 形成 2,804,874 个 sequence clusters；最终有 458 条 raw records
  不能被当前 RDKit 解析，其中进入最终 Arrow 的 fallback 为 252 rows。仅 1 个 canonical
  molecule 存在来源标签冲突，涉及 491 rows，按协议从 primary molecule metrics 排除。
- **2026-07-26 22:32 EDT 正式启动：** seed 0 使用本机 GPU0--3，seed 1/2 分别使用 node001
  GPU0--3/GPU4--7。三个 `run_manifest.json` 的 dataset rows、global batch、checkpoint SHA、
  split SHA 和 `time_conditioning=True` 均一致。首 1,000 steps 分别耗时
  368.2/444.4/445.1 秒，mean loss 为 0.15453/0.16219/0.16105。
- **2026-07-27 已验证的失败与作者决定：** seed 1 完成 epoch 0 的 90,152 train steps 后，在
  epoch-end clean validation 的 metric broadcast 前触发 600 秒 NCCL watchdog timeout；日志无
  loss/NaN/OOM 证据。作者决定不重跑，正式结果只使用完整完成的 seed 0 和 seed 2。seed 1 的
  排除由基础设施失败触发，不基于其 clean validation AUPRC（45,000-step 时为
  `0.9999999951`），失败产物保留作审计但不进入 ensemble。

sequence extraction 和 clustering：

```bash
PYTHONPATH=src python scripts/reproduce/prepare_peptide_classifier_split.py \
  extract-sequences \
  --output-dir experiments/peptide_classifier/reviewer_retrain \
  --uniprot-sequences DataPrepare/MDLM/Data/unique_peptide_sequences.txt \
  --smprot-csv /data1/tianang/SMILES_data/SmProt2_All.csv

PYTHONPATH=src python scripts/reproduce/prepare_peptide_classifier_split.py \
  cluster-sequences \
  --output-dir experiments/peptide_classifier/reviewer_retrain \
  --mmseqs experiments/peptide_classifier/tools/mmseqs/bin/mmseqs \
  --threads 192 --min-seq-id 0.4 --coverage 0.8
```

canonical split、审计和四卡训练：

```bash
PYTHONPATH=src python scripts/reproduce/prepare_peptide_classifier_split.py \
  assign-splits \
  --output-dir experiments/peptide_classifier/reviewer_retrain \
  --cluster-tsv experiments/peptide_classifier/reviewer_retrain/mmseqs_clusters_cluster.tsv \
  --raw-csv DataPrepare/MDLM/Data/all_smiles_pep_SM_cls.csv \
  --dataset-dir DataPrepare/MDLM/Data/hf_pep_SM_cls_1024 \
  --workers 64 --seed 20260726

PYTHONPATH=src python scripts/audit/audit_peptide_classifier_split.py

PYTHONPATH=src torchrun --standalone --nproc-per-node=4 \
  scripts/reproduce/run_peptide_classifier_reviewer.py \
  --dataset-dir DataPrepare/MDLM/Data/hf_pep_SM_cls_1024 \
  --split-dir experiments/peptide_classifier/reviewer_retrain \
  --producer-root /data2/tianang/projects/mdlm \
  --v1-checkpoint /data2/tianang/projects/mdlm/cls-guide-pad-no-mask-checkpoints/epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt \
  --output-dir experiments/peptide_classifier/reviewer_retrain/runs/seed_0 \
  --seed 0 --batch-size 225 --epochs 2 --validation-interval-steps 45000

# 三个 task 启动后，以 task_manifest.json 为冻结输入接力最终测试与汇总
python scripts/reproduce/orchestrate_peptide_classifier_reviewer.py
```

已验证的代码级 smoke test：本机 GPU1 上 forward/backward 为 finite，backbone 保持
`eval()`，全部 344,705 个可训练参数均属于 head；100-row train→best-checkpoint→clean/`t=0.5`
evaluation→prediction artifact→molecule bootstrap 全链路通过；2-GPU DDP 的 train、分布式
validation gather 和 best checkpoint 也已通过。正式 split 与训练结果完成后在本节追加精确
指标。
