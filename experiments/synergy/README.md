# Synergy 三折 CV（experimental）

本目录迁移论文时期的 strain-wise synergy 二分类候选实现。当前状态是
`protocol frozen / model behavior migrated / exact paper identity unresolved`，不能把现存
checkpoint 的结果直接写成已经精确复现论文数值。

安装本模块依赖：`pip install -e '.[synergy]'`。验收环境使用 PyTorch 2.6.0、PEFT 0.15.1、
Transformers 4.49.0 和 SELFIES 2.1.1。

## 已由代码、数据和日志验证的事实

- 候选 legacy driver 是
  `synergy_Evo_train_new_reg_MDLM_one_base_model_classification.py`，其 SHA-256 为
  `d4cff701c592c6f9eb2973b72834687fdfb21c33fb9b41cc0182a8f72488f119`。
- 原始输入 `synergistic_pairs_Evo.csv` 有 4,285 行，SHA-256 为
  `ff57e2152159be950a9823f5d94f24c0771b18465bd17cfd11d9d0318db393be`；动态 embedding/mapping
  过滤后得到论文所述的 2,732 行，其中 genome+text 2,263 行、text-only 469 行。
- `FICI < 0.5` 映射为正类 1，`FICI >= 0.5` 映射为 0。
- legacy split 依赖 Python 无序 `set`，并跨 fold 原地修改 taxonomy alias list。固定
  `PYTHONHASHSEED=0` 后，四路划分的过滤前和过滤后行数逐项等于 2025 年日志：

  | fold | genome train | genome test | text train | text test |
  |---|---:|---:|---:|---:|
  | 0 before | 289 | 1,974 | 727 | 31 |
  | 0 after | 258 | 1,917 | 653 | 27 |
  | 1 before | 2,179 | 84 | 2,436 | 212 |
  | 1 after | 2,110 | 65 | 2,345 | 187 |
  | 2 before | 2,250 | 13 | 2,538 | 181 |
  | 2 after | 2,162 | 13 | 2,422 | 162 |

- token filter 使用 SELFIES、最大长度 512，revision 固定为
  `55e83392264cb998f7aa5014847df29868aefeb8`。
- 现存 checkpoint 是完整的 `3 folds × 7 members` 网格，21 个文件均为
  2,238,287,881 bytes，且 schema/shape 完全一致。实际启用的 fusion LoRA rank 是 1024；
  synergy head 是完整参数训练的 `24576→3072→128→1` MLP，没有启用代码中构造但未使用的
  rank-256 head LoRA config。
- 三个现存日志的 ensemble AUROC/AUPRC 为 `0.6690/0.6159`、`0.7614/0.6853`、
  `0.8489/0.9307`，未加权均值约为 `0.7598/0.7440`。

## 根据现有证据作出的推断

该 driver 和 checkpoint family 是当前最接近论文 synergy 结果的完整实现，因为任务定义、
2,732 行 eligible 数据、三折 strain-wise 协议、7-member ensemble 和 base checkpoint 血缘均
吻合。但是，现存 mean 指标与论文 `0.7539/0.7454` 并不完全相同，因此只能标为高相关候选，
不能标为精确 paper run。

## 仍待作者确认或进一步实验核验

- Methods 写 fusion LoRA rank 64，而候选 CV checkpoint 的 tensor shape 明确证明 rank 1024。
- Methods 写 base model 训练 13 epochs，而候选 driver 明确加载的 checkpoint 来自一份完整
  100-epoch run；仓库中另有 13-epoch checkpoint，但候选 CV 没有加载它。
- Methods 中 MLP 维度写成 `12,294→3,073`，实际 tensor 为 `12,288→3,072`；synergy head
  实际为其双倍输入 `24,576→3,072→128→1`。
- 旧日志没有记录 `PYTHONHASHSEED`，所以 seed-0 membership 只是与行数精确一致的确定性
  候选，不能证明是 2025 年三个独立进程的逐 strain 精确 membership。

## 复现入口

冻结 split manifest（不会修改原始数据）：

```bash
PYTHONHASHSEED=0 python scripts/prepare_data/build_synergy_legacy_split_manifest.py \
  --local-files-only \
  --output /tmp/synergy_legacy_split.json
```

配置位于 `configs/synergy/legacy_cv.yaml`，统一入口为：

```bash
PYTHONHASHSEED=0 python scripts/reproduce/run_synergy_cv.py \
  --fold 0 --device cuda:0 --local-files-only \
  --confirm-experimental-protocol
```

完整训练必须显式确认 experimental protocol。统一 runner 已完成 fold 2、1 member、1 epoch
的真实 H100 smoke，成功写出 checkpoint、175 条逐样本预测、metrics 和 summary；该 smoke
的 `0.8054/0.9011` 只验证执行路径，不是论文结果。`fold_0/ensemble_0` 真实 checkpoint 的
genome+text 与 text-only forward 也均和 inline legacy 公式逐值完全一致。全部 22 个大型
binary（1 个 base + 21 个 member）的 SHA-256 位于 `checkpoint_file_manifest.csv`。root legacy
driver 暂时保留，直到 exact paper identity 冲突得到作者确认或明确决定归档为候选。

## All-data guidance classifier（post-paper）

Guidance classifier 已与上述论文候选 CV 明确分离。它读取
`synergy_DBAASP_inhouse_Evo.csv`，把全部 eligible genome+text 数据作为一路训练集，并把全部
eligible 数据再次作为 text-route 训练集；没有 held-out fold。过滤前两路分别为 2,320 和
2,789 行，SELFIES token filter 后为 2,213 和 2,635 行，与现存两份日志完全一致。

该路径使用外部 `/data2/tianang/projects/mdlm` 的 `last_reg_v1.ckpt` 在线编码两个分子，输入固定
padding 到 1,024 token；MDLM 保持 eval/frozen，代码中的 `noise_input` 概率实际为 0。fusion
使用 rank-64 LoRA，head 为完整训练的 `24576→3072→128→1`。checkpoint 按同一个训练集的
AUROC 严格提升保存，因此其中的 0.8065/0.8562 不能解释为泛化指标。

为保持训练随机轨迹，canonical step 继续消费旧脚本中那次结果恒为 false 的 CPU `torch.randn`
调用，并保留两条 route 不同的 attention/dropout 调用顺序。旧脚本还通过 Python `set` 迭代
拼接 strain block；历史进程没有记录 `PYTHONHASHSEED`，因此当前入口冻结成员和转换行为，但不
声称跨进程恢复了历史逐行顺序。完整训练必须显式确认这一边界。

两个已观察到的运行通过 profiles 分开：

- `short_judger`：2 epochs，对应 `synergy_judger/cls`；
- `guidance_40epoch`：40 epochs，对应 `guidance_noise_synergy/cls`。

只读 dry-run 可在 base 环境执行：

```bash
python scripts/reproduce/run_synergy_guidance.py \
  --profile guidance_40epoch --dry-run --local-files-only
```

完整 MDLM 训练/验证必须按项目约定使用 `mdlm` conda 环境，并显式确认这是 post-paper guidance：

```bash
conda run -n mdlm python scripts/reproduce/run_synergy_guidance.py \
  --profile guidance_40epoch --device cuda:0 --local-files-only \
  --confirm-post-paper-guidance --acknowledge-dynamic-legacy-order
```

canonical 默认写入 `results/synergy_guidance/<profile>`，不会覆盖历史 checkpoint 目录。两份
4.1 GB checkpoint 均已严格加载，并在 H100 上完成真实样本 forward；完整 SHA-256、schema、
固定批次值和 source-version 边界见 `guidance_checkpoint_audit.json`。prospective in-house
screening 仍是独立的 regression ensemble consumer，不属于本 guidance 训练入口。

## Prospective regression producer（post-paper）

`scripts/reproduce/run_synergy_regression.py` 迁移了 screening 所消费的两组七模型 regression
checkpoint 的上游 producer。它读取 DBAASP pair 作为训练来源，并只在 57 行 prospective
in-house 数据上测试；动态 mapping 和 SELFIES 过滤后的三条实际路径为：

- genome+text training：`2263→2175`；
- combined-text training：`2732→2597`；
- genome+text in-house test：`57→38`。

旧脚本在每个 `zip_longest` iteration 中先执行 genome+text optimizer step，再执行
combined-text step，因此 genome 样本会在两个 route 中重复训练。target 是
`-log10(FICI/10)`，loss 为 MSE，fusion LoRA rank 64，head 为完整训练的
`24576→3072→128→1`。两类 checkpoint 语义不能混淆：

- `best_test` 直接由 38 行 in-house test R² 的严格提升选择，存在 test-selection leakage；
- `fixed_epoch` 保存 epoch index 5（第 6 epoch）结束后的参数，但 payload 的 `R2` 仍是截至
  当时观察到的 best-test R²，而不一定属于所保存参数。

只读 dry-run 会核验 source、mapping、两个 molecule cache 和 9.17 GB base checkpoint 的哈希：

```bash
PYTHONHASHSEED=0 python scripts/reproduce/run_synergy_regression.py \
  --dry-run --local-files-only
```

真实训练只写 `results/synergy_regression_producer/`，并拒绝写入 `DataPrepare/Data` 或
`Checkpoints`。由于 2025 年进程没有记录 `PYTHONHASHSEED`，必须显式确认动态 set-block 行序：

```bash
python scripts/reproduce/run_synergy_regression.py --device cuda:0 \
  --local-files-only --confirm-post-paper-regression \
  --acknowledge-dynamic-legacy-order
```

真实全 route、1 member、1 epoch 的 H100 smoke 已完成并写出符合历史六键 schema 的 checkpoint；
这只验证执行路径，不声称重新得到历史 tensor。完整数据、日志、stored R² 和证据边界见
`regression_producer_audit.json`。

## Prospective BAA-3170 pair screening（post-paper）

Canonical 入口为 `scripts/reproduce/run_synergy_screening.py`，配置为
`configs/synergy/legacy_screening.yaml`。它消费 3,441 个预计算 molecule embedding、BAA-3170
的 frozen genome/text embedding，以及两组各 7 个 rank-64 regression checkpoint。默认
`DBAASP_train_best` 使用 `fixed_epoch` 权重；`inhouse_best` 使用按 in-house test R² 选择的权重。
两组都不是论文三折 CV 证据。

历史输出存在两个必须保留并披露的位置语义：4-rank `DistributedSampler` 的 stride predictions
按 rank block 拼接后没有恢复原始顺序；随后这些位置又被直接用于筛选比 SMILES 表多 31,005 行的
sequence 表。canonical reproduction 默认精确保留该行为，并要求显式确认。1,280 个真实 pair、
7 个真实 member 的 H100 验证中，第一个完整 rank batch 的 110 条入选记录与历史 CSV 在 ID、
sequence 和 FICI 上逐值一致，最大绝对差为 0。

只读 dry-run：

```bash
python scripts/reproduce/run_synergy_screening.py \
  --profile DBAASP_train_best --dry-run
```

小规模验证或完整复现都只写 `results/synergy_screening/`（也可用 `--output` 指定），并拒绝写入
`DataPrepare/Data`：

```bash
python scripts/reproduce/run_synergy_screening.py \
  --profile DBAASP_train_best --device cuda:0 \
  --confirm-legacy-positional-alignment
```

完整输入、14 个 checkpoint 和两份历史输出的 SHA-256 见 `screening_audit.json`。当前没有生成
按 pair ID 修正对齐的新筛选结果；将来如做该分析，必须作为单独 sensitivity result 发布，不能
替换历史结果。
