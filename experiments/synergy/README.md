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
