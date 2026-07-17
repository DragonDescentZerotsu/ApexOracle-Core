# Fig. 2b：共享数据的 molecule encoder benchmark

本目录用于 reviewer 要求的新版正式 benchmark。它不沿用旧脚本中“每个模型先各自过滤、再各自 KFold”的做法。

## 已冻结的数据规则

所有 encoder 必须读取同一份：

- `common_molecule_ids.csv`；
- `folds.csv`；
- `shared_molecules.csv`；
- `dataset_manifest.json`。

APEX 使用有损序列投影：noncanonical residue 为 `X`，D-residue 仅保留 residue identity，DBAASP 的 `intrachainBonds`、`interchainBonds`、`coordinationBonds` 和 multimer topology 被线性化，超过 50 residues 的内容被截断。重构版 adapter 为 `X` 分配独立的 index 23，其冻结 AAindex 向量取 20 种 canonical residue 向量的均值，避免与 padding 混淆。所有这些情况逐条写入 `apex_projection_audit.csv`。

其他 encoder 遇到输入长度上限时同样只能确定性截断并记录，不能从 fold 中删除样本。任何无法产生 prediction 的 ID 都应使该 fold 明确失败，而不是静默缩小测试集。

## 统一训练协议

- encoder 全部 frozen，并在 `eval` mode 中生成 feature；
- 所有模型使用同一个 `384 → 128 → 19` regression head；
- 每个 outer training fold 内固定划出 10% molecule 作为 validation；
- best checkpoint 只根据 validation macro-task R2 选择，outer test fold 最后只评估一次；
- 最终报告五个 outer fold 的 mean ± sample SD，并保存逐 molecule、task 的预测。

这与旧 capsule 的派生结果不是同一协议。旧脚本会在各 encoder 自己过滤后的数组上重新 KFold，并在每个 epoch 用 outer test fold 选择 best checkpoint；APEX 还使用不同大小的 head。旧结果只保留作历史审计，不作为 reviewer 要求的新版公平结果。

## 构建共享数据

```bash
python scripts/prepare_data/build_fig2b_shared_dataset.py
```

默认产物写入被 Git 忽略的 `DataPrepare/Data/fig2b_shared_v1/`。最终发布时只提交不含原始数据的 manifest 模板或校验和；数据能否再分发需另行核对许可。

## 当前状态

- 共享 molecule IDs、APEX 投影、五折划分和审计 manifest：已实现；
- 共享数据 loader、训练折内 validation 划分、label transform 和 R2 实现：已实现；
- 严格校验 ID 的 `.npz` feature-cache 契约和统一 regression-head runner：已实现；
- APEX feature adapter：已实现，并用真实 pretrained checkpoint 为全部 11,398 个公共 molecule 生成和严格回读 `(11398, 128)` cache；
- ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM frozen/eval feature adapter：已实现并完成小样本 backbone smoke test，待在可用 GPU 上生成全量 feature；
- DLM MTR+DLM 与 DLM-only adapter：待从外部 `mdlm` 实现迁移；
- 正式五折训练、mean ± SD、论文图和 reviewer response 更新：尚未运行。

统一 head runner 的入口为：

```bash
python scripts/reproduce/run_fig2b_shared_heads.py \
  --feature-cache /path/to/encoder_features.npz \
  --output-dir /path/to/results \
  --encoder-name chemberta_mtr \
  --device cuda:0
```

旧 `.pt` cache 没有共享协议版本和完整 ID 契约，不能直接传入；必须由对应 adapter 在公共 ID 上重新生成 `.npz` cache。

全量 tokenizer 审计没有丢弃任何 ID：ChemBERTa-MTR、ChemBERTa-MLM 和 MolFormer 各有 512 条输入超过 512 tokens 并被截断，且没有 UNK；PeptideCLM 有 24 条被截断、8,150 条含 `[UNK]`。PeptideCLM 的高 UNK 比例必须随最终结果报告，不能通过过滤这些分子来改善指标。

APEX 或 Hugging Face comparator 的 cache 入口为：

```bash
python scripts/reproduce/cache_fig2b_shared_features.py \
  --encoder apex \
  --output /path/to/apex_features.npz \
  --device cuda:0
```
