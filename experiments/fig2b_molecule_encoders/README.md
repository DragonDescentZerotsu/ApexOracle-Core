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

## 构建共享数据

```bash
python scripts/prepare_data/build_fig2b_shared_dataset.py
```

默认产物写入被 Git 忽略的 `DataPrepare/Data/fig2b_shared_v1/`。最终发布时只提交不含原始数据的 manifest 模板或校验和；数据能否再分发需另行核对许可。

## 当前状态

- 共享 molecule IDs、APEX 投影、五折划分和审计 manifest：已实现；
- 各 encoder adapter 和统一 regression head：待迁移；
- 正式五折训练、mean ± SD、论文图和 reviewer response 更新：尚未运行。
