# AMR/MGE fragment probes

该目录保存正式 fragment annotation probes 的冻结产物。目录名 `historical_probe` 是已有产物路径的
legacy 名称，不表示当前入口依赖外部历史源码。

Canonical commands：

```bash
PYTHONPATH=src python scripts/audit/prepare_historical_genome_annotation_probes.py
PYTHONPATH=src python scripts/audit/run_historical_genome_annotation_probes.py
```

Preparation 从三个论文数据集匹配到的 563 个 embedding IDs 开始，只保留 bacterial genomes，并
要求存在 matching GenBank、FASTA/GenBank sequence 与 record order 完全一致、saved-window
reconstruction 与 tensor rows 完全一致。最终 cohort 为 264 genomes、96,716 fragments。

共享实现位于 `src/apexoracle/evaluation/genome_fragment_validation.py`。该模块集中管理兼容性 manifest、
保守 annotation dictionary、deterministic probe cohort、固定 `1e14` scaling 和五折 L2 logistic
probe，避免 preparation/evaluation wrappers 互相复制逻辑。

每个 annotation 使用独立固定超参数 L2 logistic regression（`C=1`、
`class_weight=balanced`、`liblinear`），embeddings 保持冻结。同一 genome 的所有 fragments 始终处于
同一个 fold。正式 OOF AUPRC/AUROC 为：

- AMR-associated：`0.2033/0.5775`，evaluation prevalence `0.1667`；
- mobile-element-associated：`0.4456/0.7415`，evaluation prevalence `0.1977`。

`manifests/manifest.json`、`analysis/summary.json` 和 `task_manifest.json` 记录资产、源码、输出及协议
血缘。标签来自已有 GenBank annotation 的保守词典，不是完整 resistome 或 mobile-element
catalogue，也不支持 single-gene 或 causal attribution。
