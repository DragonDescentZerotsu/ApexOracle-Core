# 预计算 Evo-2 genome embedding 契约

本仓库不负责生成 genome embedding。论文训练、评估和 reviewer scaling 分析直接读取已经
生成的 `DataPrepare/Data/Genome_embs/*.pt`。这些 tensor 属于版本化数据资产，不进入 Git。

## 已确认的架构决定

- 未来整合后的 ApexOracle 主仓库计划在 `external/evo2` 放置 Evo-2 Git submodule。
- submodule 只固定生产代码，不包含模型权重、genome 输入或预计算 embedding。
- 当前 `/data2/tianang/projects/evo2` HEAD 为
  `afd0dae0a4bb25f3ca55f171fbdac4907b937afd`，commit object 已验证存在，但工作树有本地修改。
  因此不能直接把当前 checkout 当成 clean submodule，也不能声称该 commit 已由原始日志证明是
  567 个 tensor 的精确生产版本。
- 当前 Synergy 仓库不新增 submodule、不迁移 extraction 代码、不重新运行 Evo-2。

## 已由代码和真实文件验证的事实

- `Genome_embs` 包含 567 个文件，共 3,437,540,485 bytes，没有解析后 ID 冲突。
- AMP MIC、小分子分类和 synergy 三份论文数据实际匹配 563 个 embedding。
- 未被这三份数据匹配的 4 个 ID 为 `1041`、`11696`、`BAA-3170`、`BAA-3197`；它们仍保留在
  发布 manifest 中，不能因当前未消费而删除。
- reviewer scaling CSV 恰好覆盖上述 563 个 ID。全部 tensor dtype 为 `torch.bfloat16`，hidden
  dimension 为 8192，第一维随 genome 长度变化。
- 重构后的只读 loader 使用 `torch.load(..., map_location="cpu", weights_only=True)`；完整重算的
  CSV 和 PNG 与现有文件逐字节一致。PDF 仅因生成 metadata 不同而 SHA 不同。
- 563 个 embedding 的 `mean(abs(E))` median 为 `2.2325736552308637e-15`；固定乘以 `1e14`
  后 median 为约 `0.223257`。

## 身份与发布

`file_manifest.csv` 记录每个 embedding 的 genome ID、文件名、字节数、SHA-256 和是否被三份
论文数据消费。manifest 自身 SHA-256 为：

```text
0420058138fd0473f9c3c6d92a0dae0ebe4ffedc1924a14ab6038259f0aa7496
```

重新审计时必须写入数据目录之外的新路径：

```bash
python scripts/audit/audit_precomputed_genome_embeddings.py \
  --data-dir DataPrepare/Data \
  --output results/evo2_precomputed_embedding_manifest.csv
```

完整 reviewer scaling 复现仍使用：

```bash
python scripts/plot_evo2_genome_embedding_abs_mean_distribution.py \
  --data-dir DataPrepare/Data \
  --output-dir results/evo2_scaling
```

## 证据边界

- **已验证：** 当前 tensor 文件身份、消费集合、shape/dtype contract、scaling 数值和输出等价性。
- **根据现有证据作出的推断：** embeddings 来自项目记录的 Evo-2-40B layer 46 流程。
- **仍待确认：** 精确 producer commit、输入 genome FASTA 版本、windowing 参数和模型权重身份。
  在这些信息恢复前，未来 submodule commit 只能标为候选，不能冒充完整数据生产血缘。
