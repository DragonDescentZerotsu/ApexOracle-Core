# 预计算 Evo-2 genome embedding 契约

本仓库不负责生成 genome embedding。论文训练、评估和 reviewer scaling 分析直接读取已经
生成的 `DataPrepare/Data/Genome_embs/*.pt`。这些 tensor 属于版本化数据资产，不进入 Git。

## 已确认的架构决定

- ApexOracle super-repo 已将 clean `ApexOracle-Evo2` 固定为独立 submodule；该模块只提供通用
  extraction code，不包含模型权重、genome 输入或预计算 embedding。
- Core 只消费版本化 embedding，不复制 Evo-2 extraction 实现，也不因整理清单而重新运行 40B 模型。
- 当前公开 Evo-2 模块已经通过通用 40B runtime smoke，但原始 567 个 tensor 的 extraction log、精确
  producer commit 和模型权重身份没有完整恢复；不得把当前 module commit 写成这些历史 tensor 的精确
  producer。

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
- 2026-08-05 saved-tensor-compatible window reconstruction 与 567/567 tensor shapes 精确一致，其中
  370 个为 multi-record FASTA。**已验证边界：** frozen tensors 只代表 saved fragment condition，
  不外推到其余 sequence。Canonical 审计入口和 sub-species variation
  结果见 `scripts/audit/analyze_genome_fragment_variation.py` 与
  `experiments/genome_condition_reviewer/RESULTS.md`。

## 身份与发布

`file_manifest.csv` 记录每个 embedding 的 genome ID、文件名、字节数、SHA-256 和是否被三份
论文数据消费。manifest 自身 SHA-256 为：

```text
0420058138fd0473f9c3c6d92a0dae0ebe4ffedc1924a14ab6038259f0aa7496
```

`paper_genome_list.csv` 是 reviewer-facing 的论文基因组清单，只保留上述 563 个实际被至少一项论文
MIC、classification 或 synergy 数据使用的 genome。每行记录 paper-era species label、保守来源类型、
可核验来源标识、当前 filename-matched FASTA 的文件身份、embedding 文件身份，以及三项任务的独立使用
标记。清单为 171,749 bytes，SHA-256 为：

```text
64323cab44a4a287b0b63e6e60bd7b0270557d5f0ce5715acb651aeb98b1f860
```

其中 MIC/classification/synergy 分别使用 563/2/100 个 genome。`paper_genome_list_manifest.json` 固定
这些计数、输入清单 hash 与解释边界。`current_fasta_*` 只表示发布审计时 filename-matched 的现存 FASTA，
不声称它与未恢复的原始 producer 输入逐字节相同；无法核验精确外部 accession 时保留
`not_recovered`，不作猜测。

确定性重建命令为：

```bash
PYTHONPATH=src python scripts/audit/build_paper_genome_list.py \
  --data-dir /path/to/DataPrepare/Data \
  --overwrite
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
- **已验证：** 563 行论文基因组清单与三份 paper dataset 的匹配集合、当前 FASTA/embedding 文件身份和
  任务计数。
- **根据现有证据作出的推断：** embeddings 来自项目记录的 Evo-2-40B layer 46 流程。
- **仍待确认：** 精确 producer commit、模型权重身份，以及当前 FASTA 是否逐字节等于生产时输入。
  Window size/step 和 saved-tensor indexing 已有代码与 567/567 tensor-shape 一致性支持，但这
  不等同于恢复完整生产环境。
  在这些信息恢复前，未来 submodule commit 只能标为候选，不能冒充完整数据生产血缘。
