# ApexOracle / Synergy research code

本仓库正在把 ApexOracle 论文时期的大型复制脚本迁移为可审计的共享模块、版本化配置和稳定
命令行入口。重构原则是先冻结并验证历史行为，再删除被替代副本；原始源码血缘可从 Git tag
`legacy-code-snapshot-2026-07-17` 查看。

当前已完成以下主路径：

- Fig. 2b 共享 10,886 molecule / 5-fold molecular encoder benchmark；
- hierarchical MIC（strain/species/phylum）和 Fig. 1b 三菌株抗生素分类的行为保持重构。
- ApexOracle-3/12/23 sequence-similarity 分析的 cache、alignment 和输出等价重构。
- synergy 论文三折候选、modality ablation 绘图入口、k-mer producer/consumer 和论文前数据
  pipeline 的清理与证据冻结。

当前进行中的 reviewer 补实验是 Fig. 1b fine-tune 完整 10-member/fold ensemble 与
`20/10/20` no-RDKit 对照。实时查看本机、node001、node002：

```bash
watch -n 30 python scripts/reproduce/monitor_fig1b_revision.py
```

机器职责、conda/venv、共享文件系统、数据/权重和外部仓库位置统一记录在
`docs/COMPUTE_AND_ASSET_MAP.md`。

PepLink 作为独立发布的 peptide↔SMILES 工具，不复制进本仓库。ApexOracle 固定可选依赖
`PepLink==0.1.1` 并通过薄 adapter 调用；外部依赖决策和 179 条历史 structure correction
兼容性审计见 `experiments/amp_data_pipeline/README.md`。

## Lead peptide sequence similarity

```bash
python scripts/reproduce/run_sequence_similarity.py all
```

该入口重建 paper cache、计算 linear/exhaustive-cyclic similarity、提取 top hits 并验证输出。
论文的三个最大 PID `0.3667 / 0.3571 / 0.3684` 已全部复算；ApexOracle-3/23 的四份历史
核心 CSV 与新模块逐字节相同。数据大小写契约和 ApexOracle-12 的完整 tie 见
`experiments/sequence_similarity/README.md`。

## Fig. 1b 三菌株分类

查看 strict zero-shot 的真实数据契约而不加载模型：

```bash
python scripts/reproduce/run_antibiotic_classification.py \
  --mode strict-zero-shot \
  --test-group 0 \
  --dry-run
```

同一入口还支持：

```bash
# 目标菌株五折 fine-tuning
python scripts/reproduce/run_antibiotic_classification.py \
  --mode fine-tune --test-group 0 --fold 0 --dry-run

# molecule-only / 旧 wo_SAND 对照
python scripts/reproduce/run_antibiotic_classification.py \
  --mode molecule-only --test-group 0 --fold 0 --dry-run
```

完整协议、历史行为和证据边界见
`experiments/fig1b_antibiotic_classification/README.md`。

## Hierarchical MIC holdout

```bash
PYTHONHASHSEED=0 python scripts/reproduce/run_hierarchical_mic.py \
  --protocol strain \
  --test-group 0 \
  --acknowledge-dynamic-legacy-split \
  --dry-run
```

`--protocol` 可选 `strain`、`species` 或 `phylum`。历史 split 的 process hash seed 没有保存，
因此必须显式确认该 provenance 限制。详见 `experiments/hierarchical_mic/README.md`。

## Fig. 2b shared benchmark

正式 7-model × 5-fold 结果和复现入口见
`experiments/fig2b_molecule_encoders/README.md`。当前正式 mean R² ± sample SD 为：

- DLM MTR+DLM: `0.5386 ± 0.0250`
- ChemBERTa MTR: `0.4172 ± 0.0275`
- APEX: `0.4014 ± 0.0146`
- PeptideCLM: `0.3836 ± 0.0244`
- DLM-only: `0.3765 ± 0.0239`
- MolFormer: `0.3678 ± 0.0198`
- ChemBERTa MLM: `0.2247 ± 0.0131`

## 测试

默认使用 conda `base` 环境：

```bash
conda run -n base pytest -q
```

需要 CUDA autocast 的集成测试应在可见 NVIDIA device 的宿主环境运行。数据、embedding、
checkpoint 和大型结果文件不进入 Git；权重登记见 `configs/model_weights.yaml` 和
`MODEL_WEIGHTS.md`。

仍未完成或证据不足的部分包括正在运行的 fine-tuned Fig. 1b 完整 ensemble、synergy 的精确
历史 checkpoint 身份、modality ablation 的精确训练 checkpoint 血缘、Evo-2 extraction producer、
guided generation sampler 的自包含复现，以及产生论文 k-mer 单模型柱子的精确训练血缘。
synergy、modality 和 k-mer 的发布代码清理已经完成，但不能因此把候选血缘写成精确历史复现。
