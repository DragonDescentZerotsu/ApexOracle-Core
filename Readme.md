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

最近完成的 reviewer 修订包括：

- Fig. 1b 的三个 target 均已完成固定 10-member/fold 的 ApexOracle fine-tune 和 matched
  Chemprop baseline，最终 AUPRC-only panel、paired significance 和文稿已同步；
- PepLink 0.1.2 的 AA→SELFIES→AA round-trip validation，以及历史
  ChatGPT-o1/OPSIN residue-definition 审计和 reviewer Supplementary Data；
- Reviewer 4 unseen-species 的公开候选筛选；后续 generation 仍需作者选择 target、
  multi-isolate panel 并准备 exact-target embedding。

机器职责、conda/venv、共享文件系统、数据/权重和外部仓库位置统一记录在
`docs/COMPUTE_AND_ASSET_MAP.md`。

PepLink 作为独立发布的 peptide↔SMILES 工具，不复制进本仓库。ApexOracle 固定可选依赖
`PepLink==0.1.2` 并通过薄 adapter 调用。0.1.1 的 179 条历史 structure-correction
兼容性审计、0.1.2 的 round-trip 结果和 AA 数据血缘分别见
`experiments/amp_data_pipeline/README.md` 与 `experiments/peplink_validation/README.md`。

## Canonical 与 legacy 边界

新运行和复现应优先使用 `src/apexoracle/`、`scripts/prepare_data/`、
`scripts/reproduce/`、`configs/` 和 `experiments/` 中记录的 canonical 入口。
`DataPrepare/` 的 46-file ledger、源码 hashes 和恢复点已冻结。其唯一仍有独立公共价值的
strain-text embedding producer 已迁入 `apexoracle.features.strain_text` 与
`scripts/prepare_data/embed_strain_texts.py`；其余 paper-era、debug 和外模块副本不再作为 active API，
原始版本由 recovery tags 恢复。

## Strain-text embedding

```bash
python scripts/prepare_data/embed_strain_texts.py \
  --input-dir /path/to/text \
  --output-dir /path/to/embeddings \
  --device cuda:0 \
  --local-files-only
```

入口固定 Med-LLaMA3 model revision、历史 `This strain` replacement、倒数第二层和
`[tokens, 4096]` float32 tensor contract，并输出逐文件 hash manifest。完整参数和真实历史 tensor
parity 见 `scripts/prepare_data/README.md`。

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

仍未完成或证据不足的部分包括 Fig. 2b 的容量匹配 objective comparison、部分历史 checkpoint
身份、modality ablation 与论文 k-mer 单模型柱子的精确训练血缘、Evo-2 extraction producer、
guided generation sampler 的自包含复现，以及 corrected AA successor dataset 的重新训练。
synergy、modality、k-mer 和 Fig. 1b 的发布代码及 reviewer 补实验已经完成，但不能因此把
证据不足的历史 producer 或 checkpoint 血缘写成精确复现。

旧 `DataPrepare/` active source 的最终删除仍需通过全仓测试、build 和 fresh-clone gate；它不阻塞当前
reviewer Supplementary Data 或 canonical runner 的使用。
