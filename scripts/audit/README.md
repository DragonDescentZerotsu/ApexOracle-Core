# Reviewer 审计脚本

本目录脚本默认只读项目数据，并把输出写入 `experiments/` 或明确指定的 `--output-dir`。需要 import
`apexoracle` 的入口从仓库根目录以 `PYTHONPATH=src` 运行。

## PepLink 与 AA chemistry

| 脚本 | 作用 | 主要输出 |
| --- | --- | --- |
| `audit_peplink_roundtrip_validation.py` | AA/peptide → SELFIES structural round-trip 与受支持 annotation reverse contract | `experiments/peplink_validation/peplink_0.1.2/` |
| `audit_chatgpt_opsin_noncanonical_aas.py` | 重建 169 条 ChatGPT-o1/OPSIN 血缘并合并人工判定 | `chatgpt_opsin_chemical_validation.csv` |
| `recalculate_reviewer_peptide_scope.py` | 与 canonical `<=512` loader pool 相交，得到 56 peptide / 219 MIC row | `recalculated_local_error_scope.json` |
| `build_peplink_supplementary_data.py` | 生成 56-row 英文/中文 Supplementary Data、18-definition summary 和 SHA manifest | `experiments/peplink_validation/supplementary_data/` |

`audit_peplink_roundtrip_validation.py` 默认在仓库同级目录寻找 `PepLink`，也可设置
`PEPLINK_SOURCE=/path/to/PepLink`。正式复现必须使用 PepLink 0.1.2。

## Reviewer 4 unseen-target screening

| 脚本 | 作用 | 发布边界 |
| --- | --- | --- |
| `audit_reviewer4_inhouse_species_coverage.py` | 将私有 in-house assay headers 与实际 guidance training exposure 比较 | 私有 workbook 和逐 assay 输出不提交 |
| `audit_reviewer4_unseen_atcc_pathogens.py` | 用 NCBI current taxonomy 与公开 ATCC catalogue 筛选 unseen species/genus | 公开候选表发布于 `experiments/reviewer4_unseen_targets/` |

in-house 脚本默认在同级外部 `mdlm` checkout 查找 guidance producer，也可设置
`MDLM_GUIDANCE_PRODUCER=/path/to/guaidance_regressor_all_data_pad_no_mask.py`。

## Hierarchical MIC split 与 molecule overlap

| 脚本 | 作用 | 主要输出 |
| --- | --- | --- |
| `audit_hierarchical_mic_molecule_overlap.py` | 在 strain/species/phylum pathogen holdout 中审计 train-seen 与 exact-molecule-disjoint test measurements | `experiments/hierarchical_mic/molecule_overlap/overlap_{by_group.csv,audit.json}` |
| `plot_hierarchical_mic_test_distribution.py` | 绘制 fixed strain-wise 实际 held-out MIC 的 pooled histogram 与逐 fold ECDF，并审计 MIC<=16 micromolar 比例；最终样式无总标题、panel标题不加粗并带`a/b`标记 | `experiments/hierarchical_mic/mic_distribution/` |

正式命令：

```bash
PYTHONPATH=src python scripts/audit/audit_hierarchical_mic_molecule_overlap.py \
  --protocol all
```

strain 使用 `experiments/hierarchical_mic/strain/legacy_protocol_manifest.json` 中冻结的
`PYTHONHASHSEED=0` candidate，并明确标记为非精确 2025 membership；species/phylum 使用
canonical taxonomy-cluster adapters。输出同时报告 `DBAASP_id` overlap 和模型实际 stored-token
input 的 SHA-256 identity，后者是 molecule-disjoint sensitivity 的主要过滤口径。

MIC distribution 探索图的正式命令：

```bash
python scripts/audit/plot_hierarchical_mic_test_distribution.py
```

默认直接消费 fixed strain-wise 七成员 replay 的逐测量 ensemble 表；不会重建 split、训练模型或
修改论文图片。PNG/PDF 用于作者审阅，summary/bin CSV 与 manifest 记录精确输入血缘。

## ReMDM remasking schedule reviewer figure

### Peptide-label structure audit

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/audit_remasking_peptide_classifier_structure.py
```

`audit_remasking_peptide_classifier_structure.py` 读取冻结的 3,600 raw attempts 和原
`evaluated_attempts.csv`，用相同历史 v1 checkpoint 复算概率，比较完整 token 与 first `[SEP]`
后统一 PAD 的输入，并用 runner SMARTS、RDKit general amide 和两个 reviewer-retrained heads
审计 classifier label。compact JSON 写入
`experiments/remasking_schedule_reviewer/analysis/peptide_structure_audit/summary.json`；逐 valid
row CSV 保持本地。该入口不重新生成 molecule、不重算 MIC，也不把 amide count 定义成 peptide
truth。当前结论和 reviewer-facing 使用边界见
`experiments/remasking_schedule_reviewer/STRUCTURE_AUDIT.md`。

历史 classifier-only 四面板图暂停用于 reviewer：structure audit 已证明历史 v1
classifier-positive 不能充当独立 structure identity。该 legacy 图及 violin 变体只用于内部追溯；
当前正式 reviewer 图是下一节的 structure-qualified 三面板版本，其中 peptide yield 使用联合
口径，all-RDKit-valid predicted MIC 则明确作为独立的候选总体指标展示。

### Canonical final reviewer figure

```bash
MPLBACKEND=Agg \
  /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/plot_remasking_structure_qualified_peptides.py
```

`plot_remasking_structure_qualified_peptides.py` 从冻结的 structure-audit rows 重新解析分子，并按
当前作者讨论的窄口径绘图：至少一个 RDKit general amide、至少两个
`N–Cα–C(=O)` residue-like motifs、单一组分、无自由基；明确允许 B 和卤素。常见金属候选进入
单独的 manual-review bucket，不计入主 peptide 数；其他异常元素直接排除。图中另外展示
first `[SEP]` 后正确 PAD 的历史 v1 classifier positive 作为 supporting subset，而不是独立
structure truth。默认 `--layout with-context` 同时保留原 window-wise all-RDKit-valid median
predicted-MIC panel，以及 current-vs-no-guidance dumbbell panel；后者的第三行改用新的
peptide yield。默认三面板的 panel a 只绘制同时通过 structure screen 和
SEP-padded classifier 的最严格单系列；三个 panel 单行排列且标题居中。panel c 删除历史 v1
classifier-positive 独立行；在同一个三行 dot-and-interval 坐标区中展示 RDKit-valid yield、
peptide yield 和 with/without peptide guidance 的 all-RDKit-valid median predicted MIC。
yield 误差条为三个 seed-level pooled rates 的 sample s.d.，
predicted-MIC 误差条为三个 seed-level pooled median 的 sample s.d.；图例只保留在
panel c。
panel b 的柱高仍为 all-RDKit-valid pooled median MIC，并以三个 seed-level pooled median MIC
的 sample s.d. 为纵向误差条。panel c 图例只保留简短组名，`gamma_peptide` 细节移入 caption；
数值轴的5--25区间以显式断轴压缩，每行标签分别标明 `%` 或 $\mu\mathrm{M}$ 单位。总标题、
panel subtitle 和底部说明均移到
独立 `*_caption.md`。`--layout yield-only` 可复现前一版两面板图。输出 PDF/SVG/PNG、
exact plotted-data CSV 和 SHA-256 manifest 到
`experiments/remasking_schedule_reviewer/figures/`，两种 layout 使用不同 stem、不互相覆盖。
默认 `with-context` stem 已于 2026-07-29 被作者确认为最终 reviewer 图；`yield-only` 和历史
四面板图均作为 legacy 原地保留。图稿 final 不改变科学边界：该定义仍是 preliminary narrow
screen，未经已知 peptide/small-molecule controls 验证前不得称为通用 ground truth。完整
canonical/legacy 索引与 storage policy 见
`experiments/remasking_schedule_reviewer/figures/README.md`。

```bash
# Pooled-median bar version
python scripts/audit/plot_remasking_schedule_reviewer.py

# Preserve the bar version and create a second four-panel figure whose panel b
# shows the full valid-MIC distribution
python scripts/audit/plot_remasking_schedule_reviewer.py \
  --panel-b-style violin
```

该入口读取冻结的
`experiments/remasking_schedule_reviewer/analysis/summary.json`，输出历史
PDF/SVG/PNG、exact plotted-data CSV 和 source/script/output SHA-256 manifest 到
`experiments/remasking_schedule_reviewer/figures/`。四个 panel 展示 window sensitivity 的
valid classifier-positive yield 与 clean-model predicted MIC、current-window
`gamma_peptide=15` vs `gamma_peptide=0` direct control，以及 RDKit-valid candidates 中的
classifier-positive/negative composition。该图是描述性汇总，不重新计算 classifier label 或
predicted MIC。默认 bar 版本的 panel a 以三个 seed-level pooled rates 的 sample s.d. 为
error bar；panel c 只显示冻结的 descriptive effect sizes。六个 matched
`strain × seed` tasks 的 two-sided exact paired sign-flip test 保留在 plotted-data CSV 和
manifest 中用于内部审计，不显示在 reviewer-facing figure。violin 版本另读取冻结的本地
`analysis/evaluated_attempts.csv`，在 log scale 上显示全部 RDKit-valid finite predicted MIC；
不会覆盖默认 bar 输出。

## Reviewer 2 peptide classifier split

| 脚本 | 作用 | 主要输出 |
| --- | --- | --- |
| `audit_peptide_classifier_split.py` | 逐文件核对 split 资产 SHA-256，并从 canonical molecule digest 与 real-peptide sequence component 重新计算全部 82,795,051 个 split code | `experiments/peptide_classifier/reviewer_retrain/split_audit.json` |

正式命令：

```bash
PYTHONPATH=src python scripts/audit/audit_peptide_classifier_split.py \
  --split-dir experiments/peptide_classifier/reviewer_retrain
```

审计通过表示 canonical molecule 和 sequence component 的确定性 group assignment 可逐行复算；
不能把 v1 来源标签重新解释为 v2 parser 标签。
