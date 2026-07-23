# PepLink round-trip、AA 化学核验与 peptide 血缘

本目录的 reviewer-facing canonical 口径已经按完整 structure source lineage 修订。先阅读：

1. `AA_AND_PEPTIDE_LINEAGE_ZH.md`：AA definition 与 whole-peptide structure 的中文血缘；
2. `reviewer_response_scope_summary.json`：机器可读范围；
3. `reviewer_response_draft.md`：可直接采用的 reviewer 回复；
4. `revised_training_impact_scope.csv`：确认问题的类别与 definition-level 计数；
5. `supplementary_data/Supplementary_Data_AA_conversion_errors.xlsx`：期刊用英文补充数据；
6. `supplementary_data/补充数据_AA转换错误_中文.xlsx`：逐字段中文镜像。

## 已验证的 round-trip

- 16,075/16,075 个可 forward construct peptide 通过 SMILES → SELFIES → SMILES molecular-graph
  exact round-trip；
- 已发布 PepLink 0.1.2 的公开 reverse contract 为 4,939/4,939；
- 其中支持范围内的 head-to-tail cyclic peptide 为 523/523；
- PepLink 0.1.2 对应 commit `90f627cc7fd65daaf9c5d0a973d17b79bcd097d5`、tag `v0.1.2`，
  测试为 23 passed。

0.1.1 的 1,210 个 Histidine reverse mismatch 来自后来新增 reverse parser 的 tautomer template
不一致，不影响论文的 forward data generation；reviewer 回复只汇报修复后的 0.1.2 正式结果。

## source-aware 训练影响口径

frozen MIC 表覆盖 16,430 个 DBAASP peptide、105,547 条 MIC record。完整来源包括：

- DBAASP-linked PubChem CID whole-peptide structure：840 peptide / 8,434 row；
- local residue-based builder：15,521 peptide / 96,747 row；
- DBAASP-offered structure branch：69 peptide / 366 row。

PubChem whole-peptide 支路按 DBAASP 已给定 CID 直接查询，不经过 ChatGPT-o1、OPSIN 或 PepLink，
因此 PepLink 当前无法从 annotation 重建并不意味着 frozen training structure 错误。

经过论文实际 strain/genome eligibility 且应用 `<=512` token loader filter 后，DBAASP pool 为
15,177 peptide / 74,103 row。coordination omission 按作者决定作为去金属预处理而不计错；DBAASP
sequence/unusual-residue annotation 内部不一致属于上游源数据质量问题，也不计入 reviewer 所问的
ChatGPT-o1/OPSIN/PepLink 转换错误。重新去重后：

- reviewer-facing 确认错误：56 peptide / 219 row（row prevalence 0.296%，报告为 0.30%）。

该 union 不包含 PubChem whole-peptide 记录、不包含仅因当前 PepLink API unsupported 的记录，也不
包含 non-exact polymer proxy。三类 reviewer-facing 确认错误之间没有 peptide 重叠。

2026-07-23 已将这 56 个 peptide 与 18 个确认错误 definition 一一映射。Supplementary Data
每个 peptide 一行，记录错误 residue code/位置、实际 loader MIC row 数、historical erroneous
SMILES/formula、corrected residue 或 terminal cap、corrected SMILES/formula、具体错误、处置方式和
证据链接；另有 18-row definition summary。16 个可直接匹配的 corrected definition 已再次通过
PubChem PUG REST 核对名称、分子式和结构；`NNar` 与 `D-3-OH-ASN` 由于没有 exact public compound
record，明确标为依据 DBAASP name/formula 的中等置信度判定，未指定的 stereocenter 保持未指定。
本表提供 residue/free-compound/cap 层级的正确版本，不代表 56 个完整 peptide 已经全部重新生成。
16 个 PubChem direct-match 的原始 PUG REST 响应冻结在
`supplementary_data/pubchem_corrected_definitions_20260723.json`，生成脚本会逐项断言 formula 和
structure（`D-End` 使用 PubChem L-form connectivity 加文献立体化学指认）。

## 459、251、44 与 169 的准确关系

459 条 mapping = 39 条标准 L/D + 420 条 noncanonical。420 条 noncanonical 的最终血缘为：

- 207 条保留的 PubChem name-lookup structure；
- 44 条在复查后进入第二条 GPT-refinement + OPSIN correction branch；
- 169 条无 PubChem 命中的主 ChatGPT-o1 + OPSIN branch。

因此，旧文档中“251 条都未经过 GPT/OPSIN”的说法已撤回。251 是初次 PubChem lookup success 数，
其中 44 条后来进入了第二条 GPT/OPSIN correction branch。

169 条主 branch 的人工判定原始记录继续保存在 `chatgpt_opsin_manual_review.csv` 和
`chatgpt_opsin_chemical_validation.csv`。它们是完整化学审计，不应直接当作训练影响范围：必须再与
完整 peptide source 和 model eligibility 相交。

## evaluation sensitivity 决定

evaluation-only sensitivity 原意是不重训，只从 ID-aligned held-out predictions 删除问题 peptide 后
重算指标。当前不执行/不声称这一分析，因为：

- historical hierarchical MIC 结果没有保存带 DBAASP ID 的逐行 prediction；
- strain-wise 2025 精确 split membership 未恢复；
- 即使只删测试行，也无法消除 training exposure，不能等同于 corrected-data retraining。

因此 reviewer 回复只报告 0.30% 的实测 prevalence 和 limitation，不写 `unchanged`、`robust` 或
`no effect`。

## 已退役但保留用于审计追溯的产物

- `flagged_peptide_exclusions.csv`：把主 169 branch 的所有人工 flag 合并得到的早期 conservative
  集合；不等于确认训练错误。
- `peplink_validation_sensitivity_exclusions.csv`：曾把上述 flag 与 PepLink forward failure 合并的
  探索性列表；已退役，不得用于 reviewer 回复或当前 0.30% 口径。
- `roundtrip_summary.json` 与根目录 `roundtrip_records.csv`：0.1.1 历史运行；正式引用
  `peplink_0.1.2/`。

## 复现入口

```bash
python scripts/audit/audit_peplink_roundtrip_validation.py \
  --output-dir experiments/peplink_validation/peplink_0.1.2 --workers 16

python scripts/audit/audit_chatgpt_opsin_noncanonical_aas.py \
  --manual-review experiments/peplink_validation/chatgpt_opsin_manual_review.csv \
  --output-dir experiments/peplink_validation

PYTHONPATH=src python scripts/audit/recalculate_reviewer_peptide_scope.py

PYTHONPATH=src python scripts/audit/build_peplink_supplementary_data.py
```

第二个入口仍复现 169-row 化学审计；reviewer-facing 的 source-aware 训练影响口径以
`reviewer_response_scope_summary.json`、`recalculated_local_error_scope.json` 和
`recalculated_local_error_peptides.csv` 为准。

2026-07-23 已将审计结果正式写入作者 manuscript checkout 中 `sn-article.tex` 的
ChatGPT-o1/OPSIN Methods 段落，并将精简英文回复写入同目录的 reviewer-response DOCX。两份投稿
文档不进入本源码仓库；TeX 临时编译为 28 页，DOCX 经 LibreOffice 临时渲染为 25 页，PepLink
回复文本完整且未改变页数。正式论文 PDF 未自动覆盖。
