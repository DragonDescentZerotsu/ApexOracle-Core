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
