# 实验与 reviewer 审计索引

`experiments/` 保存可复现结果、统计摘要和 reviewer-facing 审计证据；原始数据、模型权重、投稿
DOCX/PDF 和临时运行目录不进入本仓库。每个子目录均应有 README 或机器可读 manifest，说明输入、
输出、边界和复现入口。

## 当前 reviewer 资源

| 目录 | 内容 | Canonical 入口 |
| --- | --- | --- |
| `peplink_validation/` | PepLink 0.1.2 round-trip、ChatGPT-o1/OPSIN 化学审计、AA/peptide 血缘及 56-peptide Supplementary Data | `peplink_validation/README.md` |
| `reviewer4_unseen_targets/` | NCBI current taxonomy + ATCC catalogue 的 unseen species/genus 可购买靶点初筛 | `reviewer4_unseen_targets/README.md` |
| `fig1b_antibiotic_classification/` | 三菌株小分子分类的 10-member ensemble、paired statistics 与 reviewer 文稿 | `fig1b_antibiotic_classification/README.md` |
| `fig2b_molecule_encoders/` | 共享分子五折 encoder benchmark | `fig2b_molecule_encoders/README.md` |
| `evo2_genome_embeddings/` | 预计算 Evo-2 embedding 的身份、消费范围和 scaling 审计 | `evo2_genome_embeddings/README.md` |

## 发布边界

- `peplink_validation/supplementary_data/` 中的英文 Excel 是后续投稿文件；中文 Excel 是逐字段核对版。
- 公开 reviewer 表格仅在 `.gitignore` 中按精确目录显式放行，避免全局放开 CSV/XLSX。
- `DataPrepare/Data/private_inhouse_amp/` 的原始 workbook 及其逐 assay 派生文件始终被忽略，不得提交。
- 已退役的 PepLink 0.1.1/dev exploratory outputs 不作为当前 reviewer 证据；正式结果只引用
  `peplink_validation/peplink_0.1.2/`。
- 结果 manifest 中的 SHA-256 用于验证本地冻结输入；数据二进制本身仍遵循根目录 `.gitignore`。
