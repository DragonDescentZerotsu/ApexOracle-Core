# 实验与分析索引

`experiments/` 按研究主题组织实验方法、统计结果和可视化。各子目录的 README 与 manifest
说明输入、运行入口、参数、输出位置和验证方法。

## 实验目录

| 目录 | 内容 | 入口说明 |
| --- | --- | --- |
| `peptide_classifier/` | Peptide classifier 的数据来源、checkpoint 身份与评估协议 | `peptide_classifier/README.md` |
| `remasking_schedule_reviewer/` | ReMDM window sweep、peptide correction 与结构分析 | `remasking_schedule_reviewer/README.md` |
| `peplink_validation/` | 氨基酸结构来源、peptide conversion 与 round-trip 验证 | `peplink_validation/README.md` |
| `reviewer4_unseen_targets/` | 基于 NCBI taxonomy 与 ATCC catalogue 的 unseen species/genus 筛选 | `reviewer4_unseen_targets/README.md` |
| `fig1b_antibiotic_classification/` | 三菌株小分子分类、10-member ensemble 与配对统计 | `fig1b_antibiotic_classification/README.md` |
| `fig2b_molecule_encoders/` | 共同 molecule IDs 和共享五折上的 molecular encoder 比较 | `fig2b_molecule_encoders/README.md` |
| `hierarchical_mic/` | MIC prediction、数据划分与敏感性分析 | `hierarchical_mic/README.md` |
| `genome_condition_reviewer/` | 同源基因组片段 variation 与 AMR/MGE linear probes | `genome_condition_reviewer/README.md` |
| `generated_candidate_diversity/` | 生成候选的序列与结构多样性 | `generated_candidate_diversity/README.md` |
| `sequence_similarity/` | Lead peptides 与训练集的序列相似性 | `sequence_similarity/README.md` |
| `synergy/` | 三折 synergy classification 与 checkpoint replay | `synergy/README.md` |

PepLink conversion 的参考实现为 0.1.2；方法、支持范围和统计定义见对应实验说明。
