# `DataPrepare/` legacy 功能与清理 ledger

> 冻结日期：2026-08-10
> 来源基线：`core-pre-public-cleanup-2026-08-10` → `56c57e5`

本文逐项登记当前 46 个 tracked `DataPrepare/` 文件。完整 SHA-256 位于
`reproducibility/dataprepare_legacy_source_2026-08-10.sha256`；原始内容同时由 annotated tag
`legacy-code-snapshot-2026-07-17`、archive branch 与新的 pre-public cleanup tag 恢复。

Disposition 含义：

- `remove`：功能已迁移、是一次性探索/debug、为空文件，或只保留历史价值；从 active tree 删除。
- `migrated/remove`：独立功能已经迁为参数化 canonical 入口并验证；旧文件下一批删除。
- `license-hold/remove`：脚本本身不作为公共 API；其数据库派生输入/输出受 Core data-license 决策约束，
  完成公开边界记录后删除，不能把旧脚本的可运行性当成数据再分发许可。

执行状态（2026-08-10）：表中 46 个 disposition 已全部履行，tracked legacy source 已从 active tree
移除；`DataPrepare/Data/` 下 ignored 数据、文本、embedding 和 reviewer assets 原地保留。下表继续记录
删除前角色和处置理由，不表示这些路径仍存在于默认分支。

## 逐文件 ledger

| Legacy 文件 | 已验证功能 | Canonical destination / 证据 | Disposition |
| --- | --- | --- | --- |
| `APEX_in_house_to_SMILES.py` | 将旧 APEX in-house peptide 表转为 SMILES，并处理 sentinel MIC | `src/apexoracle/data/amp_mic.py`、`amp_training_data.py`、PepLink 0.1.2；paper reconstruction audit | `remove` |
| `APEX_in_house_to_SMILES_merge_w_DBAASP.py` | 上述转换并与 DBAASP peptide 表合并 | 同上；只读 canonical builder 已验证 15,718-row reconstruction | `remove` |
| `ATCC_genome_annotation_get.py` | 使用 ATCC API 下载指定 strain annotation；顶层硬编码 BAA-3170/3197 | Genome/annotation 作为外部受控资产；Core 不发布 ATCC credential workflow | `remove` |
| `ATCC_genome_get.py` | 使用 ATCC API 下载指定 strain genome；与上一文件高度重复 | `ApexOracle-Evo2` 只消费用户提供 FASTA；Core data manifest 负责 provenance | `remove` |
| `DBAASP_SELFIES_Token_see.py` | 查看单个 tokenized DBAASP CSV 的长度分布 | Fig. 2b/loader eligibility 与 max-length contracts 已在 canonical tests 固定 | `remove` |
| `DataCheck.ipynb` | 193-cell 无输出探索 notebook；串联 DBAASP、AA chemistry、MIC、small molecule、synergy、taxonomy 等历史数据检查 | `experiments/amp_data_pipeline/`、small-molecule audit、各 canonical builders 和 recovery tags | `remove` |
| `DownloadAllPep.py` | 从 DBAASP API 拉取全量 peptide JSON 到作者机器路径 | 输入 hash/provenance 已冻结；官方条款存在冲突，不能以此脚本替代许可决定 | `license-hold/remove` |
| `DownloadOnePep.py` | 按 peptide ID 下载单条 DBAASP JSON 到作者机器路径 | 必要 record links 保留在 reviewer Supplementary Data；非公共 API | `license-hold/remove` |
| `ExtractSynData.py` | 从 DBAASP JSON 粗筛具有 synergy records 的 peptide IDs | 论文最终 synergy table、eligible counts 和 checkpoint family 已冻结 | `license-hold/remove` |
| `Get_text_embedding.py` | Ben 测试目录的 Med-LLaMA3 倒数第二层 text embedding producer，含绝对路径 | `src/apexoracle/features/strain_text.py` + `scripts/prepare_data/embed_strain_texts.py`；ATCC real parity | `migrated/remove` |
| `Get_text_embedding_wo_genome.py` | 对 text-only strain descriptions 替换精确 strain 名后提取 Med-LLaMA3 倒数第二层 token embeddings | 同一 canonical producer；text-only real parity exact | `migrated/remove` |
| `MDLM/MDLM_data.ipynb` | PubChem/SmProt/UniProt/UniRef/CycloPS peptide 预训练语料探索与去重 | 合作者 producer 归 `ApexOracle-DLM-Pretraining`；不属于 Core | `remove` |
| `MDLM/debug_notebook.py` | 6 行 tensor debug scratch | 无独立功能 | `remove` |
| `MDLM/filter_non_209.py` | 把 descriptor dataset 投影到固定 209 维 | DLM-pretraining 的 209-descriptor contract 与统计文件已冻结 | `remove` |
| `MDLM/split_selfies_csv_file.py` | 将大型 SELFIES CSV 分片 | 预训练数据运维，不属于 Core；无 Core caller | `remove` |
| `MDLM/tokenize_SELFIES_descriptors_hf.py` | 多进程 tokenization、descriptor 数据集写入 Hugging Face disk format | 合作者 pretraining producer 边界；Core 不维护第二份 | `remove` |
| `Morgan_Fingerprint_Similarity.py` | 旧 Morgan/Tanimoto 两两相似度脚本 | `src/apexoracle/evaluation/generated_candidate_diversity.py` 与 canonical reviewer audit | `remove` |
| `Morgan_fingerprint_sim_generation.py` | 对硬编码 generation outputs 做 fingerprint 分布并画图 | canonical exact 84,226-output/all-pairs diversity workflow | `remove` |
| `Morgan_fingerprint_sim_generation_SM_rediscover.py` | 上一脚本的 small-molecule rediscovery 变体 | canonical diversity library/figure；无独立 release caller | `remove` |
| `__init__.py` | 空 package marker | 删除整个 legacy package 后不再需要 | `remove` |
| `aa_seq_to_smiles.py` | 2,326 行历史 DBAASP unusual-residue/peptide SMILES builder | PepLink 0.1.2、AA lineage audit 和 15,718-row compatibility evidence | `remove` |
| `bacteria_get.py` | 统计 DBAASP target bacteria/strain 频次 | canonical data manifests、strain mapping 和 experiment summaries | `remove` |
| `canonical-peptide-check.py` | RDKit peptide-bond切分/20 AA 子结构判断 prototype，文件末尾直接执行示例 | PepLink 0.1.2 与 MDLM canonical peptide parser；prototype 不作为判定标准 | `remove` |
| `clean_smiles_compare.py` | 比较 stereocenters 的一次性诊断 | AA chemistry audit/structure-aware canonicalization 已接管 | `remove` |
| `compare_all_mol_diff.py` | molecule pair 结构差异的串行旧实现 | canonical fingerprint/diversity and sequence-similarity modules | `remove` |
| `compare_all_mol_diff_parallel.py` | 上述 pair/window comparison 的多进程变体 | 同上；无公共 caller | `remove` |
| `complete_compare_smiles_MIC.py` | 给 pair comparison 表补 mean MIC 并按差异筛选 | 历史 Fig. 2b exploration，不属于最终 benchmark protocol | `remove` |
| `concentration_unit_transfer.py` | DBAASP concentration/MIC 单位和 censor 转换早期实现 | `src/apexoracle/data/amp_mic.py` 和 censor sensitivity tests | `remove` |
| `concentration_unit_transfer_all_bact.py` | 全 bacteria MIC 转换与 mean 统计变体 | canonical paper-mode builder 与 exact reconstruction audit | `remove` |
| `concentration_unit_transfer_new.py` | 19 target strains MIC 转换变体 | 同上 | `remove` |
| `correct_SMILES_offered_by_DBAASP.py` | 对 DBAASP-offered peptide SMILES 重新按 residue/手性构建 | PepLink 0.1.2 + source-aware chemistry audit | `remove` |
| `debug.py` | 为一次 MolPort/milk-style scan 抽取 small-molecule SMILES，含两个绝对路径 | 通用 catalogue conversion/matching 已在 MDLM clean module；case provenance 留文档 | `remove` |
| `debug_notebook.py` | 8 行 dataframe scratch | 无独立功能 | `remove` |
| `discription_generation.py` | 调用旧 DashScope/Qwen 生成 ATCC strain descriptions | 历史文本作为 data asset；退役模型调用不作为可复现 producer | `remove` |
| `discription_generation_w_ATCC.py` | 与上一文件 byte-identical duplicate | 同一 SHA-256 已证明重复 | `remove` |
| `discription_generation_wo_ATCC.py` | text-only strain mapping、judge text 和 description generation 混合 driver | strain mapping 已迁入 `src/apexoracle/data/strain_mapping.py`；文本资产单独登记 | `remove` |
| `get_synergy.py` | 从 DBAASP synergy records 解析 FICI、查询 antimicrobial structures、生成早期 pair table | 最终 4,285-row input、2,732 eligible rows 和 canonical CV 已冻结 | `license-hold/remove` |
| `get_synergy_Evo.py` | 上述 producer 的 Evo/text-availability 扩展，含 LLM judge 和多个中间写入 | 最终 paper input/checkpoint evidence；原 producer 只由 recovery tag追溯 | `license-hold/remove` |
| `group_peptides.py` | BLOSUM50 Smith-Waterman 全量 peptide grouping，硬编码 230 workers | canonical `evaluation/sequence_similarity/` 使用冻结 BLOSUM62/paper protocol | `remove` |
| `rename_judge_text_file.py` | 将 judge-text filename 做一次性替换，含绝对目录 | mapping/filename rules 已进入 canonical strain contracts | `remove` |
| `resistant_gene_check.py` | 用 Gemini 对 genome annotation 文本判断耐药基因；key 从环境变量读取 | 正式 AMR/MGE probes 使用 GenBank annotation + conservative dictionary，不使用 LLM judgement | `remove` |
| `smiles_to_peptide.py` | 历史 standard peptide SMILES→sequence parser | downstream MDLM 已迁为 `apexoracle_mdlm.chemistry.peptides`；历史 parser hash/行为由 MDLM recovery audit 保留 | `remove` |
| `text_generation` | 0-byte placeholder | 无功能 | `remove` |
| `train_genome_mcr_check.py` | 对训练 genome annotations 粗查 MCR/耐药相关文本 | canonical saved-window AMR/MGE probes 已接管公开分析 | `remove` |
| `try.py` | `aa_seq_to_smiles` 的一次性 reconstruction/debug caller | PepLink compatibility audit 已覆盖 | `remove` |
| `visualize_mol_diff.py` | RDKit atom difference 可视化探索 | canonical exact plotted-data/figure workflows；无公共 caller | `remove` |

## 依赖和删除顺序

已由 AST/import 与 repository reference scan 验证的 `DataPrepare` 内部依赖只有
`APEX_in_house_to_SMILES*.py`、`correct_SMILES_offered_by_DBAASP.py`、`try.py` 对
`aa_seq_to_smiles.py` 的 legacy import。它们必须作为同一批删除，不能先删共享文件后留下坏 caller。

唯一需要先迁移的公共功能族是两份 text embedding producer，现已关闭。新入口：

1. 接受显式 input/output directory、device、model ID/revision、layer 和 overwrite policy；
2. 默认固定 `YBXL/Med-LLaMA3-8B` revision
   `567e7e71d8b6b433d8bc494f8112176bec4afccf`；
3. 保留历史精确 strain-name → `This strain` 两步 replacement；
4. 保存每个 `.pt` 的 source text SHA-256、resolved model revision、shape 和 output SHA-256 manifest；
5. 已用 7 项 synthetic tests 固定 selection、replacement、layer、filename 和 local-cache contract；
   两条真实 historical text 完成 H100 parity，其中 text-only tensor 逐元素相等，ATCC tensor shape/dtype
   一致并在 `rtol=1e-5, atol=1e-4` 下 allclose。机器可读证据见
   `reproducibility/strain_text_embedding_parity_2026-08-10.json`。

迁移通过后，46/46 个 legacy files 已从 active tree 删除；raw data、embedding、checkpoint 和 ignored
reviewer assets 未随源码删除。删除按显式路径执行，没有使用 `git add -A`；全仓 tests、wheel/sdist、
fresh-clone install/import/CLI 和 absolute-path scan 是本批提交前的剩余 release gates。
