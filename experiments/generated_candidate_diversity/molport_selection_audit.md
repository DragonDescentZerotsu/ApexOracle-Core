# MolPort small-molecule selection audit

本审计用于回答 reviewer 关于 18 条 purchased compounds 的 screening denominator、predicted-MIC
范围、结构/化学风险筛选和采购缩减过程。外部 `mdlm` checkout 与 Mac 历史资产均只读核验，未改写。

## 已由代码和历史资产验证的事实

1. 两份冻结 prediction table 对同一组 44,608 条唯一 SMILES entries 分别给出 `BAA-3170` 与
   `BAA-3197` 的预测 MIC。这些 entries 对应 39,995 个 RDKit-valid canonical isomeric structures，
   另有 1 条不能被当前 RDKit 解析。以 `predicted MIC <=15 µM` 过滤后分别保留 1,554 和 395 行；按
   RDKit canonical isomeric SMILES 计为 1,526 和 387 个结构，两菌株并集为 1,535 个结构。
2. 44,608 的上游是论文 Fig. 1b 使用的三套公开 small-molecule antibiotic classification data，
   不是生成输出，也不是 MolPort 下载：*E. coli* BW25113 2,335 rows、*A. baumannii* ATCC 17978
   7,684 rows、*S. aureus* RN4220 39,312 rows，合并为 49,331 个 molecule--strain rows。
   `DataPrepare/debug.py` 只读取该合并表的 `SMILES` column，将全部 49,331 行逐条编码为 SELFIES；
   原始表的 binary activity labels 不进入本次 unseen-target MIC scoring。
3. 两个 target input files 各有 49,331 行且 SHA-256 完全相同
   (`56607905...457ab`)。其中有 44,608 个唯一 SELFIES；历史 inference/export 逻辑以
   `dict`/`set` 汇总相同 SELFIES，因此两份 prediction CSV 各有 44,608 entries。逐集合复核显示，
   每份 prediction CSV 的 `SMILES_Sequence` 都与输入文件的 44,608 个唯一 SELFIES 解码结果完全
   一致。这 44,608 entries 进一步对应 39,995 个 RDKit-valid canonical isomeric structures，且该
   canonical set 与 `small_molecule_Evo_binary_data.csv` 完全一致。
4. 历史 MolPort snapshot 的 12 个 catalogue files 共含 5,887,458 rows / 唯一 MolPort IDs。
   `/data2/tianang/projects/mdlm/match_molecules.py` 对上述低预测 MIC 结构与该 MolPort catalogue
   做 RDKit canonical-isomeric-SMILES exact match。输出为 276 个 `target × query × MolPort ID`
   rows，对应 179 个唯一 MolPort ID / canonical structures；其中 `BAA-3170` 和 `BAA-3197`
   分别覆盖 177 和 90 个结构。
5. `/data2/tianang/projects/mdlm/temp_data/small_molecules/cluster_filter.py` 使用 radius-2、
   2,048-bit Morgan fingerprints 和 Tanimoto 0.75 阈值进行 Butina clustering，并标记 PAINS A/B/C、
   BRENK 以及 nitro、azide、isocyanate、aldehyde、Michael acceptor 和 quinone-like 自定义 SMARTS。
   179 个唯一结构形成 167 clusters；99 个结构命中至少一个 alert，80 个未命中。
6. Mac 上人工审核表
   `/Users/kirianozan/Documents/Study/Penn/projects/mdlm/temp_data/small_molecules/filtered_molecule_by_marcelo_1.csv`
   含 43 个唯一 MolPort IDs。该文件保留了 alert columns，其中 24 个有 alert、19 个无 alert；
   因此它是人工审核资产，但不能描述成“alert filter 后的 43 条”。
7. 2026-01-27 MolPort quote
   `/Users/kirianozan/Downloads/Quote_LPA27A4653274.pdf` 含 19 个全部 in-stock 的候选。19 个结构均
   未命中上述 alerts，并分别位于 19 个 Butina clusters。报价中 18 个单价为 USD 5--90，
   `Molport-002-070-273` 单价为 USD 950；交期均为 14--21 business days。
8. 正式 Supplementary heatmap 的 18 个 MolPort IDs 与 quote 相比恰好只缺少
   `Molport-002-070-273`。所以“19 条进入采购报价，因一条价格显著高于其他条目而最终购买
   18 条”由 quote 和最终 figure 共同直接支持。最终 18 条均无 structural alert，分别位于
   18 个 Butina clusters。
9. 最终 18 条在已保留的 target--compound prediction rows 中覆盖 0.898--12.0 µM；若每个结构
   取两个 target 中较低的 prediction，则范围为 0.898--5.969 µM。

## 根据现有证据作出的推断

- 19 条 quote-stage structures 的低预测 MIC、无 alert、19/19 cluster coverage 和全部在库状态，
  支持将这一步概括为 predicted activity、chemical-liability、structural diversity 与 procurement
  feasibility 的联合 prioritization。
- 43-row collaborator table、19-row quote 与最终 18-row figure 共同说明存在人工 expert review；
  但 19 条不是 43 条的严格子集（17 条重合，2 条来自 43 条之外），因此论文不应写成简单的
  `43 -> 19 -> 18` 严格流水线。

## 仍待作者确认的事项与文稿边界

- 当前没有找到能证明“因运输时间过长排除若干候选”的早期 quote/cart。现有 quote 中最终保留的
  化合物本身包含完整 14--21 business-day 范围。因此 reviewer response 可写 procurement
  feasibility 和 cost，不应把 long delivery time 写成已验证的排除原因。
- 44,608 是冻结 prediction tables 中的 SMILES-entry denominator，39,995 才是对应的 canonical
  structure denominator。MolPort snapshot 的 catalogue denominator 是 5,887,458 IDs，但模型并未
  对这 588 万条逐条评分；论文应写先预测 processed screening collection，再与 MolPort catalogue
  exact-match，而不要写成对整个 MolPort commercial library 逐条预测。
- shell history 验证了 `python DataPrepare/debug.py`、两个 target-specific tmux sessions、
  `python temp_judge_generated_mols_MIC.py` 及后续 MolPort matching 的执行顺序，但没有保存带
  timestamp 的完整 stdout/stderr 或当时 mutable script 的 byte-exact snapshot。输入、输出和集合
  血缘已由文件内容直接闭合；文稿不应额外声称完整恢复了原 producer command line。
- structural alerts 是计算筛选规则，不等同于实验毒性测试；应写 chemical-liability / reactive or
  assay-interfering structural alerts，不应写成“已证明对 mammalian cells 无毒”。

## 建议写入 Methods 的段落

> **Small-molecule virtual screening and procurement.** We assembled the
> screening collection from the three public small-molecule antibiotic datasets
> used in the molecular-classification benchmark (*E. coli* BW25113,
> *A. baumannii* ATCC 17978 and *S. aureus* RN4220). The merged table contained
> 49,331 molecule--strain rows. After SMILES-to-SELFIES conversion and
> consolidation of repeated SELFIES representations, the saved screening table
> contained 44,608 entries representing 39,995 distinct RDKit-valid canonical
> isomeric structures. The same molecular set was scored against each unseen
> target strain. Applying a predicted-MIC cutoff of
> <=15 µM retained 1,554 entries for *E. coli* AR-0349 and 395 for
> *P. aeruginosa* PA5257; after canonicalization, the union
> comprised 1,535 distinct structures. Rather than scoring every MolPort entry,
> we then matched RDKit canonical isomeric SMILES against a 5,887,458-entry
> MolPort catalogue snapshot and identified 179 commercially
> available structures. These were grouped by Butina clustering of radius-2,
> 2,048-bit Morgan fingerprints at a Tanimoto threshold of 0.75 and screened for
> PAINS, BRENK and additional reactive or assay-interfering structural alerts.
> Eighty structures had no detected alert. Nineteen compounds occupying 19
> distinct clusters were advanced to procurement review on the basis of
> predicted activity, structural diversity and practical availability. One was
> not purchased because its quoted cost substantially exceeded that of the
> other compounds, leaving 18 compounds for experimental testing. Across the
> retained target--compound predictions for these 18 compounds, predicted MICs
> ranged from 0.90 to 12.0 µM. Predicted MIC was used only for computational
> prioritization and was not interpreted as experimental activity.

## Frozen source checksums

- `SMs_mic_predictions_BAA-3170.csv`: `6c6b9e99...ac2f9e`
- `SMs_mic_predictions_BAA-3197.csv`: `63850d62...cd06e`
- `purchasable_molecules_match.csv`: `03c6529e...a5ab9`
- `purchasable_molecules_match_with_clusters_flags.csv`: `a7619da1...aa7b6`
- Mac `filtered_molecule_by_marcelo_1.csv`: `aace0772...c9df6`
- Mac MolPort quote PDF: `2f53826d...1b3743f`
- `Fig_SI_heatmap.pdf`: `42a9624c...cbc3b1`（本机正式文稿与 Mac 原图一致）
