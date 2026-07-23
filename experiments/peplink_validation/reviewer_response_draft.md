# Reviewer response draft：AA/SELFIES round-trip 与化学核验（修订口径）

## 当前建议

本回复只使用与最终模型输入血缘直接相关的证据，不再把 PepLink 0.1.2 的 355 个 forward
failure 当作训练数据错误，也不再使用 617/4,095 的旧 sensitivity 口径。原因是其中很大一部分
peptide 的 frozen SMILES 来自 DBAASP 所链接的 PubChem CID，PepLink 是否能从 annotation 重建它们
与模型实际消费的结构无关。

可以写“确认的本地化学转换错误只占实际 `<=512` token genome-or-text DBAASP pool 的 0.30%”，但
不能把它泛化为全部派生训练路径的统一比例，也不能据此写“因此对模型没有影响”。准确数字为：

- 56/15,177 peptide（0.369%）；
- 219/74,103 MIC record（0.296%，四舍五入为 0.30%）。

这里的 56/219 是 ChatGPT-o1/OPSIN-derived residue definition 或本地 residue template 的确认错误，
不包含 PubChem whole-peptide 记录、不包含仅因当前 PepLink API 不支持而失败的记录，也不包含
polymer proxy。coordination omission 按作者决定视为去金属/忽略配位预处理，不计为错误。20 个
source-ambiguous definition 和第二条 44-residue correction branch 中的 3 个
source/site conflict 单独列为 unresolved，不冒充已确认错误。

## “evaluation sensitivity”的含义与当前决定

此前建议的 evaluation sensitivity 是：不重训模型，只从已有 held-out row-level predictions 中删去
56 个问题 peptide 对应的行，再重算 R²、Spearman、Pearson 和不确定性。它只能回答“这些问题行
是否直接驱动测试集指标”，不能消除同一 peptide 在训练集中的暴露。

当前不能严谨执行这一分析：hierarchical MIC 的历史论文运行没有保存带 DBAASP ID 的逐行 held-out
prediction 表，而且 strain-wise 2025 split 的精确 membership 因未记录 `PYTHONHASHSEED` 已无法恢复。
现有 Fig. 1b 逐行 prediction 属于 small-molecule classification，不是这批 DBAASP MIC peptide。
因此本轮不在回复中声称做过 evaluation sensitivity，也不使用 `unchanged`、`robust` 或 `no effect`。

在没有时间全量重训时，最小且诚实的补救是：

1. 发布 PepLink 0.1.2 和完整 round-trip audit；
2. 提供 AA definition 与 whole-peptide structure 的来源级血缘；
3. 披露确认错误的 56/219（0.30%），并提供逐项处置表；
4. 在下一版数据中修正或排除这些记录，保留原 frozen dataset 仅用于复现既有模型；
5. 明确 0.30% 说明 prevalence 很低，但不是“零模型影响”的证明。

## 英文回复草稿

> We thank the reviewer for highlighting the need to validate amino-acid-to-structure conversion. We added two complementary validation layers. First, for every curated peptide that completed forward structure construction, decoding the generated SELFIES reproduced the identical canonical isomeric molecular graph (16,075/16,075). Within PepLink's documented reverse-parsing contract—linear and head-to-tail cyclic monomer peptides composed of standard L/D residues—the AA-annotation-to-SELFIES-to-AA-annotation round trip was 4,939/4,939 using the released PepLink package, including 523/523 supported head-to-tail cyclic peptides. The package, tests, version manifest, and record-level audit outputs are publicly versioned.
>
> Second, we reconstructed the provenance of the historical amino-acid mappings and manually adjudicated the ChatGPT-o1/OPSIN-derived definitions against the original DBAASP annotations, formulas, and available public chemical records.  After intersecting the confirmed local residue-definition and residue-template errors with the actual <=512-token genome-or-text DBAASP pool used by our model, the affected scope was 56 of 15,177 peptides (0.369%) and 219 of 74,103 MIC records (0.296%). We provide the record-level decisions and source lineage as Supplementary Data, while retaining the original frozen dataset solely to reproduce the reported models.

## 给正文/Methods 的一句话

已于 2026-07-23 以 `\rev{}` 正式插入 ChatGPT-o1/OPSIN 所在 Methods 段落：

> We subsequently performed a source-aware manual audit of the ChatGPT-o1/OPSIN-derived definitions and the local residue templates used in this conversion. Confirmed definition or template errors affected 56 of 15,177 eligible DBAASP peptides (219 of 74,103 MIC records; 0.296%); these records were flagged for correction or exclusion in the successor dataset, while the original frozen inputs were retained for reproducibility.

上述英文 reviewer 回复也已替换作者 manuscript checkout 中 reviewer-response DOCX 的旧
future-tense 草稿。投稿 DOCX 不进入本源码仓库；英文/中文 Supplementary Data 位于
`supplementary_data/`。

## 上述英文回复的中文对照

> 感谢审稿人指出 amino-acid-to-structure 转换需要验证。我们增加了两层互补验证。第一，对于所有
> 能完成 forward structure construction 的 curated peptide，生成的 SELFIES 解码后均恢复为完全
> 相同的 canonical isomeric molecular graph（16,075/16,075）。在 PepLink 已公开说明的 reverse
> parsing contract 内——由标准 L/D residues 构成的 linear 与 head-to-tail cyclic monomer
> peptides——使用已发布的 PepLink v0.1.2，AA annotation→SELFIES→AA annotation round-trip 为
> 4,939/4,939，其中包括 523/523 个受支持的 head-to-tail cyclic peptides。软件包、测试、版本
> manifest 和逐记录审计输出均已版本化公开。
>
> 第二，我们重建了 historical amino-acid mappings 的来源，并依据原始 DBAASP annotations、
> formulas 和可获得的公开化学记录，对 ChatGPT-o1/OPSIN-derived definitions 进行了人工判定。
> 我们同时将本地 residue-based construction 与按 DBAASP record 所链接 PubChem CID 直接取得的
> whole-peptide structures 分开；后者不是由 ChatGPT-o1、OPSIN 或 PepLink 生成，因此不被归类为
> 本地转换失败。
>
> 将确认的本地 residue-definition/residue-template 错误与 hierarchical MIC loader 实际使用的
> `<=512` token genome-or-text DBAASP pool 相交后，影响范围为 56/15,177 peptides（0.369%）和
> 219/74,103 MIC records（0.296%，四舍五入为 0.30%）。coordination-bond omission 被视为预定义的
> 去金属表示，不计为错误。该范围也不包含由 DBAASP-linked PubChem CID 获得的 whole-peptide structure、仅超出当前
> PepLink API contract 的记录及 non-exact polymer proxies。逐记录判定和 source lineage 作为补充
> 数据提供；原 frozen dataset 仅保留用于复现已报告模型。
>
> 由于这些记录可能在训练阶段暴露，低 prevalence 本身不能证明模型影响为零，我们也不作此声明。
> 只删除测试行的 post hoc 分析无法消除 training exposure；此外，并非所有 historical run 都保留了
> ID-aligned row-level predictions 和精确 legacy strain-wise split membership。因此，我们将实测
> prevalence 与完整血缘作为 limitation 报告，而不把不完整的 evaluation-only sensitivity 描述成
> 与 corrected-data retraining 等价的修正。

## 不应使用的表述

- 不写“355 个 PepLink forward failures 进入训练并有问题”；
- 不写“617/4,095 strict sensitivity”；
- 不写“0.30% 很小，所以对训练没有任何影响”；
- 不写已经重算了 held-out sensitivity，除非未来取得 ID-aligned prediction 并给出真实数值；
- 不把 ambiguous source annotation 计为 confirmed error，也不把它计为 confirmed correct。
