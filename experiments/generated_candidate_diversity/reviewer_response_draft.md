# Reviewer response draft: generated-candidate diversity

> We thank the reviewer for raising this point. We have now quantified diversity at
> three complementary levels: sequence diversity within the 24-peptide experimental
> panel, structural diversity within the peptide candidate pool and structural
> diversity across the full guided-generation output set. At the latter two levels,
> pairwise structural similarity was calculated using radius-2, 2,048-bit Morgan
> fingerprints with chirality enabled and the Tanimoto coefficient. To define the
> peptide candidate pool, generated outputs were scored against their respective
> target strain using the MIC regressor. We retained outputs with predicted
> MIC <=15 µM that could be resolved as synthesizable peptides without an
> unidentified residue. This post-generation screen produced 73 peptide
> candidates. The 24 peptides taken forward for synthesis were selected from this
> pool to represent diverse structural and sequence classes while remaining
> practically synthesizable. Predicted MIC was used only for computational
> prioritization and was not treated as an experimental activity measurement.
>
> We additionally quantified sequence diversity within each target-specific
> selection group. The 24 synthesized peptides were stratified by generation
> target and topology, yielding 66, 3, 15 and 3 within-group pairs for the
> *P. aeruginosa* PA5257 linear/cyclic and *E. coli* AR-0349 linear/cyclic groups,
> respectively.
> All 87 pairs had PID below 50%, with median PIDs of 16.7%, 25.0%, 22.6% and
> 18.2%, respectively (Supplementary Fig. C5a).
>
> The 73-candidate pool corresponded to 71 distinct canonical isomeric structures
> (97.3% unique). Its median pairwise Tanimoto similarity was 0.375 (95th
> percentile, 0.695), and only 0.685% of pairwise similarities were >=0.8. We also
> generated 84,226 MIC-guided outputs across the two target strains and all
> generated lengths. These
> outputs comprised 83,433 distinct canonical isomeric structures (99.1% unique),
> with a median pairwise Tanimoto similarity of 0.190 (95th percentile, 0.412);
> only 0.825% of pairwise similarities were >=0.5. We have added the selected-panel
> PID distribution and the two Tanimoto-similarity distributions to Supplementary
> Fig. C5a--c. Together, the high
> structural uniqueness and predominantly low pairwise similarities indicate that
> guided generation did not collapse to a narrow set of repeated structures.

## 中文核对

我们在三个互补层级回答 reviewer：

- selected-panel level：24 条 synthesized peptide candidates，按 target strain 与 topology
  分成四组，穷举组内 87 个无序 pairs；
- candidate level：全部 73 条 peptide candidates，71 个 distinct structures，穷举 2,628 pairs；
- generation level：两个目标菌株、全部生成长度的 84,226 条 guided generation outputs，精确去重并
  穷举全部 3,546,967,425 个无序 non-self pairs。

73 条 candidate pool 的筛选流程为：fixed-`t=1e-3` MIC regressor 对相应 target strain 评分，保留
predicted MIC <=15 µM 且能够解析为不含 unresolved `X` residue 的 peptide structures；
再从中按 structural/sequence diversity 与 synthesis feasibility 选择 24 条进入合成，并降低
过度疏水或有 aggregation-prone motifs 的候选优先级。predicted MIC 只用于 computational
prioritization，不得写成实验活性。

对外文稿统一写成 `we generated 84,226 MIC-guided outputs`。内部数据边界上，这 84,226 条对应
generation pipeline 的 output collection，不等同于所有 raw sampling attempts。Generation pair distribution 已穷举全部 3,546,967,425 个
无序 non-self pairs；不能改写成 raw-attempt distribution。Candidate-level 数值只允许从
`canonical_candidates/candidates_73.csv` 重算。

## 论文对应修改与回复方案（正式 TeX 已执行）

### 同时回答 24 peptides / 18 purchased compounds 的 selection-process comment

当前 peptide prioritization 与 diversity 两段能够回答 reviewer 对 24 条合成 peptide 的
denominator、predicted-MIC cutoff 和 diversity criteria，但不能单独回答 18 条 purchased
compounds。小分子部分的已核验证据和边界见 `molport_selection_audit.md`。同一条 reviewer
reply 并列报告两条筛选链：

> We thank the reviewer for noting that the original manuscript did not provide
> sufficient denominators and selection criteria. For peptide generation, we
> generated 84,226 MIC-guided outputs across the two target strains. Applying a target-strain predicted MIC cutoff of <=15 µM
> together with peptide resolvability yielded 73 peptide
> candidates; 24 were selected to span structural and sequence classes while
> accounting for hydrophobicity, aggregation-prone motifs and practical
> synthesis feasibility. For small-molecule screening, we assembled 44,608
> unique SMILES entries from the three public small-molecule antibiotic datasets
> used in our molecular-classification benchmark and scored them against each
> target strain. The <=15 µM cutoff retained 1,554 entries for *E. coli* AR-0349 and 395 for *P. aeruginosa*
> PA5257. Exact canonical-structure matching identified 179 structures available
> through MolPort. A chemical-liability screen based on PAINS, BRENK and additional
> reactive or assay-interfering structural alerts left 80 unflagged structures.
> Nineteen compounds occupying 19 distinct fingerprint clusters were advanced
> to procurement review; one was not purchased because its quoted cost was
> substantially higher than that of the other candidates, leaving 18 compounds
> for experimental testing. Predicted MIC values for the 18 selected compounds
> ranged from 0.90 to 12.0 µM. Predicted MIC was used only for computational
> prioritization and was not interpreted as experimental activity.
>
> Consistent with this diversity objective, all 87 within-target,
> topology-matched pairs among the 24 selected peptides had sequence PID below
> 50%, with median PIDs of 16.7%, 25.0%, 22.6% and 18.2% for the linear and
> cyclic *P. aeruginosa* PA5257 candidates and the linear and cyclic *E. coli*
> AR-0349 candidates, respectively (Supplementary Fig. C5a).
>
> This diversity-aware selection was intended to test distinct regions of
> peptide and small-molecule chemical space rather than a set of closely related
> top-ranked analogues. Figure 3a provides an *in silico* comparison of
> model-predicted candidate distributions under unconditional and guided
> generation. It is not presented as experimental validation of enrichment;
> the prospective MIC measurements provide the direct experimental evaluation
> of the selected candidates. Revised Supplementary Fig. B2 reports the complete
> MIC matrices for all 24 tested peptides and all 18 tested small molecules
> across the 20-strain panel.

不能加入“因 delivery time 过长排除候选”：目前找到的 19-compound quote 中全部为 in stock、交期
14--21 business days，且最终 18 条覆盖这个完整范围；只有排除一条显著高价候选有直接文件证据。

### Reviewer 4 selection / hit-rate comment

Reviewer 4 的 comment 与上一条 selection-process comment 应共用同一组 Results、Methods 和
Supplementary revisions，不需要重复增加论文段落或图。Reviewer 4 的回复另外明确回答
“为什么不是只选预测最强者”以及 prospective hit rate：

> We thank the reviewer for this comment. We agree that the original phrase
> “a range of predicted MIC values” was ambiguous. We did not deliberately
> select weakly predicted peptides. All 73 members of the peptide candidate pool
> passed the target-strain predicted-MIC cutoff of <=15 µM and could be resolved
> as peptides without unidentified residues. Accordingly, the revised manuscript
> reports this eligibility threshold rather than retaining the ambiguous claim
> that we deliberately selected a “range” of predictions. We selected 24 from this pool using predicted MIC as a
> prioritization threshold rather than as the sole ranking objective. Structural
> and sequence diversity, excessive hydrophobicity, aggregation-prone motifs and
> practical synthesis feasibility were considered so that the tested panel
> represented distinct candidate classes rather than a set of closely related
> top-ranked analogues. Consistent with this objective, all 87 within-target,
> topology-matched pairs among the selected peptides had sequence PID below
> 50%, with median PIDs of 16.7%, 25.0%, 22.6% and 18.2% for the linear and
> cyclic *P. aeruginosa* PA5257 candidates and the linear and cyclic *E. coli*
> AR-0349 candidates, respectively (Supplementary Fig. C5a).
>
> Using experimental MIC <=64 µM against each peptide's intended unseen target
> as the predefined activity criterion, 10 of 24 peptides were active (41.7%).
> This comprised 4 of 15 peptides directed against *P. aeruginosa* PA5257
> (26.7%) and 6 of 9 directed against *E. coli* AR-0349 (66.7%). Across the ten
> hits, experimental MICs were 4 µM (n=1), 8 µM (n=3), 16 µM (n=1), 32 µM
> (n=2) and 64 µM (n=3).
>
> The small-molecule screen used a separate two-stage workflow. We assembled
> 44,608 unique SMILES entries from the three public small-molecule antibiotic
> datasets used in our molecular-classification benchmark and scored them
> separately against each target; this was not the full MolPort catalogue. The
> <=15 µM cutoff yielded 1,535 distinct canonical
> structures across targets. Exact matching against the 5,887,458-entry MolPort
> catalogue identified 179 purchasable structures. Chemical-liability
> screening left 80 structures without PAINS, BRENK or additional reactive or
> assay-interfering alerts. Nineteen compounds occupying 19 distinct fingerprint
> clusters entered procurement review, and 18 were purchased after the quoted
> cost of one compound substantially exceeded that of the others. One of 18
> tested compounds met the experimental MIC <=64 µM criterion (5.6%):
> 2-fluoroadenosine inhibited *E. coli* AR-0349 at 16 µM. Importantly, this was
> an intentionally stringent zero-shot cross-modal screen: the MIC predictor
> used for prioritization had been trained exclusively on peptide MIC data and
> had not been trained on any small-molecule MIC measurements. The 5.6% hit rate
> should therefore be interpreted as the outcome of peptide-to-small-molecule
> transfer rather than that of a small-molecule-specific screening model. The
> recovery of one active compound supports the feasibility of such cross-modal
> transfer, while the modest hit rate also indicates clear scope for future
> small-molecule-specific fine-tuning.
>
> The diversity-aware selection was designed to evaluate distinct regions of
> chemical space rather than closely related top-ranked analogues. Figure 3a is
> now explicitly described as an *in silico* comparison of model-predicted
> candidate distributions, whereas prospective MIC testing is described as the
> direct experimental evaluation of the selected candidates. Revised
> Supplementary Fig. B2 reports the complete MIC matrices for all 24 tested
> peptides and all 18 tested small molecules across the 20-strain panel.

正式 TeX 已删除原有的 `spanning a range of predicted MIC values`，改成全部先满足 `<=15 µM`
threshold，再以 diversity/feasibility 选择 24 条。Reviewer 4 与前一条 reviewer response 可以有
各自的完整答复，但数字和 manuscript location 必须完全相同。

### Results：两处修改（已执行）

第一处，已替换当前 24 条 peptide selection 的一句话：

> We generated 84,226 MIC-guided outputs across the two target strains and all
> generated lengths. Applying a target-strain predicted MIC cutoff of <=15 µM
> together with peptide resolvability yielded a pool of 73 peptide candidates
> corresponding to 71 distinct canonical
> isomeric structures. We selected 24 peptides from this pool for synthesis,
> prioritizing structural and sequence diversity and practical synthesis
> feasibility while deprioritizing excessive hydrophobicity and
> aggregation-prone motifs. In parallel, we scored 44,608 small-molecule SMILES
> entries against each target and identified 179 low-predicted-MIC structures
> with exact canonical-structure matches in the MolPort catalogue. Following
> structural-alert, fingerprint-cluster-diversity and procurement-feasibility
> review, 18 compounds were purchased for experimental testing.

第二处，已在三条 representative leads 的 training-set sequence similarity 结果之后加入
generated-set diversity：

> The selected experimental panel also remained sequence diverse within each
> generation target and peptide topology: all 87 within-group pairs had PID
> below 50%, with median PIDs of 16.7%, 25.0%, 22.6% and 18.2% for the
> *P. aeruginosa* PA5257 linear/cyclic and *E. coli* AR-0349 linear/cyclic groups,
> respectively
> (Supplementary Fig. C5a). At the broader candidate-pool and generation levels, the 73-peptide candidate pool comprised 71
> distinct canonical isomeric structures (97.3% unique), whereas the full set of
> 84,226 MIC-guided outputs comprised 83,433 distinct structures (99.1% unique). Their
> median pairwise Morgan-fingerprint Tanimoto similarities were 0.375 and 0.190,
> respectively (Supplementary Fig. C5b,c), indicating that predictor-guided
> generation did not collapse into a narrow set of highly similar structures.

保留前面 hierarchical MIC Results 中更直观的 `we additionally evaluated exact-peptide
overlap` 及其 train-seen/train-unseen 解释，不改 line 154。将当前容易混淆的段末回指：

> Together with the exact-peptide-disjoint strain-wise sensitivity analysis
> described above, ...

改为直接描述分析对象：

> Together with the strain-wise sensitivity analysis on peptides absent from the
> corresponding training folds, these
> prospective results provide complementary experimental evidence that
> ApexOracle can generate active, structurally novel peptides in the challenging
> setting where both the peptide and target strain are unseen.

该分析仍解释为 MIC predictor sensitivity，而不是 generation-diversity test；Methods 和
Appendix Table 保持现有 train-seen/train-unseen 定义与结果。

### Methods：最终 subsection 组织（已执行）

`ApexOracle Architecture and Training` 保留 guided-generation implementation 与
remasking-window sensitivity。随后新增两个独立 subsection，避免在定义 PID 之前使用该指标。

#### Candidate prioritization and virtual screening

> **Post-generation peptide prioritization.** Outputs with predicted MIC >15 µM
> were excluded. Retained structures were required to be RDKit-valid and
> resolvable as peptides without an unidentified residue. The resulting
> candidate pool was reviewed for duplicate structures, excessive hydrophobicity,
> aggregation-prone motifs and practical synthesis feasibility. Twenty-four
> peptides were selected to span distinct structural and sequence classes.
> Predicted MIC was used only for computational prioritization and was not
> interpreted as experimental activity.

> **Small-molecule virtual screening and procurement.** We assembled the
> screening collection from the three public small-molecule antibiotic datasets
> used in the molecular-classification benchmark. The merged table contained
> 49,331 molecule--strain rows. After SMILES-to-SELFIES conversion and
> consolidation of repeated SELFIES representations, the screening collection
> contained 44,608 entries representing 39,995 distinct RDKit-valid canonical
> isomeric structures. This collection was scored separately against each target
> strain. Applying a predicted-MIC cutoff of <=15 µM retained 1,554 entries for
> *E. coli* AR-0349 and 395 for *P. aeruginosa* PA5257; after canonicalization,
> the union comprised 1,535 distinct structures. Exact matching against the
> 5,887,458-entry MolPort catalogue identified 179 commercially available
> structures. These were grouped by radius-2, 2,048-bit Morgan-fingerprint
> Butina clustering at a Tanimoto threshold of 0.75 and screened for PAINS,
> BRENK and additional reactive or assay-interfering structural alerts. Eighty
> structures had no detected alert. Nineteen compounds occupying 19 distinct
> clusters were advanced to procurement review; one was not purchased because
> its quoted cost substantially exceeded that of the others, leaving 18 compounds
> for experimental testing. Predicted MIC was used only for computational
> prioritization and was not interpreted as experimental activity.

#### Sequence and structural diversity analyses

该 subsection 先统一定义 Biopython `PairwiseAligner`、BLOSUM62、gap penalties 和 PID 公式，
再依次描述三条 representative leads 对 training set 的 novelty、24 条 selected peptides 的
sequence diversity，以及 73/84,226 两个层级的 structural diversity。Linear/cyclic topology
规则和 cyclic rotations 只定义一次。

> **Selected-peptide sequence diversity.** The 24 synthesized peptides were
> grouped by generation target and topology: 12 linear and three cyclic
> candidates for *P. aeruginosa* PA5257 and six linear and three cyclic candidates
> for *E. coli* AR-0349. PID was calculated for every unordered pair within each
> group, yielding 66, 3, 15 and 3 pairs, respectively. To ensure that PID did not
> depend on sequence input order, each pair was aligned in both orientations and
> the higher PID was retained.

> **Structural diversity.** At the candidate level, we analyzed all 73 peptide
> candidates, including duplicate rows. At the generation level, we analyzed the
> 84,226 MIC-guided outputs generated across both target strains and all generated
> lengths. Distinct structures were defined by exact equality of canonical
> isomeric SMILES. Pairwise structural similarity was calculated for every
> unordered non-self pair using radius-2, 2,048-bit Morgan fingerprints with
> chirality enabled and the Tanimoto coefficient.

### Results：prospective hit-rate 总结（已执行）

已在 24 peptides 与 18 small molecules 的实验结果首次汇总处增加：

> Using an experimental MIC cutoff of <=64 µM against the intended unseen
> target, 10 of the 24 peptides were active (41.7%): one at 4 µM, three at
> 8 µM, one at 16 µM, two at 32 µM and three at 64 µM. The target-specific hit
> rates were 4/15 (26.7%) for *P. aeruginosa* PA5257 and 6/9 (66.7%) for
> *E. coli* AR-0349. One of the 18 screened small molecules was active (5.6%):
> 2-fluoroadenosine inhibited *E. coli* AR-0349 at 16 µM.

这里的 hit 定义必须始终是 candidate 对其 intended unseen target 的 MIC `<=64 µM`，不能改成对
20-strain panel 中任意菌株有活性，否则分子和分母都会改变。

### Supplementary Information

当前单行三 panel 图已排在现有 Supplementary Fig. C4 后并核验为 Fig. C5；新 panel a 为
within-target/topology selected-peptide PID，原 Tanimoto panels 顺延为 b/c。Results 和 Methods
均已引用/描述；panel a 的 PID 纵轴固定显示完整 0--100% 范围，并使用正文中的正式 strain
编号与明确的 Lin./Cyc. 拓扑缩写；这些绘图设置不在 caption 中重复，方法学 comparison scope
保留在 Methods。Caption 使用 `figure_caption.md`。73 条 candidate list 是否作为独立 Supplementary Data 提交仍是
可选 publishing step；canonical 数据已经冻结在 `canonical_candidates/candidates_73.csv`，但本轮
未将内部 provenance columns 原样复制进投稿目录。Abstract 与 Discussion 未新增筛选细节。
