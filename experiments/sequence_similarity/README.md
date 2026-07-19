# ApexOracle lead peptide sequence similarity

本目录记录 ApexOracle-3、ApexOracle-12 和 ApexOracle-23 sequence-similarity 流程的
行为保持迁移。canonical 配置为 `configs/sequence_similarity/paper_leads.yaml`，唯一入口为：

```bash
python scripts/reproduce/run_sequence_similarity.py all
```

该命令依次重建训练 peptide cache、计算 linear 和 exhaustive cyclic alignments、提取 top
hits 并验证输出不变量。大型 CSV 写入被 Git 忽略的
`results/sequence_similarity/paper_leads/`；Git 只保存本目录中的小型审计材料。

也可以分阶段运行 `prepare`、`compute`、`summarize` 和 `validate`。例如只重算指定 lead：

```bash
python scripts/reproduce/run_sequence_similarity.py compute \
  --query-id ApexOracle-3 --query-id ApexOracle-23
```

## 已由代码和全量输出验证的事实

- 论文模式使用 Biopython `PairwiseAligner(mode="global")`。BLOSUM62 alignment 的 gap-open
  和 gap-extension 分别为 `-10` 与 `-0.5`；第二种 alignment 使用 exact-match score 1，
  mismatch 和 gap score 均为 0。
- PID 为最终 gapped alignment 中区分大小写的 exact residue matches 除以包含 gap 的
  alignment length。`max_len_identity` 的分母为两条原始序列长度的较大值。
- cyclic peptide 穷举 query 和 training peptide 的全部 rotation；tie 依次比较 primary
  metric、secondary metric、matches、较小 query rotation、较小 training rotation，完全
  相同时保留训练 cache 中先出现的行。
- 使用 `training_sequence_case: uppercase` 可以从当前 120,955 行 MIC 表和 DBAASP JSON
  逐字节重建历史的 13,077 条 linear cache 和 1,039 条 cyclic cache。两份 SHA-256 分别为
  `3d30fa6509e489312e31110dcc27da13b8ea47622b5e2dd1897a6f6f24adafb9` 和
  `c415b1278cb947c8b12121d4c3f414d2dc8b19cef6071af08e52376cf53bd9e5`。
- canonical 模块对 ApexOracle-3/23 完整重算的 26,154 条 linear、383,800 条 cyclic
  rotation 以及两份各 2,078 条 cyclic best CSV，与历史文件逐字节相同。
- 完整三条 lead 重算得到论文中的最大 BLOSUM62 PID：ApexOracle-3 `11/30 = 0.3667`、
  ApexOracle-12 `10/28 = 0.3571`、ApexOracle-23 `7/19 = 0.3684`。
- 四份完整输出通过 formula、alignment length、match upper bound 和 selection-metric
  检查，合计没有发现不变量错误。详见 `equivalence_audit.json`。

## 训练序列大小写的版本边界

历史 March 2026 cache 把 training peptide 全部转为大写；这是重建论文结果所必需的输入
规范。legacy snapshot 中后来保存的 `extract_training_peptides.py` 则直接保留当前 DBAASP
JSON 的大小写。按该后来行为重建时，13,077 条 linear 中有 1,265 条、1,039 条 cyclic 中
有 636 条包含小写 residue；ApexOracle-23 的最大 PID 会由 `0.3684` 变为 `0.3158`。

因此 canonical `paper_leads.yaml` 明确冻结 `uppercase`。保留大小写只能作为 chirality
sensitivity analysis，不能覆盖论文正式结果；这不是根据模型表现作出的选择，而是由历史
cache SHA 和保存输出共同确定的数据版本契约。

## ApexOracle-12 的完整 tie

历史主输出只保存了 ApexOracle-3 和 ApexOracle-23；ApexOracle-12 是使用同一 canonical
算法和逐字节恢复的 paper cache 重新计算的。其最大 PID `0.3571428571` 有四个完全并列的
training ID：9800、9801、9802 和 15510。当前稳定规则选择先出现的 9800；论文表展示 15510
（序列 `FWFTLIKTQAKQPARYRRFC`）。二者的 matches、alignment length、PID 和
`max_len_identity` 完全相同，因此论文的 35.7% 数值不受影响，但代表序列不是唯一解。

## 仍待作者确认的事项

- 没有找到 ApexOracle-12 当时单独运行生成的历史 full CSV，因此对它的结论是算法与 paper
  cache 的确定性重算，不是旧文件逐字节恢复。
- 论文表为何在四个完全 tie 的结果中展示 DBAASP 15510 没有保留机器可读选择记录；最可能
  是人工选择或当时输入顺序不同，但现有证据不足以确认。
- `compare_linear_query_to_apex11.py` 是额外的 in-house APEX 1.1 collection 对照，不是论文
  training-set similarity 表，已随其他被替代 legacy 入口删除并由 Git tag 保留。
