# Reviewer 4 unseen-species / unseen-genus ATCC 靶点初筛

Reviewer 4 认为 strain-selective 的临床论证不成立，并要求改用「对模型从未训练过的 species 或
genus 展示 broad efficacy」作为有意义的 benchmark。本目录记录为满足该要求所做的**可购买靶点
初筛**：哪些临床相关病原体既完全不在 guidance regressor 的训练暴露中，又能在 ATCC 买到，
并且已经有 ATCC Genome Portal assembly 可以喂给外部 Evo-2 producer。

审计入口：`scripts/audit/audit_reviewer4_unseen_atcc_pathogens.py`（只读，不训练、不下载权重）。
本目录与 `DataPrepare/Data/private_inhouse_amp/` 中的 in-house workbook 初筛互补：后者回答
「已有 MIC 数据的未见 species 有哪些」，本目录回答「要新做 broad-efficacy 实验应该买什么」。

## 已由代码和公开数据源验证的事实

- 训练暴露沿用与 `audit_reviewer4_inhouse_species_coverage.py` 相同的冻结过滤：实际进入
  guidance regressor 的是 1,599 个标准化 strain ID，对应 389 个 producer-era species 名，
  加入项目 alias 后为 425 个 canonical 名。
- 425 个训练名全部被 NCBI Taxonomy 解析成功（`training_names_unresolved_by_ncbi` 为空），
  归一到**当前** 165 个 genus。unseen 判定在当前名上进行，不在 producer-era 字符串上进行。
  这一步是必需的而不是补充的：producer-era 名两个方向都会过时，直接做字符串 diff 会同时
  漏判和误判。
- 该判定确实改变了结论。`Ochrobactrum anthropi` 现已并入 `Brucella`，而 `Brucella melitensis`
  和 `Brucella abortus` 在训练集中，因此它被正确降级为 `species_only`，而不是被当成未见 genus。
- 81 个候选中，62 个为 genus-level unseen，19 个为 genus 已见、species 未见。
- ATCC 目录经由站点公开搜索索引读取，字段包括目录号、accepted name、strain designation、
  biosafety level、type-strain 标记、`webz32xavailable` 与 Genome Portal 链接。产品模板名取自
  实时 facet：细菌为 `Bacteria and Bacteriophages`，真菌为 `Mycology`，原生动物为 `Protistology`。
- `atcc_orderable_with_genome` 同时要求「在线可订购」和「已有 Genome Portal assembly」，这是
  能否进入 guided generation 的实际门槛。以该口径，18 个 genus-level unseen species 达到
  `panel_ready`（>= 3 个可订购且有基因组的 isolate）。
- 抽查核对：ATCC 33672（*Providencia stuartii*）与 ATCC 49042（*Rothia mucilaginosa*）的实时产品页
  可访问，均包含 Genome Portal 链接，且链接的 genome ID 与索引返回值一致。

### genus-level unseen 且 panel-ready 的候选

| species | genus | family | 同 family 已训练 species 数 | ATCC 产品 | 可订购且有基因组 | BSL |
| --- | --- | --- | --- | --- | --- | --- |
| *Morganella morganii* | Morganella | Morganellaceae | 4 | 38 | 24 | 2 |
| *Pantoea agglomerans* | Pantoea | Erwiniaceae | 0 | 41 | 15 | 1 |
| *Providencia stuartii* | Providencia | Morganellaceae | 4 | 13 | 11 | 1 |
| *Erysipelothrix rhusiopathiae* | Erysipelothrix | Erysipelotrichaceae | 0 | 51 | 7 | 2 |
| *Hafnia alvei* | Hafnia | Hafniaceae | 2 | 8 | 6 | 1 |
| *Providencia rettgeri* | Providencia | Morganellaceae | 4 | 15 | 6 | 1--2 |
| *Pluralibacter gergoviae* | Pluralibacter | Enterobacteriaceae | 24 | 5 | 5 | 2 |
| *Brevundimonas diminuta* | Brevundimonas | Caulobacteraceae | 0 | 15 | 4 | 1 |
| *Comamonas testosteroni* | Comamonas | Comamonadaceae | 1 | 23 | 4 | 1 |
| *Eikenella corrodens* | Eikenella | Neisseriaceae | 3 | 5 | 4 | 2 |
| *Leptospira interrogans* | Leptospira | Leptospiraceae | 0 | 19 | 4 | 2 |
| *Providencia alcalifaciens* | Providencia | Morganellaceae | 4 | 5 | 4 | 1--2 |
| *Rothia mucilaginosa* | Rothia | Micrococcaceae | 5 | 5 | 4 | 1 |
| *Treponema denticola* | Treponema | Treponemataceae | 0 | 6 | 4 | 1 |
| *Finegoldia magna* | Finegoldia | Peptoniphilaceae | 2 | 5 | 3 | 1 |
| *Kingella kingae* | Kingella | Neisseriaceae | 3 | 3 | 3 | 2 |
| *Leclercia adecarboxylata* | Leclercia | Enterobacteriaceae | 24 | 8 | 3 | 1 |
| *Plesiomonas shigelloides* | Plesiomonas | Enterobacteriaceae | 24 | 4 | 3 | 2 |

*Providencia* 三个 species 全部未见，合计 21 个可订购且有基因组的 isolate，是当前唯一能同时
支撑 genus-level 主张和多 isolate panel 的候选。

## 根据现有证据作出的推断

- 「未见 genus」不等于「未见 family」，这一列必须在回复中主动披露。*Providencia* 与
  *Morganella* 同属 Morganellaceae，而 *Proteus* 三个 species 在训练集中；审稿人很可能据此
  质疑新颖性。family 完全未见的候选只有 *Pantoea*、*Erysipelothrix*、*Brevundimonas*、
  *Leptospira* 和 *Treponema*。
- Morganellaceae 候选存在真实的生物学风险：*Providencia*、*Morganella* 与 *Proteus* 一样，
  通过 lipid A 的 L-Ara4N 修饰对 polymyxin 类天然耐药。这既让「未满足的临床需求」叙事更强，
  也让 AMP 实际起效的概率更低。这一点是从已知耐药机制推断的，不是本仓库数据证明的。
- 目录索引中 `atcc_products` 与 `atcc_orderable_with_genome` 的差值主要来自尚未测序的老 isolate，
  不代表这些菌株不可购买。
- 本次未把「临床合理性」量化。syndrome 标注是策展性的，用于让作者和 microbiology 团队排序，
  不能作为选靶依据本身。

## 仍待作者或 microbiology 团队确认的事项

- 从 genus-level unseen 且 panel-ready 的候选中选定靶点，并决定是走 genus-level 主张
  （*Providencia*，3 species / 21 isolate）还是 family 也完全未见的单 species 主张
  （如 *Pantoea agglomerans* 或 *Erysipelothrix rhusiopathiae*）。
- 逐条核对实时产品页的 BSL、运输限制与机构生物安全审批；索引中的 BSL 字段不能替代该步骤。
  厌氧菌（*Finegoldia*）、fastidious 菌（*Kingella*、*Eikenella*）和螺旋体（*Leptospira*、
  *Treponema*）的 MIC 方案与常规 CLSI 肉汤稀释不同，需要先确认可行性。
- 靶点确定后，Evo-2 genome embedding 与 strain-text embedding 仍由外部 producer 生成，并登记
  exact genome assembly、text source、SHA-256 和 producer commit；在这些资产就绪前不启动 sampler。
- 作者于 2026-08-07 确认已购买 *P. stuartii* ATCC 29914。该菌株的 exact-target asset、历史格式
  text、Evo-2/text embedding、41-length generation 和非破坏性 filter contract 记录在
  `providencia_stuartii_atcc_29914/`；其当前 NCBI/ATCC assembly identity 冲突与执行 blocker 不由
  本候选表的旧状态覆盖。
- 本初筛只证明「买得到且有基因组」。模型在这些靶点上的实际预测性能尚未评估，也没有任何湿实验
  数据；不得在回复信中把候选清单表述为已完成的 broad-efficacy 结果。

## 输出

| 文件 | 内容 |
| --- | --- |
| `unseen_atcc_pathogen_candidates.csv` | 每个候选 species 一行：当前 taxonomy、unseen 级别、family 暴露、ATCC 计数、BSL |
| `unseen_atcc_pathogen_strains.csv` | 每个 ATCC 目录号一行：strain designation、BSL、type-strain、是否可订购、Genome Portal 链接、分离来源 |
| `unseen_atcc_pathogen_summary.json` | 训练暴露统计、数据源、计数汇总与证据边界 |
| `ncbi_taxonomy_cache.json` | 名称到当前 tax ID / genus / family 的解析缓存，重跑时复用 |

重跑：

```bash
python scripts/audit/audit_reviewer4_unseen_atcc_pathogens.py
```

脚本只读取公开的 NCBI Taxonomy 与 ATCC 目录索引，以及本仓库已冻结的训练数据；不修改任何
原始数据、checkpoint 或论文结果。
