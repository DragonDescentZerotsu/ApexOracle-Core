# AA 定义与 peptide structure 血缘（中文规范记录）

## 文档目的

本文件记录 reviewer 所问的两条不同血缘：一条是“每个非标准 AA definition 从哪里来”，另一条是
“每个完整 peptide 的 frozen SMILES 从哪里来”。只有第二条血缘决定某个 AA/PepLink 问题是否真正
影响模型消费的结构。不能仅凭 PepLink 当前能否重建某条 annotation，就把 frozen training SMILES
判为错误。

## 已由代码、文件和行数验证的事实

### 1. AA definition 血缘

最终 mapping `DataPrepare/Data/all_aa_smiles_new_handcrafted.csv` 共 459 条：

| 分支 | 数量 | 实际处理 |
| --- | ---: | --- |
| 标准 L/D amino-acid code | 39 | 来自 `L-_and_D-Amino_Acid_SMILES.csv` |
| 初次 PubChem name lookup 成功 | 251 | 形成 `unusual_aa_w_PubChem_smiles.csv` |
| 上述 251 中复查后继续保留 | 207 | 重跑后形成 `unusual_aa_w_PubChem_smiles_new.csv` |
| 上述 251 中转入第二条修正支路 | 44 | DBAASP description 经 GPT refinement，再由 OPSIN 解析；见 `wrong_ones_desc_refined.txt` 与 `wrong_ones_OPSIN_output.txt` |
| 初次 PubChem lookup 无结果的主支路 | 169 | DBAASP annotation → ChatGPT-o1 standardized name → OPSIN SMILES；见 `unusual_aa_names_wo_PubChem_smiles.json`、`unusual_aa_text_transfered_by_GPT.txt`、`unusual_aa_smiles_OPSIN_output.txt` |

因此，“另外 251 条都没有经过 ChatGPT/OPSIN”是错误表述。准确关系是：420 条 noncanonical
definition = 207 条保留的 PubChem lookup + 44 条二次 GPT/OPSIN correction + 169 条主 GPT-o1/OPSIN
branch。

最终 historical peptide builder `DataPrepare/try.py` 明确读取
`all_aa_smiles_new_handcrafted.csv`。三条 handcrafted replacement（`S-ALA-4-pen`、
`R-ALA-7-oct`、`Me-PENT-GLY`）因此已经进入 historical producer；它们不是“发现错误后仍继续使用
旧 OPSIN structure”的记录。

### 2. 完整 peptide structure 血缘

| frozen MIC structure source | peptide 数 | MIC row 数 | 结构如何得到 |
| --- | ---: | ---: | --- |
| whole-peptide PubChem | 840 | 8,434 | 从 DBAASP peptide record 读取 `pubChemCid.cid`，按 CID 直接请求 PubChem IsomericSMILES |
| local residue-based builder | 15,521 | 96,747 | 使用最终 AA mapping、sequence、unusual residues、terminal modifications 和 intra/interchain bonds 本地拼接 |
| DBAASP-offered structure branch | 69 | 366 | 使用 DBAASP 提供结构并经过 historical record-level correction script |
| frozen union | 16,430 | 105,547 | 三条来源合并后的论文 MIC 表 |

上述来源计数是集合暴露统计；historical merge 代码按三个 CSV 拼接，移除 4 条 RDKit 无法解析记录后
得到 21,120 条 peptide structure，随后 MIC 表实际覆盖 16,430 个 peptide。whole-peptide PubChem
支路是按 DBAASP 已给定 CID 查询，不是使用短名猜测 PubChem 命中；其结构也没有经过 ChatGPT-o1、
OPSIN 或 PepLink 构建。

### 3. 真正进入模型数据池的本地结构问题

在 frozen MIC 表之后应用 strain/genome eligibility，长度过滤前的 genome-or-text DBAASP pool 为
15,664 个 peptide、76,604 条 MIC row；再应用论文 loader 的 `<=512` token 过滤后，实际 pool 为
15,177 个 peptide、74,103 条 row。作者确认 coordination bond omission 是既定的去金属/忽略配位
预处理，不再把它计为错误。DBAASP sequence 与 unusual-residue annotation 的内部不一致属于上游
source-data quality 和历史 producer 的容错问题，不是 ChatGPT-o1/OPSIN 或 PepLink 转换错误，也从
reviewer-facing 范围中排除。重新逐条去重后的 reviewer-facing 结果为：

- **确认的本地转换错误：**56/15,177 peptide（0.369%），219/74,103 row（0.296%，报告为 0.30%）。

若与旧报告保持长度过滤前的相同分母，则为 56/15,664（0.358%）和 219/76,604（0.286%）。正文和
reviewer response 优先使用实际 `<=512` loader pool 的 0.30% 口径。

三类确认问题之间没有 peptide 重叠：

| 类别 | peptide 数 | model row 数 | 说明 |
| --- | ---: | ---: | --- |
| 主 169 branch 中仍进入模型的错误 AA definition | 30 | 116 | 11 个 definition |
| 非完整或不可聚合 residue template | 22 | 86 | `Aic`、`Agb`、`Nae`、`MIM`、`Cl-Th2CA` |
| 第二条 44-residue branch 的确认错误 | 4 | 17 | `D-End` 与 `D-IGln` |

11 个主支路确认错误为：`N-TYR`、`LYS-C18`、`3-Me-Trp`、`2-OH-Me-SER`、`NNar`、
`D-3-OH-ASN`、`IAA-Cys`、`6F-LEU`、`HCha`、`D-Me-Trp`、`BisHomo-Pra`。definition-level
计数和理由见 `revised_training_impact_scope.csv`。

### 3.1 面向 reviewer 的 56-peptide Supplementary Data

2026-07-23 已生成
`supplementary_data/Supplementary_Data_AA_conversion_errors.xlsx`（英文期刊版）及
`supplementary_data/补充数据_AA转换错误_中文.xlsx`（逐字段中文镜像）。56 个 peptide
各自恰好对应上述 18 个错误 definition 中的一个，因此 affected-peptide sheet 为一肽一行，不存在
因多个错误 definition 而重复计数。每行包含：

- DBAASP peptide ID、名称、序列和 record URL；
- 错误 residue code、位置、occurrence 数和进入 `<=512` loader 范围的 MIC row 数；
- historical erroneous SMILES 及由 RDKit 重算的 formula；
- corrected residue/free-compound 或 terminal-cap 名称、SMILES 和 formula；
- 具体错误、后续修正或排除动作、stereochemistry/attachment 边界和证据 URL。

16 个具有直接 PubChem compound record 的 corrected definition 已使用 PubChem PUG REST 重新核验，
名称、分子式和结构均与表中版本一致。`NNar` 和 `D-3-OH-ASN` 没有 exact public compound record：
前者依据 DBAASP 的 C5 formula 和 N-(2-guanidinoethyl)glycine connectivity，后者依据
D-3-hydroxyasparagine 名称及 C4H8N2O4 formula；两者在表中均为 `Moderate`，来源未确定的
stereocenter 不作猜测。`Cl-Th2CA` 的正确身份是 N-terminal acyl cap，而不是可聚合的 alpha-AA；
`Nae` 的自由化合物定义可确定，但 incorporation 需要 peptoid/PNA backbone attachment semantics，
在实现前应排除。

这些 corrected structure 是 residue/free-compound/cap 层级的 reference，不表示 56 个完整 peptide
已经重新生成，也不改变 frozen paper CSV。生成脚本
`scripts/audit/build_peplink_supplementary_data.py` 会重新调用 canonical loader，并强制断言
56/15,177 peptide、219/74,103 row 和 18 definition；输入/输出 SHA-256 位于
`supplementary_data/supplementary_data_manifest.json`。
16 条原始 PubChem PUG REST 响应同时冻结于
`supplementary_data/pubchem_corrected_definitions_20260723.json`；生成脚本将其作为输入，逐项
断言分子式与结构。

### 3.2 内部记录：排除的 DBAASP source-data inconsistency

DBAASP sequence 用 `X` 表示 unusual residue，并在 `unusualAminoAcids` 中另给每个 unusual residue
的名称和位置；两部分原则上应一一对应。旧 builder 在 `DataPrepare/aa_seq_to_smiles.py` 中先按声明
位置排序 annotation，再从左到右扫描 `X`，但遇到不一致时不会停止：

- **数量不一致 9 peptide / 36 model row（上游 completeness mismatch）：**`X` 多于 annotation 时，
  `IndexError` 分支直接把多出的 `X` 建成 Leucine；annotation 多于 `X` 时，多余 annotation 从未被
  消费。例如 DBAASP 10433 有 3 个 `X` 但只有 2 条 annotation，第三个 `X` 被当作 Leucine；6672
  有 10 个 `X` 但有 12 条 annotation，最后两条修饰被忽略；14902 有 4 个 `X`、0 条 annotation，
  四处都被当作 Leucine。
- **位置不一致 9 peptide / 45 model row（位置待确认）：**`X` 数量与 annotation 数量相同，但
  声明位置不同。旧 builder 只打印 warning，仍把“下一条 annotation”放到当前扫描到的 `X`。
  例如 DBAASP 4284 的 `X` 在 position 19，而 `MET-OXD` annotation 声明 position 1；旧 builder
  把 `MET-OXD` 放到 position 19。仅凭冲突字段无法判断 position 1 还是 position 19 才是文献中的
  真实修饰位置；这属于 DBAASP 源字段需要后续清理的事项。

这 18 peptide/81 model row 记录的是 DBAASP 上游字段不一致和历史 producer 的容错行为。作者于
2026-07-22 确认：它们不属于 ChatGPT-o1/OPSIN 或 PepLink 转换错误，不计入 56/219，不写入 reviewer
response 或论文修改。该内部记录仅用于解释旧 producer 如何处理源数据。

### 4. Round-trip 已验证范围

- 16,075/16,075 个可 forward construct 的 peptide 通过 SMILES → SELFIES → SMILES molecular-graph
  exact round-trip；
- PepLink 0.1.2 的公开 annotation reverse contract 为 4,939/4,939；
- 其中支持范围内的 head-to-tail cyclic peptide 为 523/523；
- 这证明序列化和受支持 reverse contract，不等同于证明所有 source annotation 的化学身份唯一。

## 根据现有证据作出的判断

- 0.30% 可以用于说明确认的本地化学转换错误在实际 `<=512` token genome-or-text DBAASP pool 中 prevalence
  很低；不能把它称为所有派生训练路径的统一比例。
- 0.30% 不能单独证明模型指标“不受影响”，因为其中部分记录可能出现在训练 fold。
- 仅删除 held-out prediction 行的 evaluation sensitivity 不是 retraining sensitivity，不能消除 training
  exposure。
- PepLink forward failure 不是 chemical error 的同义词；尤其 whole-peptide PubChem source 即使无法由
  当前 PepLink annotation API 重建，模型仍消费按 DBAASP-linked CID 获得的 frozen structure。

## 仍待作者或 chemistry coauthor 确认的事项

- 20 个主支路 source-ambiguous definition（85 个 model-eligible peptide、366 行）需要回查 synthesis
  paper/supplier record，当前既不算 confirmed error，也不算 confirmed correct。
- 第二条 44-residue correction branch 的 `Me-Ser` D/L conflict、`ACT-D-Orn` 与 `iPr-D-Orn`
  substitution-site ambiguity 共涉及 5 个 peptide、19 行，需要原始来源确认。
- 56 个 reviewer-facing 问题 peptide 的 corrected/excluded successor dataset 尚未生成；原 frozen dataset 不原地覆盖。
- 若要证明模型性能对这些记录稳健，需要使用确定的 corrected/excluded dataset 重新训练；目前没有足够
  证据把只删测试行的结果写成等价补救。

## 作者已确认的范围决定

- coordination records 不作为错误；历史 builder 忽略配位信息被记录为明确的去金属/忽略配位预处理。
  其中 6 个含 `MIM` 的 peptide 仍因 `MIM` 本身不是可聚合 residue template 而进入问题范围，而不是
  因为 coordination omission。
- DBAASP sequence/unusual-residue annotation 内部不一致不属于 reviewer 所问的本地化学转换错误；
  18 peptide/81 row 全部从 reviewer-facing 错误报告与论文修改中排除。
- non-exact polymer proxy 不进入 56-peptide/219-row reviewer scope；只在内部完整化学审计中
  保留其 representation limitation 记录。
- 355 个 PepLink forward failure 和旧 617-peptide/4,095-row union 不进入 reviewer response。
- reviewer 回复可报告 0.30% confirmed prevalence；不得
  扩展为“不会影响训练”或“no effect”。

## Canonical 产物

- `reviewer_response_scope_summary.json`：机器可读的修订范围；
- `recalculated_local_error_scope.json` 与 `recalculated_local_error_peptides.csv`：可复现的新计数与
  56 个 reviewer-facing peptide 逐条清单；
- `revised_training_impact_scope.csv`：类别与 definition-level 计数；
- `reviewer_response_draft.md`：可直接使用的 reviewer 措辞；
- `supplementary_data/Supplementary_Data_AA_conversion_errors.xlsx`：56 个受影响 peptide 的英文
  Supplementary Data；
- `supplementary_data/补充数据_AA转换错误_中文.xlsx`：上述表的中文镜像；
- `supplementary_data/supplementary_data_manifest.json`：补充数据 scope 与输入/输出 SHA-256；
- `supplementary_data/pubchem_corrected_definitions_20260723.json`：16 个 direct-match definition
  的 PubChem PUG REST 原始结构快照；
- `chatgpt_opsin_manual_review.csv`：169 条主支路逐项人工判定；
- `peplink_0.1.2/roundtrip_records.csv` 与 `roundtrip_summary.json`：正式 round-trip 记录。
