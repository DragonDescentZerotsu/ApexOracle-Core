# *Providencia stuartii* ATCC 29914 inventory screening and generation

本目录记录 *P. stuartii* ATCC 29914 的 exact-target 条件、生成分母和候选筛选血缘。`Providencia` 未进入
guidance regressor 的训练 exposure；但它与训练中出现的 *Proteus* 同属 Morganellaceae，因此只能
表述为 unseen genus/species，不能表述为 unseen family。

## 已由公开主数据源与本地代码验证的事实

- ATCC 29914 的 strain designation 为 CDC 2896-68，并与 NCTC 11800、DSM 4539、LMG 3260 等
  type-strain deposits 对应。ATCC Genome Portal 的 primary assembly 为 1 条 contig、4,438,675 bp。
- NCBI Datasets v2 当前把 `GCF_029075745.1` 标为 *P. stuartii* strain `1194-23`，79 contigs、
  4,405,218 bp。它与 ATCC portal assembly 的 strain metadata 和 assembly dimensions 均不一致，
  因此 canonical exact-target 入口会拒绝把它当作 ATCC 29914。
- `GCA_900455155.1` 标为 NCTC 11800，属于等价 type-strain deposit；它只可作为 assembly-source
  sensitivity，不能未经说明替代实际购入 ATCC culture 的 primary condition。
- 历史文本不是直接送入 sampler。四段 Qwen 格式原文先把完整 strain 名替换为 `This strain`，再由
  `YBXL/Med-LLaMA3-8B` 的倒数第二层生成 token-level tensor。这里复用相同格式和 embedding contract，
  但不继续调用已退役的 `qwen-max-0125`。

## 文本条件

`strain_description.txt` 是人工策展、可审计的历史 Qwen 四段格式文本；未找到 ATCC 29914-specific
mutation 或 MIC 时明确写 `None`/不外推具体数值。`med_llama3_input.txt` 是实际 encoder input，
`asset_manifest.json` 冻结两者 SHA-256。文本内容可能强烈影响 condition prediction，因此后续不得
静默改写；如需另一版本，应新建 sensitivity condition。

```bash
python scripts/reproduce/prepare_reviewer4_providencia_assets.py --install
apexoracle-embed-strain-texts \
  --input-dir DataPrepare/Data/Text_Description/ATCC/Text \
  --output-dir DataPrepare/Data/Text_Description/ATCC/embeddings \
  --filename-encoding atcc --device cpu --local-files-only --existing skip
```

## Genome 资产与 Evo-2 embedding

ATCC Portal FASTA 与 GenBank 通过 SHA-256 核验。安装和复核入口为：

```bash
python scripts/reproduce/prepare_reviewer4_providencia_assets.py \
  --atcc-fasta DataPrepare/Data/Genome/ATCC/Providencia_stuartii_ATCC_29914.fasta \
  --atcc-genbank DataPrepare/Data/Genome_annotation/ATCC/Providencia_stuartii_ATCC_29914.gbk \
  --install

CUDA_VISIBLE_DEVICES=0,1 conda run -n evo2 python \
  -m apexoracle_evo2.cli \
  --input DataPrepare/Data/Genome/ATCC/Providencia_stuartii_ATCC_29914.fasta \
  --output-dir DataPrepare/Data/Genome_embs \
  --model-name evo2_40b --batch-size 3 --input-device cuda:0
```

入口验证了 FASTA 与 GenBank 都是 single record、4,438,675 bp，且两者 sequence 精确相等；GenBank
含 8,268 个 features。正式 Evo-2 运行使用专属 `evo2` conda environment，不是 base 环境；
环境为 Python 3.11.11、`evo2 0.6.0+apexoracle.1`、`vtx 1.1.0` 和对应 cp311 FlashAttention 扩展。
Evo-2 40B 使用 `blocks.46.mlp.l3`、11-kb windows、10-kb step，并对真实序列位置 mean-pool。
输出 key 由历史 loader 解析为 `29914`；compact 运行与 tensor 验证血缘见
`genome_embedding_manifest.json`，逐 window provenance 位于 canonical embedding 相邻 manifest。

## Peptide inventory screening

使用 ApexOracle-MDLM 的 `scripts/reproduce/peptide_inventory_screen.py` 准备 peptide 表格，
以 `scripts/reproduce/score_peptide_table_mic.py` 计算 MIC，再用 inventory reporting 入口汇总。
通过 CLI 显式指定输入文件、列名、strain key 和输出目录；输入顺序与重复记录保持不变。

该 scorer 按氨基酸 sequence 编码。声明了末端修饰、环化或其他 chemistry 的记录需要单列为
chemistry approximation，不能与 exact-unmodified 结果混合。Predicted MIC 用于计算排序，
不等同于湿实验活性。

## Generation 与所有 filter 分母保存

默认 protocol 复用论文条件：length 256--416（步长 4）、seed 1、每个长度 20×50=1,000 attempts、
256 steps、target MIC 1、`t_on/t_off=0.55/0.45`、`eta=0.02`、`alpha_on=0.5`、
`gamma_MIC/gamma_peptide=15/15`。默认共 41,000 raw attempts；如果增加 seed，必须通过同一 manifest
入口冻结，不能在运行目录手改。

```bash
python scripts/reproduce/prepare_reviewer4_providencia_generation_tasks.py
```

每个 task 使用 `scripts/reproduce/run_remasking_schedule_reviewer.py`。该 runner 在任何 molecule、
MIC 或 peptide filter 之前保存每次 token attempt 的 JSONL，包括 token IDs、completion、SELFIES、
SMILES、canonical SMILES、RDKit validity 和历史 weak-amide flag。GPU 运行待 exact genome/text
embeddings 就绪后编排。

全部任务完成后，先用 `scripts/reproduce/evaluate_remasking_schedule_reviewer.py` 添加 frozen clean
MIC prediction（历史 v1 classifier 只作为附加诊断），再运行：

```bash
python scripts/reproduce/filter_reviewer4_providencia_candidates.py \
  --evaluated-attempts experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/evaluated_attempts.csv \
  --output-dir experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/filters
```

`all_attempts_with_filter_flags.csv` 保留全分母与各层独立 flags：attempted、complete、RDKit-valid、
legacy amide、finite clean MIC、MIC<=15、PepLink standard peptide 和 strict candidate。正式 candidate
选择仍为 inclusive `predicted MIC <= 15 µM` 且 PepLink 0.1.2 能可靠 reverse-parse 为 standard linear
或 head-to-tail cyclic peptide；严格输出为 `strict_candidates_mic_le_15_peplink.csv`。历史 73-candidate
流程实际使用 legacy parser，而不是 PepLink 本身；本次新实验采用更明确、更严格的 PepLink contract。
同一入口还把每一级完整物化到 `tiers/00_attempted.csv` 至 `tiers/07_strict_candidate.csv`，所有输出
均登记 size 和 SHA-256，因此后续分析不需要重跑或覆盖上一级分母。

## 当前状态与边界

- 文本原文、normalized encoder input、assembly identity audit 和 41-task manifest 已冻结。
- Med-LLaMA3 text tensor 已在 CPU 完成，shape 为 `[174, 4096]`、SHA-256 为
  `4e33a0869f66ca568d7edf791ccc11df4471903a42591fe55aa3eabc80795825`；完整路径和 input hash 见
  `text_embedding_manifest.json`。
- exact ATCC FASTA/GenBank 已安装并互相验证；Evo-2 tensor 已完成，shape `[444,8192]`、bfloat16、
  444/444 rows finite/nonzero，SHA-256 `56ee84b1...35cde`。
- 正式 genome embedding 只由 `ApexOracle-Evo2` canonical CLI 生成；早期 target-specific duplicate
  wrapper 已删除，避免 float32/bfloat16 与 manifest 双入口分叉。
- 本目录记录生成协议与输入资产；单个 strain 的结果不能支持 broad species/genus efficacy。

## 主数据源

- ATCC product: <https://www.atcc.org/products/29914>
- ATCC Genome Portal: <https://genomes.atcc.org/genomes/bac4c442bbed4a2f>
- NCBI Datasets assembly record: <https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_029075745.1/>
