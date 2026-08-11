# *Providencia stuartii* ATCC 29914 inventory screening and generation

本目录冻结 Reviewer 4 新购菌株的 exact-target 条件、生成分母和候选筛选血缘。`Providencia` 未进入
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

作者提供的 exact ATCC Portal FASTA 和 GenBank 已从 Mac 复制到本项目 ignored canonical data path；
源文件保留在 Mac，项目侧文件已逐字节 SHA-256 核对。安装和复核入口为：

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

## 私有 lab peptide inventory screening

私有 workbook 位于
`DataPrepare/Data/private_inhouse_amp/de_la_Fuente_Lab_Peptides_Inventory_outsourced.xlsx`，SHA-256 为
`21b37ac03b892b88a6d2da6f5f42e8d68cc1393cab0bc86c4e77f511b4970f24`。筛选保留全部 4,842 个源行、
原始顺序和重复项，不按 sequence 去重。Inventory preparation 与 strain 无关，因此只在 canonical private
source 下准备一次；任何后续 strain 直接复用同一 `screen_input.csv`，不再复制 Reviewer-specific adapter：

```bash
MDLM_ROOT=/path/to/ApexOracle-MDLM
PYTHONPATH="$MDLM_ROOT/src" conda run -n mdlm python \
  "$MDLM_ROOT/scripts/reproduce/peptide_inventory_screen.py" prepare \
  --input DataPrepare/Data/private_inhouse_amp/de_la_Fuente_Lab_Peptides_Inventory_outsourced.xlsx \
  --sheet 'DLF Peptide List' \
  --sequence-column Sequence --identifier-column 'Lab Code' \
  --residue-count-column 'Number of residues' \
  --n-terminus-column N-terminus --c-terminus-column C-terminus \
  --cyclic-column 'Cyclic (Position)' \
  --output-directory DataPrepare/Data/private_inhouse_amp/prepared_sequence_screen
```

正式模型分数使用 downstream MDLM 的 canonical `score_peptide_table_mic.py`、fixed-epsilon scorer、
strain key `29914`、历史 batch size 32 和 GPU3。Genome condition 使用只含 `.pt` symlink 的只读 bank，
是首次运行时对历史 loader 会误读 Evo-2 相邻 JSON manifest 的临时规避；symlink 不改变任何 tensor bytes
或 key。该通用 loader 已修复为只读取 `.pt`，后续命令直接使用 canonical `Genome_embs`；临时 symlink
bank 在 formal hash/key parity 后已删除，原 raw manifest 仍记录当时路径。运行后以固定 tokenizer revision
做精确长度审计并物化 shortlist：

```bash
PYTHONPATH="$MDLM_ROOT/src" conda run -n mdlm python \
  "$MDLM_ROOT/scripts/reproduce/score_peptide_table_mic.py" \
  --runtime-root "$MDLM_ROOT" \
  --input DataPrepare/Data/private_inhouse_amp/prepared_sequence_screen/screen_input.csv \
  --strains 29914 --config-dir "$MDLM_ROOT/configs" \
  --checkpoint Checkpoints/genome_text_learnable_emb/guidance_regressor_non_pad_t1e-3/mic_candidate_scorer_all_peptide_non_pad_t1e-3_epoch13.pth \
  --genome-embeddings DataPrepare/Data/Genome_embs \
  --genome-scale 1e14 \
  --atcc-text-embeddings DataPrepare/Data/Text_Description/ATCC/embeddings \
  --text-only-embeddings DataPrepare/Data/Text_Description/wo_ATCC/embeddings \
  --tokenizer-revision 55e83392264cb998f7aa5014847df29868aefeb8 \
  --device cuda:3 --batch-size 32 --hash-checkpoint \
  --output-directory experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/inventory_screen/model_output

PYTHONPATH="$MDLM_ROOT/src" conda run -n mdlm python \
  "$MDLM_ROOT/scripts/reproduce/peptide_inventory_screen.py" summarize \
  --inventory DataPrepare/Data/private_inhouse_amp/prepared_sequence_screen/inventory_rows.csv \
  --predictions experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/inventory_screen/model_output/peptide_mic_predictions.csv \
  --model-manifest experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/inventory_screen/model_output/manifest.json \
  --strain 29914 --target-label 'Providencia stuartii ATCC 29914' \
  --stock-column 'Remaining Weight (mg)' --stock-unit mg \
  --mic-cutoff 15 --max-token-length 1024 \
  --output-directory experiments/reviewer4_unseen_targets/providencia_stuartii_atcc_29914/analysis/inventory_screen/results
```

筛选结果已完成：4,842 个源行中 4,817 个得到模型分数，25 个 sequence 无法转换。精确 tokenizer
审计确认所有 4,817 个有效 rows 均处于 resolved DLM `model.length=1024` 范围，观测最大长度为 789。
Tokenizer repository 自带的 512 metadata 不是本 scorer 的 position contract；早期按 512 排除 188 rows
属于过度保守的汇总错误，已修复且没有重跑或改变 raw model predictions。inclusive predicted MIC
`<=15 µM` 的范围内记录为 2,164 个/1,892 个 unique sequences；其中 exact unmodified 记录 1,922 个，
进一步要求 remaining weight `>0` 后为 1,772 个/1,681 个 unique sequences。完整结果、各层 shortlist 和
hash manifest 位于 ignored
`analysis/inventory_screen/results/`。

上述 prepare/reporting 入口是通用 MDLM contract，不包含 `Providencia`、Reviewer 编号或固定 workbook 列名；
换 strain 只需让 genome/text embedding filename 解析到新的 strain key，并改变 scorer/reporting 的
`--strains/--strain`。换 molecule inventory 只需重新执行一次 `prepare` 并通过 CLI 声明列名，不新增代码。

这是 model prioritization，不是湿实验活性结果。库存表有 735 个 canonical sequence rows 声明了末端、
环化或其他 chemistry，而历史 scorer 只编码氨基酸 sequence；其中达到门槛的 242 行单列为 chemistry
approximation，不混入 exact-unmodified shortlist。剩余重量为 0、负值或缺失的行保留审计，但不进入
in-stock shortlist。

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
- 私有 lab inventory 的 ATCC 29914 computational MIC screen 已完成；修正后的正式 in-stock
  exact-unmodified shortlist 为 1,772 rows/1,681 unique sequences。它只能用于候选排序，不能写成实验
  活性或 prospective validation。
- 正式 genome embedding 只由 `ApexOracle-Evo2` canonical CLI 生成；早期 target-specific duplicate
  wrapper 已删除，避免 float32/bfloat16 与 manifest 双入口分叉。
- generation 尚未启动；后续启动 sampler 前仍须重新核验 GPU availability，并保持全部 raw-attempt
  denominator 与逐层 filter contract。单个 strain 不能支持 broad species/genus efficacy。

## 主数据源

- ATCC product: <https://www.atcc.org/products/29914>
- ATCC Genome Portal: <https://genomes.atcc.org/genomes/bac4c442bbed4a2f>
- NCBI Datasets assembly record: <https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_029075745.1/>
