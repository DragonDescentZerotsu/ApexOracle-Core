# AMP data pipeline 与 PepLink 外部依赖

本目录记录 ApexOracle 对独立 PepLink 工具的依赖边界及论文 peptide structure 数据血缘。
PepLink 不会复制到 ApexOracle 仓库，也不作为 Git submodule。ApexOracle 通过可选依赖安装：

```bash
pip install -e '.[peplink]'
```

对应公开包为 `PepLink==0.1.1`，Git tag 为 `v0.1.1`，commit 为
`cec2a02427766e4ba95806924801af31bdcc9939`。上游仓库：
<https://github.com/DragonDescentZerotsu/PepLink>。

## 为什么不用 Git submodule

PepLink 已经具备独立的 PyPI 发布、MIT license、公开 API、版本 tag 和测试。Git submodule
只会把 source checkout 生命周期耦合到 ApexOracle，不能改善普通用户的安装体验。当前采用：

- `pyproject.toml` optional dependency 固定 `PepLink==0.1.1`；
- `src/apexoracle/data/peplink_adapter.py` 只调用 `from_dbaasp_record` 和
  `aa_seqs_to_smiles` 两个公开 API；
- `configs/data_pipeline/peplink_v0.1.1.yaml` 固定 PyPI artifact、Git revision 和 paper data
  SHA-256；
- PepLink 自身的升级和发布继续在独立仓库完成。

## 已由代码和真实数据验证的事实

- PepLink 工作树 clean，本地 `main`、`origin/main` 和 tag `v0.1.1` 均指向上述 commit。
- 使用 Synergy 的同一份 DBAASP JSON 运行独立仓库测试：`22 passed`。
- PepLink 内置 amino-acid mapping 与论文数据使用的
  `all_aa_smiles_new_handcrafted.csv` SHA-256 完全一致：
  `42fe91726f2d6a8fb951d7ef7d27298f3151bb45bc5fa222c5d7d9d9e5777544`。
- 旧 structure-correction 表共 179 条：169 条由 peptide builder 重建、8 条已知异常记录保留
  DBAASP offered SMILES、2 条 non-monomer 保留 offered SMILES。
- PepLink v0.1.1 与历史 correction output 有 177/179 条 SMILES 字符串完全一致；全部
  179/179 条在 PepLink 的 `Cleanup → FragmentParent → Uncharger` 规则下结构一致。
- 两条非字符串一致记录为 DBAASP 19000 和 21769。v0.1.1 分别移除了 legacy 输出中的两个
  游离 fragment 和一个游离 methane；主体 peptide structure 不变。
- 这两个 ID 在最终 120,955 行 token cache 中合计出现 9 行（2 + 7）。完整审计见
  `peplink_compatibility_audit.json`。

## 论文复现与新数据的边界

- **论文复现：** 读取由 SHA-256 标识的 frozen
  `DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv` / token cache，不用当前 PepLink 重新生成后冒充原始
  paper data。
- **新数据构建：** 使用 PepLink 0.1.1 的规范化输出。两个游离 fragment 清理属于明确的
  versioned data change。
- **完整数据处理现状：** MIC parsing、in-house merge 和 SELFIES/token filtering 已迁移并
  完成真实数据验证，结果见 `paper_data_reconstruction_audit.json`。

审计命令：

```bash
python scripts/audit/audit_peplink_compatibility.py \
  --peplink-source /path/to/PepLink \
  --output experiments/amp_data_pipeline/peplink_compatibility_audit.json
```

安装 PyPI 包后可省略 `--peplink-source`。

## MIC、in-house merge 与 token filtering

canonical 实现位于：

- `src/apexoracle/data/amp_mic.py`：MIC/inhibition 选择、浓度解析和 µg/ml→µM；
- `src/apexoracle/data/amp_training_data.py`：in-house wide→long、表合并、SELFIES/token filter；
- `scripts/prepare_data/build_amp_mic_dataset.py` 和
  `scripts/prepare_data/build_amp_training_dataset.py`：只读 CLI。

所有 CLI 都拒绝输入输出同路径，也拒绝覆盖已经存在的输出。建议始终使用新的 `results/`
子目录，例如：

```bash
python scripts/prepare_data/build_amp_mic_dataset.py \
  --dbaasp-json DataPrepare/Data/all_peptides_data.json \
  --smiles-csv DataPrepare/Data/DBAASP_id_SMILES_merged.csv \
  --molecular-weight-smiles-overrides \
    DataPrepare/Data/DBAASP_id_wo_PubChem_SMILES_w_DBAASP_smiles.csv \
  --output-dir results/amp_data_rebuild/mic

python scripts/prepare_data/build_amp_training_dataset.py merge \
  --dbaasp-mic DataPrepare/Data/DBAASP_id_bact_name_SMILES_MIC_Evo.csv \
  --inhouse-mic DataPrepare/Data/inhouse_Evo_style_SMILES_MIC.csv \
  --output results/amp_data_rebuild/DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv

python scripts/prepare_data/build_amp_training_dataset.py tokenize \
  --input DataPrepare/Data/DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv \
  --output results/amp_data_rebuild/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv \
  --revision 55e83392264cb998f7aa5014847df29868aefeb8 \
  --local-files-only
```

### 已由代码和真实数据验证的事实

- DBAASP MIC 重建得到相同的 105,547 行；ID、行顺序、strain 和 corrected SMILES 全部精确
  一致。MIC 最大绝对误差为 `4.55e-13`，621 个文本差异来自历史 pandas float CSV
  序列化，不是标签含义变化。
- structure correction 的历史顺序已经显式复现：179 条 correction 之前的 SMILES 只用于
  分子量换算，最终表展示 correction 之后的 SMILES。它消除了 5 个 DBAASP ID、21 行 MIC
  的实质差异。
- frozen DBAASP MIC 与 frozen in-house long table 合并后，121,265 行 CSV 逐字节一致。
- IBM SELFIES tokenizer 固定到 revision
  `55e83392264cb998f7aa5014847df29868aefeb8`；310 行因超过 1024 tokens 排除，invalid/UNK
  均为 0。最终 120,955 行 token cache 逐字节一致。
- 用 PepLink 0.1.1 从 1,642 条 in-house sequence 新建结构时，ID、strain 和 MIC 全部一致；
  legacy 结构各含一个显式 terminal `[OH]`，PepLink 输出等价 canonical `O`。归一化该写法后
  15,718/15,718 行一致。因此论文复现读取 frozen in-house long table，新数据使用 PepLink
  canonical 结构。

`paper_legacy` 不是新的推荐科学协议。它只用于诚实复现论文数据，包括 inhibition unit 的
历史选择行为以及 `>`/`>>` 的倍增规则；未来修正规则必须使用新 protocol/version，不能静默
替换 frozen 数据。
