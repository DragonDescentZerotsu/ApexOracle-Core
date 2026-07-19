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
- **不能声称的事项：** 177/179 exact 不等于论文完整 121,265 行数据已经从零重建；MIC
  parsing、in-house merge 和 SELFIES/token filtering 仍需分别迁移和验证。

审计命令：

```bash
python scripts/audit/audit_peplink_compatibility.py \
  --peplink-source /path/to/PepLink \
  --output experiments/amp_data_pipeline/peplink_compatibility_audit.json
```

安装 PyPI 包后可省略 `--peplink-source`。
