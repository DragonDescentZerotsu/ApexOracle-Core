# ApexOracle-Core 公开发布审计

> 建立日期：2026-08-10
> 审计基线：`Synergy/main` `56c57e51b0bc594e23609f7996de55b55946f716`
> 当前结论：**代码基线可构建且测试通过，但仓库尚不得改为 public。**

本文只记录当前 `Synergy` 原仓库转换为 `ApexOracle-Core` 的公开发布边界。最终仍复用同一个
GitHub repository 和完整科学代码 history，不创建第二份 Core repository。任何删除都必须能从
`legacy-code-snapshot-2026-07-17`、远端 archive branch 或本地审计备份恢复。

## 1. 已由代码、Git 和运行日志验证的事实

### 1.1 Repository 与 history

- GitHub repository `DragonDescentZerotsu/Synergy` 当前为 **private**，默认分支为 `main`，根许可为
  MIT。
- 当前 reachable history 为 132 commits；远端有 6 个 branches 和 2 个 annotated tags。
- 原始 source recovery point 已由 annotated tag `legacy-code-snapshot-2026-07-17` 和 remote branch
  `archive/legacy-code-snapshot-2026-07-17` 双重保存。
- 当前 tracked tree 有 448 files；reachable history 与当前 tree 的最大 blob 均为
  `experiments/peplink_validation/peplink_0.1.2/roundtrip_records.csv`，大小 4,645,044 bytes。
- current tree 和 reachable history 都没有 `.pt`、`.pth`、`.ckpt`、`.safetensors`、NumPy、FASTA、
  archive 或 wheel 等模型/数据二进制对象；不存在接近 GitHub 100 MB 限制的 blob。
- 常见 GitHub/Hugging Face/AWS/OpenAI/Google/OAuth token、private-key header 和硬编码 password
  pattern 对 current tree 与全部 reachable commits 的扫描均为 0 hit。legacy
  `DataPrepare/resistant_gene_check.py` 的 `api_key_list` 已核验为从 `GEMINI_API_KEYS` 环境变量读取，
  不包含 literal key。

### 1.2 Canonical package 与验证

- `src/apexoracle/` 中没有 `/data*`、`/home/*`、`/root/*` 或 `/mnt/*` 作者机器绝对路径。
- `conda run --no-capture-output -n base python -m pytest -q`：`206 passed`，14 条 warning；warning
  均为现有 dependency/API deprecation 或合成常数测试的统计 warning。
- 基线 `python -m build` 成功生成约 171 KiB wheel 和 188 KiB sdist；后续 clean candidate 已将
  `Readme.md` 标准化为 public-facing `README.md`，将 license metadata 改为 SPDX `MIT`，并新增根
  `NOTICE`。当前 build 不再产生缺失标准 README 或旧 license-table warning。
- 从 remote `main` 重新 shallow clone，在新 venv 中 `pip install --no-deps` 后：
  `import apexoracle`、`apexoracle-run-hierarchical-mic --help` 和
  `apexoracle-run-synergy-cv --help` 均通过。
- 根 MIT license 与 vendored PeptideCLM tokenizer 的独立 MIT license 都存在；vendor README 固定了
  upstream URL、revision 和 vocabulary/merge hashes。

### 1.3 当前不可移植范围

- canonical Python package 为 0 个绝对路径文件；`scripts/` 有 6 个、legacy `DataPrepare/` 有 13 个
  tracked files 含作者机器绝对路径。
- `environment.yml` 是 563 行的完整机器环境导出，不适合作为 Core 最小安装环境。
- `configs/model_weights.yaml` 仍把多个本机 checkpoint path 作为历史来源记录；公开 quickstart 需要
  使用稳定 URI、revision、size、SHA-256 和 redistribution status，不能依赖这些本机路径。
- Core 已有严格 hierarchical MIC checkpoint loader 和 inference-only checkpoint contract，但公开
  MIC inference checkpoint、molecule/strain example assets 与稳定下载 URI 尚未闭合，因此还不能声称
  fresh end-to-end inference 已完成。

## 2. 发布阻塞项

### P0：数据库派生 row-level 文件的再分发边界

`experiments/peplink_validation/peplink_0.1.2/roundtrip_records.csv` 含 16,896 条 record-level
结果和 DBAASP identifiers。它由 commit `76ab6a1d821715519ae8245f80d7265cde9379c9` 加入，当前只存在于
remote `main` 与 `refactor/mdlm-bridge` 的 history；2026-07-17 recovery tag/branch 不含该文件。

2026-08-10 核验的 DBAASP 官方条款同时出现“public domain / freely distributed”与 visitor
“Non-Distribution of Data”要求，文字存在直接冲突。仓库公开前必须二选一：

1. 获得 DBAASP 对该派生 row-level audit table 的明确再分发许可，并在 data notice 中引用；或
2. 从 active tree 和所有将公开的 reachable refs/history 中移除该 16,896-row table，只保留 compact
   aggregate、生成脚本、输入 manifest/hash 和 reviewer Supplementary Data 的必要最小表。

在作者确认采用哪一种之前，repository 必须保持 private。仅在 current tree 删除文件不足以解决问题，
因为 public Git history 仍可取得旧 blob。若采用方案 2，必须先建立离线 Git bundle，再做精确单路径
history rewrite；不得泛化为删除全部实验 CSV。

### P0：未完成 reviewer 工作不得混入 Core baseline

当前 worktree 有 Providencia/reviewer 相关 tracked modifications 和 untracked source。它们属于另一条
尚未收口的任务，不得用 `git add -A` 混入 Core release commit。Core recovery tag、release branch 和
最终 gitlink 必须基于显式暂存并通过 clean-clone 复核的 commit。

### P1：active tree 收口

- 完整 `DataPrepare/` ledger 已建立于 `docs/DATAPREPARE_LEGACY_LEDGER.md`，46/46 source SHA-256
  已验证。唯一独立 producer 已迁为参数化 strain-text embedding library/CLI：7 项 synthetic tests、
  ATCC/text-only 两条真实 H100 parity 与逐输出 manifest 均通过。全部 46 个旧 source 已从 active tree
  删除，只由 recovery tag 恢复；ignored 的 35 GB `DataPrepare/Data/` 资产原地保留。下一 gate 是重跑
  全仓 release checks。
- 将 563 行机器环境导出替换为可维护的 Core package/test profiles。
- `README.md`、SPDX `MIT` metadata 与根 `NOTICE` 已完成；最终 fresh-clone gate 继续验证 wheel/sdist
  同时包含 license/notice，并保持 public-facing 安装、模块边界与 asset policy 一致。
- 将 runtime asset lookup 全部改为 manifest + environment override；本机 path 只能保留在明确标注的
  provenance 文档，不能作为公共默认值。

### P1：公开 inference 资产与 quickstart

- 为至少一个正式 hierarchical MIC inference checkpoint 决定 redistribution status，生成 inference-only
  copy，登记 stable URI/revision/size/SHA-256。
- 为一个 known strain 和一个 example molecule 登记可公开的 molecule、genome/text embedding 输入；
  如果某项不能再分发，只能提供下载/生成说明，不能把本机 asset 当作 public example。
- 新增稳定 prediction CLI/API，并在空 cache/fresh clone 中运行一条真实 inference；synthetic tensor
  smoke 只验证代码 contract，不能替代这一步。

## 3. 根据现有证据作出的判断

- `src/apexoracle/`、严格 checkpoint loader、训练/evaluation modules 和现有 tests 可以作为 Core clean
  release 的主体，不需要再次大规模重排 package。
- 剩余工作主要是 legacy active-tree 清理、数据许可、资产发布和一个薄 inference 入口；不应改写已验证
  fusion/head/checkpoint schema。
- 旧 `DataPrepare/` 不是公共 API。其文件数少于 canonical `src/`/`scripts/`，且多数作用已有 ledger、
  replacement 或 recovery tag，因此完成逐文件 gate 后应从默认分支移除，而不是继续展示为推荐代码。
- Git history 当前没有 credential 或大型模型资产问题；是否需要 rewrite 只取决于 DBAASP row-level
  distribution decision，不应为了“看起来更干净”重写其他历史。

## 4. 仍待作者确认的事项

- DBAASP 16,896-row round-trip audit：取得明确许可，或执行带离线 bundle 的单路径 history rewrite。
- 哪一个正式 hierarchical MIC checkpoint family 作为首个 public prediction quickstart；这决定需要发布
  的 checkpoint、strain embeddings 和 molecule embedding profile。
- reviewer/Providencia 当前 worktree 是先独立提交到 `main`，还是在 Core release candidate 中明确排除。

## 5. 固定执行顺序与验收门槛

1. 冻结当前 committed Core baseline，建立 source manifest 和 annotated pre-public cleanup tag。
2. `DataPrepare/` 逐文件 ledger、唯一功能迁移和 active-source 删除已完成；执行全仓回归。
3. 完成 README/environment/pyproject/NOTICE、绝对路径和 asset-manifest 清理。
4. 关闭 DBAASP 再分发决策；如需 rewrite，先生成并校验离线 Git bundle，再只改命中的路径和 refs。
5. 完成 clean wheel/sdist、全仓 tests、fresh-clone install/import/CLI 和真实 inference smoke。
6. 保持 repository private，直到以上门槛全部通过；随后才把同一 repository 重命名为
   `DragonDescentZerotsu/ApexOracle-Core`、切换 public，并把固定 Core commit 加入现有 ApexOracle
   super-repo。

官方条款核验入口：

- DBAASP Terms and Conditions: <https://dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf>
- DBAASP API policy: <https://dbaasp.org/api?page=rest>
- PubChem downloads and source-specific licensing: <https://pubchem.ncbi.nlm.nih.gov/docs/downloads>
