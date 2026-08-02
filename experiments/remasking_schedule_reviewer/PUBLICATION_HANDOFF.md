# ReMDM / peptide-guidance reviewer 轮次发布交接

> 状态：2026-08-02 暂时收束。本轮补实验、结构审计、正式 reviewer 图、论文 Methods/
> Supplementary Fig. C4 和三处 reviewer response 已完成；尚未执行 Git commit、push 或 PR。

本文档只记录发布边界和跨仓库血缘。科学协议、精确数值及判定边界分别见
`README.md`、`RESULTS.md` 和 `STRUCTURE_AUDIT.md`。

## 1. 已由代码、文件和远程状态验证的事实

### 1.1 本轮改动所在位置

| 项目路径 | 本轮角色 | 当前状态 |
| --- | --- | --- |
| `/data2/tianang/projects/Synergy` | reviewer 实验编排、评估、结构审计、绘图、测试和内部文档 | 本轮代码主仓库；工作树还混有其他 reviewer/refactor 改动，不能整体 stage |
| `/data2/tianang/projects/discrete-diffusion-guidance` | 实际 MDLM/ReMDM guided-generation producer | 本轮只读消费；checkout 历史上已有大量本地修改，本轮没有在该仓库写代码 |
| `/data2/tianang/projects/mdlm` | pretrained MDLM、clean MIC 和 peptide classifier 资产/实现来源 | 本轮只读消费；checkout 预先 dirty，本轮没有在该仓库写代码 |
| `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech` | 正式 TeX、response DOCX 和 Supplementary Fig. C4 | 非 Git 目录；正式文稿已修改并独立渲染核验，不能从当前位置直接 push |

正式文稿资产为：

- `sn-article.tex`；
- `Response to reviewers letter.docx`；
- `Fig_SI_remasking_schedule.pdf`，SHA-256
  `23ce3a58f82b82f1fb1f458efd08152fbf1284f9b2e2f6b97cd2e1030e9bb847`；
- 修改前 DOCX 备份只用于本地恢复，不进入公共代码仓库。

### 1.2 本轮 canonical 代码

生成与评估入口：

- `scripts/reproduce/prepare_remasking_schedule_reviewer_tasks.py`；
- `scripts/reproduce/run_remasking_schedule_reviewer.py`；
- `scripts/reproduce/orchestrate_remasking_schedule_reviewer.py`；
- `scripts/reproduce/evaluate_remasking_schedule_reviewer.py`。

审计与绘图入口：

- `scripts/audit/audit_remasking_peptide_classifier_structure.py`；
- `scripts/audit/plot_remasking_schedule_reviewer.py`，只保留为 legacy classifier-only 图入口；
- `scripts/audit/plot_remasking_structure_qualified_peptides.py`，为当前 canonical reviewer 图入口。

验证入口：

- `tests/test_remasking_schedule_reviewer.py`；
- `python -m pytest -q tests/test_remasking_schedule_reviewer.py`，最近一次结果为 `8 passed`。

上述 8 个代码/测试文件合计约 142 KB。连同紧凑 README、结果、manifest、provenance、summary
和本交接文档，当前未忽略 reviewer capsule 共 414,113 bytes，约 0.41 MB。

### 1.3 GitHub 与 Git remote 审计

| Checkout | Remote / visibility | 已验证状态 | 本轮发布判断 |
| --- | --- | --- | --- |
| `Synergy` | `DragonDescentZerotsu/Synergy`，private | local `main` 与 `origin/main` 均为 `f52626b`; reviewer 文件尚未提交 | 可作为当前 capsule 的临时集成仓库，但必须从 clean worktree 做显式白名单提交 |
| `mdlm` | 上游 `kuleshov-group/mdlm`；自有 `DragonDescentZerotsu/ApexOracle-MDLM`，public | local HEAD 与自有 remote 的 `master` 均为 `7a6a7d1`；当前 branch 跟踪上游 `origin`，checkout dirty | 本轮没有需要推送的改动；未来清理后应显式 push 到 `custom`，不能误推上游 |
| `discrete-diffusion-guidance` | 只有上游 `kuleshov-group/discrete-diffusion-guidance` | local HEAD/upstream main 为 `edb0f8c`；checkout 含 ApexOracle 历史修改且 dirty；不存在自有同名 GitHub fork | 当前不能推送；须先建立 clean fork/独立仓库并参数化机器路径，再固定 commit |
| `ApexOracle_github` | `DragonDescentZerotsu/ApexOracle`，public | clean，但 packed history 约 235 MiB，已包含大数据、模型和外部仓库副本 | 不应把本轮工作直接复制进该 legacy monorepo，否则会继续膨胀并混淆 producer 血缘 |
| `ApexOracle_cleaned` | 无 Git metadata | 正式论文修改只存在本地文件系统 | 后续若公开文稿，只迁移精简的 TeX/bib/必要 figure，不迁移编译缓存和 DOCX 备份 |

本轮没有执行 commit、push、创建 fork 或 PR。

## 2. 公共发布白名单与禁止项

### 2.1 建议提交

1. 上述 8 个 canonical 脚本/测试；
2. 本目录的 `README.md`、`RESULTS.md`、`STRUCTURE_AUDIT.md`、`PUBLICATION_HANDOFF.md`；
3. `task_manifest.json`、`analysis/summary.json`、
   `analysis/peptide_structure_audit/summary.json`、紧凑 provenance/figure manifest 和 caption；
4. 与入口发现有关的 `.gitignore`、`AGENTS.md`、`REFACTOR_PLAN.md`、
   `docs/COMPUTE_AND_ASSET_MAP.md`、`experiments/README.md` 和 `scripts/audit/README.md` 中的
   reviewer-specific 修改。

紧凑 provenance/manifest 中的绝对路径和机器名是历史取证信息，不是可移植配置。公开时可以保留
它们以验证原运行，但必须同时提供 CLI 参数和面向公共环境的示例；不得把本机路径描述成用户必须
复现的目录结构。

### 2.2 不得提交

- `experiments/remasking_schedule_reviewer/runs/`、`smoke/` 和 superseded smoke runs；
- `sampler.log`、queue state、逐 batch raw generation 和 completion marker；
- `analysis/evaluated_attempts.csv`、逐结构 audit CSV 或其他逐 attempt 大表；
- PNG/PDF/SVG 的 legacy 重复版本；公共仓库最多保留论文实际引用的一份 figure，或由脚本重建；
- checkpoint、embedding、Arrow dataset、原始/私有训练数据、node asset copies；
- TeX build auxiliaries、临时渲染目录、DOCX 备份和一次性转换产物；
- 外部 `mdlm` 或 `discrete-diffusion-guidance` 的整个 dirty checkout。

现有 `.gitignore` 已覆盖 raw runs、smoke、逐 attempt CSV、图片二进制、checkpoint 和常见缓存。
发布前仍须检查 staged 内容，不能以 `.gitignore` 代替白名单审查。

## 3. 建议的公共仓库整合顺序

### 阶段 A：隔离并提交本轮 reviewer capsule

1. 从 `Synergy` 当前远程 `main` 建立 clean worktree/branch；
2. 只迁移第 2.1 节列出的 reviewer-specific 文件或 hunk，禁止 `git add -A`；
3. 运行单元测试、入口帮助检查、敏感信息扫描和 staged-size 审计；
4. 先提交到 `Synergy` 的独立 branch/PR，保持与其他未完成 reviewer 工作解耦。

当前工作树中的共享文档同时包含多个 reviewer/refactor 更新，因此不能只凭文件名整体提交后声称
它们只属于本轮；应在 clean worktree 中按 hunk 移植并复读。

### 阶段 B：清理两个外部 producer

- **MDLM：** 以公开的 `ApexOracle-MDLM` 为目标，创建独立 release branch；只收录公共运行真正
  需要的修改，删除临时脚本、缓存和硬编码路径。由于本地 `master` 跟踪上游 `origin`，发布时须
  显式指定 `custom` remote。
- **Discrete guidance：** 先创建自有 fork 或新的 clean repository；从已核验 producer 文件
  提取最小 patch，参数化数据/权重/output roots，加入环境与 smoke test，再产生一个可固定的 clean
  commit。不要把当前 dirty checkout 直接推向上游。

### 阶段 C：统一公共 ApexOracle repo

未来统一仓库建议保留：

```text
src/apexoracle/                    # 共享模型和数据接口
scripts/reproduce/                 # 可复现实验入口
scripts/audit/                     # reviewer/结果审计
experiments/.../                   # 紧凑 manifest、summary 和说明
external/mdlm                     # 固定 clean commit/submodule 或版本化依赖
external/discrete-diffusion-guidance
configs/model_weights.yaml         # 权重身份与外部下载位置
paper/                             # 如需公开，仅精简 TeX/bib/正式 figure
```

不建议在已有约 235 MiB packed history 的 `ApexOracle_github` 上继续复制完整外部仓库和二进制。
正式统一发布应选择经过清理的新 release history/branch 或新的仓库布局，并以 submodule、固定 commit
或版本化依赖连接 producer；权重和数据继续由 manifest 指向 Zenodo/Hugging Face 等外部资产。

## 4. 仍待作者确认的事项

1. 统一公共 repo 最终沿用 `DragonDescentZerotsu/ApexOracle`，还是建立新的 clean-history repo；
2. `discrete-diffusion-guidance` 采用 GitHub fork 还是独立的 ApexOracle-specific producer repo；
3. 正式论文 TeX 是否随代码公开，response DOCX 是否只作为投稿资产保留；
4. 外部代码、模型权重和数据的 license/再分发条件；
5. 是否需要把 canonical Supplementary Fig. C4 PDF 纳入公共 repo，或仅保留可重建脚本、caption
   和 source hashes。

在这些决定确认前，可以安全完成阶段 A 的 clean、path-scoped reviewer capsule，但不应把 dirty
外部 checkout 或 legacy 大仓库做整体合并。

## 5. 发布前验证命令

```bash
python -m pytest -q tests/test_remasking_schedule_reviewer.py

git --git-dir=.git-state --work-tree=. status --short --ignored
git --git-dir=.git-state --work-tree=. diff --cached --stat
git --git-dir=.git-state --work-tree=. diff --cached --check

# staged 文件大小和可疑凭据须在 clean worktree 中再次检查
git --git-dir=.git-state --work-tree=. diff --cached --name-only
```

验收条件是：测试通过、没有 raw runs/权重/私有数据/重复二进制进入 staged set、外部 producer 由
明确 commit 标识、公开命令不依赖作者机器的绝对路径。
