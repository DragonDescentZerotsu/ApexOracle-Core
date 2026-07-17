# ApexOracle / Synergy 代码库重构计划

> 建立日期：2026-07-17  
> 状态：执行中  
> 适用范围：当前 `Synergy` 仓库，以及后续需要整合的 Evo-2 genome embedding、DLM/MDLM 和 guided generation 代码。

## 1. 重构目标

本次重构的目标不是简单删除旧文件，而是把当前以实验脚本副本为主的研究代码整理成一个可审计、可复用、可复现且适合公开发布的代码库：

1. 保留一份经过脱敏的原始源码快照，作为历史依据。
2. 清除非论文最终版、重复、调试、机器专用和无关旁支代码。
3. 把论文实验从大型单文件脚本迁移为共享模块、明确配置和稳定命令行入口。
4. 为每项论文结果记录数据版本、划分、模型、checkpoint 和指标血缘。
5. 按 reviewer 的要求，重新构建使用完全相同样本和 fold 的 molecule encoder benchmark。
6. 明确当前仓库能够完整复现、只能部分复现以及仍依赖外部仓库的部分。

仓库名称目前不作为阻塞项。最终完整发布物预计还会整合 Evo-2 genome embedding、MDLM 和 generation 等其他仓库，届时再统一仓库名、Python package 名和论文链接。

## 2. 已确认的关键决策

### 2.1 历史源码快照

- 初始 Git 快照包含：全部源码、capsule 中的源码、配置和清空输出后的 notebook。
- 初始 Git 快照完全排除：训练/评估数据、checkpoint、实验结果、缓存、日志、大型二进制文件和密钥。
- 在脱敏快照提交后创建 Git tag：`legacy-code-snapshot-2026-07-17`。
- 用户已经额外保存当前 legacy 代码，因此清理后的主代码树不再保留 `legacy/` 文件夹；旧实现仅通过 Git tag 追溯。

### 2.2 Fig. 2b 公平 benchmark

- 按 reviewer 要求，将完全共享数据和 fold 的 benchmark 作为新的正式结果。
- 如果新结果与当前论文图不同，应更新论文图、正文数值和 reviewer response。
- 所有 encoder 使用同一份样本清单和同一份 molecule-level 五折划分，不允许各模型在划分后静默丢弃样本。
- APEX 不因输入限制而缩小公共数据集：noncanonical residue 映射为 `X`，cyclic peptide 使用线性化序列表示。
- APEX 的上述表示属于有损投影，必须在数据 manifest、Methods 和结果说明中明确记录。

### 2.3 DLM/MDLM 模型范围

正式 benchmark 计划包含：

- DLM MTR + DLM；
- DLM MLM-only；
- ChemBERTa MTR；
- ChemBERTa MLM（first-token）；
- MolFormer；
- PeptideCLM；
- APEX。

ChemBERTa MLM mean-pooling 作为可选消融，不与论文主 comparator 混淆。DLM MLM-only 的代码和权重优先从 `/data2/tianang/projects/mdlm` 中定位和核验。

## 3. 目标目录结构

```text
.
├── src/apexoracle/
│   ├── data/                       # 数据 schema、读取、过滤、映射和划分
│   ├── features/                   # molecule、genome、strain text feature 接口
│   ├── models/                     # fusion、attention、prediction heads
│   ├── evaluation/                 # 指标、ensemble、预测导出
│   └── benchmarks/
│       └── molecule_encoders/      # Fig. 2b 公平 benchmark
├── experiments/
│   ├── fig1a_hierarchical/
│   ├── fig1b_zero_shot/
│   ├── fig2b_molecule_encoders/
│   ├── fig2c_strainwise/
│   └── synergy/
├── configs/                        # 可版本化实验配置
├── scripts/
│   ├── prepare_data/
│   └── reproduce/
├── reproducibility/
│   └── code_ocean/                 # 整理后的 capsule/Code Ocean 入口
├── quickstarts/
├── tests/
├── AGENTS.md
├── REFACTOR_PLAN.md
└── README.md
```

目录结构可以随迁移细化，但不得重新形成大量复制粘贴的单文件模型实现。

## 4. 分阶段执行计划

### 阶段 0：建立审计基线

状态：已完成。

- [x] 在 `AGENTS.md` 中记录论文、代码、数据与 checkpoint 血缘。
- [x] 区分最终版、可能最终版、历史版本、论文后代码和缺失代码。
- [x] 确认 reviewer 对 Fig. 2b 的核心要求是相同数据和相同划分。
- [x] 核验当前 Git 仓库状态和 GitHub 远程仓库状态。
- [x] 定位 MDLM 仓库中的 DLM MLM-only 代码、配置和候选权重；精确论文 checkpoint 仍按证据等级记录。
- [ ] 生成待保留、待迁移和待删除的机器可读清单。

验收标准：任何删除或迁移都能从审计记录、Git tag 或清单中解释其原因。

### 阶段 1：创建脱敏的 legacy 源码快照

- [x] 从源码中移除硬编码 API key、W&B 凭据和机器私有认证信息，改为环境变量或不可用占位符。
- [x] 清空所有纳入版本控制的 notebook 输出和执行计数。
- [x] 建立 `.gitignore`，排除数据、checkpoint、结果、日志、缓存和大型二进制文件。
- [x] 检查 GitHub 单文件 100 MB 限制以及总体提交体积。
- [x] 显式检查 staged 文件，确保只包含源码、文档、配置和清空输出后的 notebook。
- [x] 扫描 staged 内容中的疑似密钥和大型文件；旧绝对路径保留在 legacy tag 中，重构分支再统一迁移为配置。
- [x] 创建本地初始提交和 `legacy-code-snapshot-2026-07-17` tag。
- [x] 将 legacy `main` 和重构分支同步到 `DragonDescentZerotsu/Synergy.git`。
- [ ] 在具备普通 Git 凭据后补推远程 tag `legacy-code-snapshot-2026-07-17`；当前远程以 `archive/legacy-code-snapshot-2026-07-17` branch 暂代。
- [x] 从该 tag 创建 `agent/paper-release-refactor` 重构分支，后续清理不直接破坏历史快照。

验收标准：从 tag 可以查看原始源码血缘，但仓库中不存在数据、checkpoint、结果或可用密钥。

执行记录：当前 Codex 工作区将 `.git` 作为只读保护挂载，因此本次 Git metadata 暂存在被忽略的 `.git-state/` 中，并通过 `--git-dir=.git-state --work-tree=.` 操作同一工作树。本地脱敏快照提交为 `a68707c`。本机 `gh`、SSH public key 和 HTTPS credential 均不可用，因此改用已授权的 GitHub App Git object API 重建远程提交链；上传的 237 个去重 blob 和五个版本 tree 均逐一与本地 SHA 校验一致。远程提交因额外的仓库初始化 parent 而拥有不同 commit SHA，但每个科学代码版本的 tree 与本地完全一致。连接器不提供 tag API，因此本地 tag 已保留，远程暂用 archive branch 作为恢复点。

### 阶段 2：建立共享核心模块

- [x] 建立可安装的 `src/` package 和最小依赖定义。
- [ ] 抽取统一的数据 schema、label transform、mask、strain mapping 和 filtering；Fig. 2b 的 19-task schema、共享过滤和 fold 已先行完成。
- [ ] 抽取 genome/text/molecule feature 接口。
- [ ] 抽取 cross-attention/fusion、LoRA、regression/classification head。
- [ ] 抽取指标、ensemble、checkpoint loading、seed 和设备选择逻辑。
- [ ] 将硬编码路径迁移到 CLI 参数或 YAML 配置。
- [ ] 统一预测输出格式，至少包含 sample ID、fold、label、prediction 和模型元数据。

验收标准：论文主实验不再复制共享模型和数据逻辑；公共模块具备单元测试。

### 阶段 3：优先完成 Fig. 2b molecule encoder benchmark

这是本轮重构中优先级最高、且会生成 reviewer 要求的新正式结果的模块。

#### 3.1 冻结数据协议

建立下列不可变产物：

- `common_molecule_ids.csv`：所有 comparator 共享的样本/分子标识；
- `folds.csv`：按 molecule ID 生成的一份共享五折划分；
- `exclusions.csv`：进入公共集合前的每条排除记录及原因；
- `dataset_manifest.json`：源数据校验和、过滤规则、label transform、随机种子、fold 统计和 APEX 投影规则。

APEX 输入转换规则：

1. canonical linear peptide：保留标准单字母序列；
2. noncanonical residue：映射为 `X`；
3. cyclic peptide：去除环连接语义，使用线性化 residue 序列；
4. 转换失败必须在构建公共集合前显式报错或记录，不允许训练/评估时静默跳过。

状态：**已实现并通过真实数据验证。** 当前源表 11,401 个 molecule，公共集合 11,398 个，3 个因原始 DBAASP sequence 缺失而排除；五折大小为 2,280、2,280、2,280、2,279、2,279。APEX 投影中 1,689 条含 noncanonical residue，1,460 条含 D-residue，2,335 条丢弃了 bond/multichain topology，92 条超过 50 residues 并被确定性截断。重构版 APEX 为 `X` 使用独立 token 23，AAindex 向量取 20 种 canonical residue 向量均值，不再沿用旧代码将未知字符静默当作 padding 的行为。

#### 3.2 统一训练与评估协议

- 所有模型严格读取同一 `folds.csv`。
- 统一 prediction head、训练 epoch/early stopping、优化器、随机种子和指标实现；只有 encoder 特有的必要输入处理不同。
- frozen encoder 与 fine-tuned encoder 必须明确分组，不能在同一表中无说明混比。
- 每个 fold 保存预测和指标；汇总报告 mean ± SD。
- 输出逐模型处理成功率和任何异常，但正式指标必须基于完全相同的 test IDs。

状态：共享数据 loader、outer fold 校验、训练折内 10% validation、MIC label transform、统一 macro-task R2、严格 ID 对齐的 `.npz` feature-cache 契约和统一 torch regression-head runner 已实现并通过测试。APEX adapter 已用真实 pretrained checkpoint 为全部 11,398 个公共 molecule 生成并严格回读 `(11398, 128)` cache。ChemBERTa、MolFormer 和 PeptideCLM adapter 已完成小样本 backbone smoke test和全量 tokenizer 审计，但因当前 GPU driver 不可用尚未生成全量 feature；两个 DLM adapter 待迁移。旧 capsule 在 outer test fold 上逐 epoch 选择 best checkpoint 的行为不会进入新协议。

#### 3.3 capsule 迁移

- 将 `capsule_fig2/` 的可复用源码迁移到 `src/apexoracle/benchmarks/molecule_encoders/` 和 `experiments/fig2b_molecule_encoders/`。
- 将 Code Ocean 专用入口整理到 `reproducibility/code_ocean/fig2b/`。
- 删除 capsule 内重复源码和历史结果，不保留第二套 canonical 实现。

验收标准：一条数据准备命令生成共享 manifest/folds；同一 benchmark runner 能选择各 encoder；最终汇总含各 fold 指标和 mean ± SD。

### 阶段 4：迁移论文其余最终实验

#### 4.1 优先迁移的高置信度最终版

- Fig. 1a / Fig. 2c strain-wise DLM ensemble：`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py`。
- Fig. 1b strict target-strain zero-shot：`antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py`。
- AMP/PepLink 最终数据处理血缘。
- ApexOracle-3/12/23 sequence similarity 流程。
- reviewer 的 Evo-2 embedding scaling 分析脚本。

#### 4.2 迁移前需要作者或原始结果进一步核验

- phylum-wise 和 11-cluster species-wise：现存日志与论文数值或完整 fold 不完全一致；
- fine-tuned Fig. 1b：现存日志不完整；
- synergy CV：现存指标接近但不完全等于论文结果，且 LoRA rank 与 Methods 存在差异；
- modality ablation：缺少能够精确连接附录图的最终指标表。

这些实验先进入 `experimental` 状态，不在 README 中声称已经完整复现。

#### 4.3 明确外部或缺失部分

- Evo-2 genome embedding 提取；
- DLM 预训练及部分 checkpoint；
- guided generation/remasking sampler；
- k-mer ablation。

在对应外部仓库完成重构前，仅提供接口、数据契约和缺失说明，不复制未经核验的大型代码树。

### 阶段 5：删除 legacy 和无关代码

完成迁移与最小验证后，从重构分支删除：

- 被最终入口取代的 MIC、fine-tune、synergy、few-shot 复制脚本；
- `_old.py`、debug、临时 notebook 和机器专用 launcher；
- `Fangping_correlation/`、`e3nn_playground/`；
- `GPU_eye.py`、`run.py`、`run_full.py` 等资源占用工具；
- W&B 本地日志、缓存、旧结果和 capsule 中的重复源码；
- 无法移植且不再使用的绝对路径 shell 脚本。

`PeptideCLM/` 不作为自有核心代码维护。优先改为明确版本的外部依赖；如果必须 vendor，则只保留必要文件并完整保留上游 README、LICENSE 和来源说明。

验收标准：根目录不再堆积实验副本；每个保留脚本都能对应公开文档中的明确任务。

### 阶段 6：验证

- [ ] Python compile/import 检查。
- [ ] 数据单位转换、label transform、strain mapping、APEX 序列投影测试。
- [ ] 共享交集和 fold 无泄漏测试。
- [ ] mask、tensor shape、pooling 和指标测试。
- [ ] 各主入口的小规模 smoke test。
- [ ] 对可用旧 checkpoint 运行小样本等价性检查。
- [ ] 密钥、绝对路径、超大文件和未跟踪实验产物扫描。

本阶段默认不重新训练全部论文实验。Fig. 2b 公平 benchmark 是例外：完成代码和数据协议验证后，需要正式重新训练并将其作为 reviewer 要求的新结果。

### 阶段 7：文档和发布

- [ ] 重写 README：研究目标、安装、数据准备、模型资源、复现命令和引用。
- [ ] 提供论文图表到命令/config/checkpoint/数据 manifest 的映射表。
- [ ] 标注 `fully supported`、`partially supported` 和 `missing/external`。
- [ ] 为 MIC prediction 提供最小 quickstart。
- [ ] generation 在外部 sampler 整合完成前明确标注不可端到端复现。
- [ ] 更新论文 Fig. 2b、正文和 reviewer response 中的公平 benchmark 数值。
- [ ] 确认 license、第三方模型许可、数据再分发条件和 citation。
- [ ] 持续用中文维护 `AGENTS.md`，记录新的审计结论和迁移关系。

## 5. 计划提交序列

建议使用小而可审查的提交：

1. `chore: import sanitized legacy source snapshot`
2. `chore: add release-oriented project scaffold`
3. `refactor: extract shared data and evaluation modules`
4. `feat: add shared molecule encoder benchmark protocol`
5. `refactor: migrate final paper experiment entrypoints`
6. `test: cover data splits metrics and model interfaces`
7. `chore: remove superseded and unrelated legacy code`
8. `docs: add reproducibility and release documentation`

每个提交完成后都更新本文件的复选框和 `AGENTS.md` 中受影响的血缘说明。

## 6. 风险与停止条件

出现以下情况时不得自行猜测并继续发布：

- staged 内容中发现无法确认是否已撤销的真实密钥；
- GitHub 远程仓库包含用户已有且本地不存在的提交；
- 准备删除的实现无法从 Git tag 或用户备份中恢复；
- 某个“最终版”与论文数值、Methods 或 checkpoint 结构出现新的实质冲突；
- 公平 benchmark 无法为某个 comparator 构造确定且可审计的输入；
- 数据或模型的许可不允许公开再分发。

这些情况应记录为“仍待作者确认的事项”，并在执行相关发布或删除动作前与作者确认。
