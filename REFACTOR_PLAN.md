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
│   ├── hierarchical_mic/           # strain/species/phylum 共用 runner + split adapters
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
- [x] 生成待保留、待迁移和待删除的机器可读清单：
  `reproducibility/migration_inventory.yaml`。清单中的删除权限默认关闭，只有替代入口和
  checkpoint 等价验证完成后才允许改变。

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
- [x] 远程 tag `legacy-code-snapshot-2026-07-17` 已通过恢复后的 GitHub HTTPS 凭据推送；`archive/legacy-code-snapshot-2026-07-17` branch 继续作为额外恢复点。
- [x] 从该 tag 创建 `agent/paper-release-refactor` 重构分支，后续清理不直接破坏历史快照。

验收标准：从 tag 可以查看原始源码血缘，但仓库中不存在数据、checkpoint、结果或可用密钥。

执行记录：当前 Codex 工作区将 `.git` 作为只读保护挂载，因此 Git metadata 位于被忽略的 `.git-state/`，并通过 `--git-dir=.git-state --work-tree=.` 操作同一工作树。本地脱敏快照提交为 `a68707c`。最初本机没有可用 `gh`、SSH public key 或 HTTPS credential，因此使用已授权的 GitHub App Git object API 重建远程提交链；上传的 237 个去重 blob 和五个版本 tree 均逐一与本地 SHA 校验一致。远程提交因额外的仓库初始化 parent 而拥有不同 commit SHA，但每个科学代码版本的 tree 与本地完全一致。2026-07-18 已完成 `gh` 2.96.0 网页认证，Git protocol 和 `origin` 均切换为 HTTPS；随后成功 fetch 合并后的 `main` 并推送 annotated tag `legacy-code-snapshot-2026-07-17`。PR #2 已通过 merge commit `24d975c` 合入远程 `main`。

### 阶段 2：建立共享核心模块

- [x] 建立可安装的 `src/` package 和最小依赖定义。
- [ ] 抽取统一的数据 schema、label transform、mask、strain mapping 和 filtering；Fig. 2b 的 19-task schema、共享过滤和 fold 已先行完成。
- [ ] 抽取 genome/text/molecule feature 接口。
- [ ] 抽取 cross-attention/fusion、LoRA、regression/classification head。
- [ ] 抽取指标、ensemble、checkpoint loading、seed 和设备选择逻辑。
- [x] 建立 `configs/model_weights.yaml` 统一登记权重当前位置、SHA-256、消费实验和计划迁移路径；实际权重解析器与集中搬迁仍待实现。
- [ ] 将硬编码路径迁移到 CLI 参数或 YAML 配置。
- [ ] 统一预测输出格式，至少包含 sample ID、fold、label、prediction 和模型元数据。

执行进度（2026-07-19）：hierarchical MIC 路径已完成统一。strain-wise、论文中的
species-wise（11 clusters）和 phylum-wise（3 clusters）共用同一份 Dataset/collate、
precomputed feature loader、cross-attention、prediction heads、四路训练、评估、scheduler、
best-metric selection、checkpoint payload 和 ensemble runner；三个实验只保留 split adapter、
group 名称与输出路径差异。canonical 入口为 `scripts/reproduce/run_hierarchical_mic.py`，
配置为 `configs/hierarchical_mic/legacy_mdlm.yaml`。三个真实数据 dry-run、保存日志计数对照、
CPU 等价测试和 H100 一轮四路训练 smoke 均已通过；被替代的 DP/in-house/SM 和 pooling/eval
复制脚本已删除，完整历史由 `legacy-code-snapshot-2026-07-17` tag 保留。

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

状态：**已实现并通过真实数据验证。** 当前源表 11,401 个 molecule。按各论文脚本的原生输入限制审计后，ChemBERTa-MTR、ChemBERTa-MLM 和 MolFormer 各可处理 10,889 个，PeptideCLM 可处理 11,377 个，两个 DLM 版本各可处理 11,082 个，APEX 可处理 11,321 个；最终公共交集为 10,886 个，五折大小为 2,178、2,177、2,177、2,177、2,177。APEX 输入中的 noncanonical residue 写为 `X`，但继续使用原始 23-token vocabulary；按原 `onehot_encoding` 行为，`X` 留在 index 0。不得修改 APEX 的 AAindex embedding、encoder、checkpoint 或 regression head。

#### 3.2 统一训练与评估协议

- 所有模型严格读取同一 `folds.csv`。
- 保留每个模型原论文实现中的 encoder、prediction head、训练参数和 checkpoint-selection 行为；本轮只统一 molecule IDs 和 folds。
- frozen encoder 与 fine-tuned encoder 必须明确分组，不能在同一表中无说明混比。
- 每个 fold 保存预测和指标；汇总报告 mean ± SD。
- 输出逐模型处理成功率和任何异常，但正式指标必须基于完全相同的 test IDs。

状态：**已完成。** 共享 native-processability 审计、公共 ID 交集、outer fold 校验、MIC label transform、指标实现和严格 ID 对齐的 `.npz` feature-cache 契约已实现并通过测试。曾新增的 10% validation、统一 prediction head 和统一 head runner 超出了 reviewer 要求，已经撤回。APEX adapter 已严格加载完整原 checkpoint，并为 10,886 个公共 molecule 生成及回读 `(10886, 128)` 审计 cache。各原始训练实现的共享 IDs/folds 薄入口和两个 DLM checkpoint 接入已经完成；正式 7-model × 5-fold 训练于 2026-07-18 全部完成，35 个 fold 无缺失且每个模型的 test IDs 均完整覆盖公共集合。训练保持原 200 epochs、batch size 200、Adam、`1e-4`、模型特有 head 和 train/eval mode；只缓存 held-out fold 上确定性的 frozen-backbone eval feature，以避免每个 epoch 重复相同计算。正式结果记录在 `experiments/fig2b_molecule_encoders/results_shared_5fold.md`。

权重审计补充：**已完成 node002 核验。** 本机、node002、W&B 和公开 Hugging Face 权重中均未发现 24-layer/1024 纯 DLM checkpoint，因此当前正式表中的 12-layer DLM-only 与 24-layer joint 只能表述为模型版本 benchmark。随后在 node002 原始 Fangping run 中确认了 12-layer/768 joint `best.ckpt`（step 650032），可与 `best_2.ckpt` 做容量匹配的新五折比较；但二者预训练 learning rate、global batch size 和最佳 step 不完全一致，不能称为严格单变量 objective ablation。该候选尚未运行，不能提前替换正式结果。

#### 3.3 capsule 迁移

- 将 `capsule_fig2/` 的可复用源码迁移到 `src/apexoracle/benchmarks/molecule_encoders/` 和 `experiments/fig2b_molecule_encoders/`。
- 将 Code Ocean 专用入口整理到 `reproducibility/code_ocean/fig2b/`。
- 删除 capsule 内重复源码和历史结果，不保留第二套 canonical 实现。

验收标准：**已满足。** 一条数据准备命令生成共享 manifest/folds；benchmark runner 能选择各 encoder；最终汇总包含全部 35 个 fold 指标和 mean ± SD。

### 阶段 4：迁移论文其余最终实验

#### 4.1 优先迁移的高置信度最终版

- Fig. 1a / Fig. 2c strain-wise、species-wise 和 phylum-wise DLM ensemble：统一入口
  `scripts/reproduce/run_hierarchical_mic.py`，旧 root drivers 由 legacy tag 追溯。
- Fig. 1b 三菌株分类：统一入口
  `scripts/reproduce/run_antibiotic_classification.py`，旧 root drivers 由 legacy tag 追溯。
- AMP/PepLink 最终数据处理血缘。
- ApexOracle-3/12/23 sequence similarity 流程。
- reviewer 的 Evo-2 embedding scaling 分析脚本。

状态（2026-07-19）：三个 hierarchical holdout 和 Fig. 1b 三菌株分类均已完成单一 runner
的行为保持迁移。

- 共享实现位于 `src/apexoracle/{data,features,models,training,evaluation}`；唯一入口为
  `scripts/reproduce/run_hierarchical_mic.py`，实验契约和审计材料位于
  `experiments/hierarchical_mic/`。
- 完整 outer loop 已脱离 root legacy driver。测试覆盖 logits、loss、gradient、Adam 更新、
  epoch-0/逐 epoch evaluation、prediction 分区、loader `zip_longest`、scheduler、strict
  best-metric selection、七键 checkpoint payload 和统一 runner 的 H100 一轮集成 smoke。
- 21 个历史 checkpoint 的 `3 × 7` 网格和实际消费的 fusion/head contract 已全部扫描。
  group 0/2 的 14 个文件只保存 fusion/head；group 1 的 7 个文件额外保存一个名称错误的
  `ChemBERTa_state_dict`，其结构实际是 12-layer/768 MDLM backbone。三个 group 的 optimizer
  均不包含该 backbone 参数；是否在 group 1 的 forward 中在线使用仍待旧源码确认。
- 历史 split 同时依赖无序 `set` 和原地 taxonomy-alias list mutation；日志没有记录三个独立
  Python 进程的 `PYTHONHASHSEED`。本机与 node002 的 driver、核心数据和文件名清单已经核验
  一致，因此当前只能提供 `PYTHONHASHSEED=0` 的确定性候选 manifest，不能声称恢复了 2025 年
  checkpoint 的精确 strain membership。过渡入口要求显式确认这一限制。
- group 0 / ensemble 0 已在 H100 上完成两次独立固定批次严格加载与推理，结果一致；其余
  checkpoint 已完成结构扫描，但剩余 20 个 SHA-256 和逐文件推理仍待补齐。

因此 hierarchical MIC 项当前状态是 `unified runner complete / legacy duplicates removed`。

Fig. 1b 三菌株分类也已完成统一迁移。canonical 入口为
`scripts/reproduce/run_antibiotic_classification.py`，通过 `--mode strict-zero-shot`、
`fine-tune` 或 `molecule-only` 显式选择三种旧协议。共享数据/fold adapter、feature loader、
fusion/head、四路训练、metrics、best-AUROC selection、checkpoint schema 和 ensemble runner
已经抽取；旧 root drivers 和两份机器专用 launcher 已删除并由 legacy tag 保留。

行为验证包括：真实数据 dry-run 与旧日志计数逐项一致；11 项 CPU 测试；H100 上四路合成
batch 的一轮训练/评估 smoke；以及 strict zero-shot group 0 / ensemble 0 的 2,335 条真实
checkpoint logit 与 capsule 在 batch size 70 下逐值完全一致。30 个 strict checkpoint 和
150 个 molecule-only checkpoint 网格完整；fine-tune 仅存 77/150 个 checkpoint，因此仍按
证据不完整处理。详见 `experiments/fig1b_antibiotic_classification/`。

Sequence similarity 已于 2026-07-19 完成迁移。canonical 入口为
`scripts/reproduce/run_sequence_similarity.py`，配置为
`configs/sequence_similarity/paper_leads.yaml`。训练 cache 可从当前源数据逐字节重建；
ApexOracle-3/23 的四份核心历史 CSV 也与 canonical 全量重算逐字节相同。三条 lead 的
最大 PID 均与论文一致。历史 cache 的 training sequence 使用 uppercase normalization；
直接保留当前 DBAASP JSON 大小写会改变 ApexOracle-23 结果，因此只作为 sensitivity，不能
替代正式 paper contract。ApexOracle-12 的旧 full CSV 未保存，且其 35.7% 有四个 complete
ties；论文展示 15510，而稳定输入顺序选择 9800，数值不变。旧 `DataPrepare/get_similarity`
tracked drivers/manifests 已删除并由 legacy tag 保留。详见
`experiments/sequence_similarity/`。

当前下一项高置信度迁移改为 AMP/PepLink 最终数据处理血缘；fine-tune 完整正式结果、
synergy CV 和 modality ablation 继续保留证据边界。

PepLink 外部依赖边界已于 2026-07-19 完成。作者维护的独立仓库
`DragonDescentZerotsu/PepLink` 已发布 PyPI `PepLink==0.1.1`，tag `v0.1.1` 和 commit
`cec2a02427766e4ba95806924801af31bdcc9939`。ApexOracle 不使用 submodule，也不复制其
2,300 行 chemistry core；仅保留 optional dependency、公开 API adapter 和版本/data SHA
manifest。独立 PepLink 测试为 22 passed。179 条历史 structure correction 中 177 条输出
逐字符串相同，另 2 条是 v0.1.1 明确移除 legacy 游离 fragment，全部 179 条均为 fragment
parent equivalent。论文复现继续消费 frozen paper CSV；新数据使用 v0.1.1 normalization。
旧 `DataPrepare/aa_seq_to_smiles.py` 暂不删除，因为尚有未迁移的 MIC/synergy data driver
import 它。下一子阶段为 MIC parsing、in-house merge 与 SELFIES/token filtering。

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

执行记录（2026-07-19）：hierarchical MIC 统一迁移新增了 legacy/shared 逐项等价测试，覆盖四种
collate、Dataset lookup、MIC label transform、feature filename parsing/loading、strain mapping、
cross-attention、regression head、split mutation 语义和两类 checkpoint 顶层 payload。代表性
9.08 GB checkpoint 已通过 H100 固定批次验证。后续单批次训练测试又覆盖四种 modality/task
组合，并逐参数验证 gradient、历史 clipping 目标和 Adam step 后的参数一致性；H100 上另以
有限合成 GradScaler scale 验证 CUDA autocast 路径逐位一致。正式 driver 的默认动态 scaler
保持不变。这里仅表示
strain/species/phylum 的共享子路径通过本批验收，
不等同于阶段 6 的全仓库验证已经完成。

同日 Fig. 1b 三菌株分类新增 11 项 CPU 测试和 1 项 H100 集成测试，覆盖目标 frame、
512-token 后 KFold、molecule ID 导出、full-fusion/molecule-only forward、optimizer step、
module mode、AUROC/AUPRC、best tracker 的 AUPRC 保存顺序、两种 checkpoint schema 和文件名。
strict group 0 / ensemble 0 的真实 checkpoint 又与 capsule 在 batch size 70 下完成 2,335 条
logit 逐值一致验证。全仓库当前为 50 passed / 3 skipped；三个 skipped 均为沙箱内不可见 CUDA
时跳过的测试，其中本批新增 CUDA test 已在宿主 H100 单独通过。阶段 6 仍未整体完成，因为
其余未迁移实验尚无对应 smoke/checkpoint 验证。

### 阶段 7：文档和发布

- [ ] 重写 README：研究目标、安装、数据准备、模型资源、复现命令和引用。
- [ ] 提供论文图表到命令/config/checkpoint/数据 manifest 的映射表。
- [ ] 标注 `fully supported`、`partially supported` 和 `missing/external`。
- [ ] 为 MIC prediction 提供最小 quickstart。
- [ ] generation 在外部 sampler 整合完成前明确标注不可端到端复现。
- [x] Fig. 2b 当前正式修订已完成：正文、图注和 reviewer response 已更新公平 benchmark 数值；完整 `Fig2_2.pdf` 已换入 10,886 个共享分子的七模型结果和五折 sample s.d. error bars，并经渲染核对；最新 TeX 已完整编译为 28 页。当前图文对应 24-layer joint 正式结果。12-layer joint 容量匹配实验属于后续核验；若采用其结果，仍须同步更新图、正文、回复信和相对提升。
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
