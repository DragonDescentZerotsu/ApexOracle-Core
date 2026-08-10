# ApexOracle 统一公开仓库发布计划

> 决策日期：2026-08-09
> 状态：架构已由作者确认并冻结；MDLM 与 Generation clean modules 已发布并验收，super-repo/submodule 尚未创建
> Canonical 上位计划：`REFACTOR_PLAN.md`

## 1. 目标与固定架构

最终发布采用一个轻量公开 super-repo 加五个固定 commit 的 Git submodule。各模块保留当前内部
代码结构、依赖和运行环境，不把 MDLM、Evo-2 或 guided-generation 源码重排进
`src/apexoracle/`。Reviewer-facing 的统一性由根 README、module lock、资产 manifest、bootstrap
入口和两个端到端 quickstart 提供。

```text
ApexOracle/
├── modules/
│   ├── core/                       # ApexOracle-Core；来自当前 Synergy
│   ├── dlm_pretrain/               # ApexOracle-DLM-Pretraining；合作者预训练 producer
│   ├── mdlm/                       # ApexOracle-MDLM；downstream embedding/guidance support
│   ├── evo2/                       # ApexOracle-Evo2 clean fork
│   └── generation/                 # ApexOracle-Generation clean fork
├── quickstarts/
│   ├── predict_mic.py
│   └── guided_generation.py
├── environments/
│   ├── prediction.yml
│   ├── generation.yml
│   ├── dlm_pretraining.yml
│   └── evo2.yml
├── manifests/
│   ├── modules.lock.yaml
│   ├── model_weights.yaml
│   └── data_assets.yaml
├── scripts/
│   ├── bootstrap.sh
│   ├── setup_prediction.sh
│   └── setup_generation.sh
├── .gitmodules
├── README.md
├── LICENSE
└── CITATION.cff
```

固定的目标 remotes 为：

| 角色 | 目标 remote | 来源与职责 |
| --- | --- | --- |
| 统一入口 | `DragonDescentZerotsu/ApexOracle` | 轻量 super-repo；README、quickstarts、环境和 manifests |
| Core | `DragonDescentZerotsu/ApexOracle-Core` | 当前 `Synergy` 的 canonical data/fusion/heads/runners/audits |
| DLM Pretraining | `DragonDescentZerotsu/ApexOracle-DLM-Pretraining` | 合作者的 DLM + 209-descriptor MTR 预训练 producer |
| Downstream MDLM | `DragonDescentZerotsu/ApexOracle-MDLM` | checkpoint loading、molecule embedding、guidance heads 和 candidate scoring |
| Evo-2 | `DragonDescentZerotsu/ApexOracle-Evo2` | 官方 Evo-2 clean fork 加 ApexOracle genome extraction 入口 |
| Generation | `DragonDescentZerotsu/ApexOracle-Generation` | diffusion、MIC/peptide guidance、remasking 和输出 |
| PepLink | `DragonDescentZerotsu/PepLink` / `PepLink==0.1.2` | 保持独立版本化依赖，不作为 submodule |

当前公开 `DragonDescentZerotsu/ApexOracle` 的 legacy monorepo 不继续累加。发布切换时先将其改名为
`ApexOracle-Legacy` 并归档，再让新 super-repo 使用论文已经公开的 canonical URL。不得通过把
dirty checkout、数据或权重复制进旧 history 的方式实现统一发布。

## 2. 模块边界

### 2.1 `modules/core`

- 保留当前 `Synergy` 的 `src/apexoracle/`、`scripts/`、`configs/`、`experiments/` 和 tests 布局。
- 不为适配 super-repo 重写已验证的 fusion、head、runner 或 checkpoint schema。
- 只新增公共 inference/quickstart 所需的稳定薄入口，并继续通过现有 tests 保护行为。
- 在公开前完成 private data、绝对路径、ignored artifacts、许可和 staged-file 白名单审计。

### 2.2 `modules/dlm_pretrain`

- 从当前 public legacy `ApexOracle/DLM_pretrain/` 提取 clean history，不从本地 downstream `/mdlm`
  反向拼装预训练源码。
- 只负责 DLM 与 209-descriptor MTR 联合预训练、数据 schema、训练配置和 checkpoint 生成。
- 发布前必须消除数据/cache/statistics 绝对路径，并修复 README `model=small` 与 hard-coded
  1024-dimensional regression head 的配置不一致。
- clean commit 必须通过 synthetic-batch train/save/load smoke；在 checkpoint commit 血缘未闭合前，
  只称为 verified historical producer family。

### 2.3 `modules/mdlm`

- 作为 downstream 模块保留必要的 MDLM runtime，不再承担合作者预训练代码的 canonical 记录。
- 从 dirty checkout 中只整理 checkpoint loading、molecule embedding、经血缘确认的 MIC/classifier
  guidance heads 和 candidate scoring；排除 checkpoint、Data、W&B、cache、outputs 和 temp scripts。
- Core 已接管的 hierarchical MIC/synergy CV 不在这里再维护第二份 canonical runner；generation
  sampler 继续只属于 Generation。
- clean release commit 必须能加载正式 tokenizer/checkpoint，并完成固定 SELFIES embedding 和
  guidance-head load smoke。
- 发布必须显式推向 `custom`/`ApexOracle-MDLM` remote，不得误推上游 `origin`。

两个 MDLM 相关模块的逐文件功能分类、证据边界和验收标准见
`docs/MDLM_MODULE_SPLIT_AUDIT.md`。

### 2.4 `modules/evo2`

- 以官方 Evo-2 的许可兼容 clean commit 为基线，保留其内部依赖和 nested submodule 结构。
- 仅增加参数化的 ApexOracle genome extraction 入口、window/layer/pooling 配置和小规模验证。
- 当前 dirty checkout 和未跟踪 `ATCC/get_40b_emb*.py` 不得直接成为 submodule commit。
- 在精确历史 producer 尚未恢复时，入口标为 verified reference implementation，不冒充
  byte-exact 2025 producer。

### 2.5 `modules/generation`

- 保留 discrete-diffusion-guidance 的上游组织和 Hydra 调用方式。
- 先冻结本机、node002 和保存的 resolved configs，再确定论文主调用链的 canonical source。
- `v1.0` 默认只发布论文 MIC/peptide guidance 与 remasking 路径；论文后 synergy guidance 单列
  experimental，不进入默认 quickstart。
- clean commit 必须参数化数据、权重和输出根目录，禁止覆盖历史 `outputs/`。
- 当前上游 remote 不接收 ApexOracle patch；必须发布到自有 `ApexOracle-Generation` remote。
- 已发布的默认 `main` 固定候选为 `de6c1e590c25b2ce36b4ce5c42c5a4fa0dcc7705`；annotated recovery tag
  `legacy-code-snapshot-2026-08-10` 指向 source-only snapshot `2368c25ce831c187e5b2699b85a6ae1a4cdca31a`。

## 3. 环境与资产策略

- 不强求五个模块共享一个 Python environment。根仓库维护 `prediction`、`generation`、
  `dlm-pretraining` 和 `evo2` 四个 profile，分别调用模块自己的锁定环境。
- checkpoint、embedding、训练数据、raw attempts 和大缓存不进入任何 Git repository。
- `manifests/model_weights.yaml` 与 `manifests/data_assets.yaml` 必须为每个公共资产登记稳定 URI、
  revision、size、SHA-256、许可/再分发状态和消费模块。
- 当前已验收的首个固定模型资产为 `Kiria-Nozan/ApexOracle` revision
  `77694f08c1d0664fdb24c5a7bab130c8a3bc2eda`，weight SHA-256
  `b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44`，model-card license 为 MIT；
  MDLM runtime 与 IBM tokenizer 的 Apache-2.0 attribution 由模型仓库内 notice 保留。未来
  `model_weights.yaml` 必须固定该 revision，不能使用浮动 `main` 或中间失败 revision `b16024b`。
- root bootstrap 只下载/校验资产并调用 submodule；不得复制一份模型实现到 super-repo。
- 所有 `.gitmodules` 条目固定到 super-repo commit 记录的具体 submodule SHA，不跟踪浮动 branch。

## 4. Reviewer-facing 使用契约

README 第一屏必须同时提供 recursive clone 和普通 clone 的补救命令：

```bash
git clone --recurse-submodules \
  https://github.com/DragonDescentZerotsu/ApexOracle.git
cd ApexOracle

# 如果已经执行普通 clone：
git submodule update --init --recursive
```

最低端到端入口固定为：

```bash
./scripts/bootstrap.sh prediction
python quickstarts/predict_mic.py \
  --molecule examples/molecules/example.selfies \
  --strain BAA-3170

./scripts/bootstrap.sh generation
python quickstarts/guided_generation.py \
  --strain BAA-3170 \
  --num-samples 10
```

根 quickstart 只负责参数和资产联结；核心计算继续由相应 submodule 执行。Generation 同时提供快速
smoke preset 与完整 256-step paper preset，并明确 smoke 输出不是论文结果。

GitHub 自动生成的 source ZIP 不含 submodule 内容，因此每个正式 release 必须额外由 CI 生成
`ApexOracle-v<version>-source-full.tar.gz`：展开全部固定 submodule、附带
`manifests/modules.lock.yaml`，并核验 archive 中的 module SHA 与 super-repo 完全一致。

## 5. 分阶段执行计划

### R0：冻结来源与恢复点

状态：**部分完成。** Downstream MDLM 已完成 source-only snapshot/tag、tracked ledger 和 public Hub
release provenance；Core/Evo-2/Generation/DLM-pretraining 的统一 R0 manifest 仍待收口。

- 为 `Synergy`、public legacy `DLM_pretrain/`、本地 `mdlm`、`evo2`、
  `discrete-diffusion-guidance` 建立 tracked/modified/untracked
  source inventory 和 SHA-256 manifest。
- 将文件分类为 paper canonical、reviewer canonical、post-paper experimental、upstream、debug、
  data/weight/output/cache。
- 为所有 dirty checkout 建立非破坏性恢复点；不得 reset、clean 或用上游覆盖本地实现。
- 先收口或明确排除 `Synergy` 当前未提交 reviewer 工作，避免其混入 Core release baseline。

验收：任一迁移文件都能追溯到来源 checkout、source hash、科学角色和目标 module。

### R1：准备五个 clean module commits

状态：**进行中。** MDLM 与 Generation 两个 module candidates 已进入各自 public 默认分支并通过远端
fresh-clone 验收；Core、DLM-Pretraining 与 Evo-2 三个模块待执行。

- `ApexOracle-Core`：以当前 Synergy canonical 代码为基础完成公开边界审计。
- `ApexOracle-DLM-Pretraining`：形成 portable pretraining source commit 和 synthetic train/save/load smoke。
- `ApexOracle-MDLM`：形成 downstream source-only release commit、embedding 和 guidance-head smoke。
- `ApexOracle-Evo2`：形成 clean fork commit、通用 extraction CLI 和小规模 tensor contract test。
- `ApexOracle-Generation`：形成 paper-path clean commit、参数化配置和小 batch GPU smoke。

Generation 已完成：paper preset 固定 256-step、15/15 guidance 与 remasking；1-sample GPU smoke
token-level complete 1 条但结构过滤后 0 row，只作为 runtime contract；远端 fresh clone 的 source audit、
14 tests 和 BAA-3170/3197 resolved-config dry-run 通过。13 个硬编码 launchers 已由通用 CSV grid 取代，
删除项均可由 recovery tag 恢复。仓库保留上游 Apache-2.0 `LICENSE` 与 `NOTICE`，未把外部资产纳入 Git。

验收：每个 module 独立 clone 后可按自身 README 安装，smoke test 通过且不需要作者机器绝对路径。

### R2：建立 super-repo 骨架

状态：待执行。

- 创建轻量 clean-history super-repo，加入五个 submodule 的固定 SHA。
- 新增 `modules.lock.yaml`、资产 manifests、环境 profiles、bootstrap 和统一 README。
- CI 验证 recursive clone、module SHA、license/NOTICE、secret、大文件和 broken link。

验收：全新 checkout 可完整初始化所有 submodule；普通 clone 的补救命令也通过。

### R3：MIC prediction quickstart

状态：待执行。

- 固定一个可公开的 example molecule、known strain、必要 embeddings 和正式 checkpoint manifest。
- 建立 root wrapper，调用 Core 与 MDLM canonical inference，不复制实现。
- 输出至少包含 molecule ID/input、strain ID、checkpoint ID、prediction、unit 和 provenance。
- 在 fresh environment 中记录安装时间、峰值内存/显存和推理 runtime。

验收：一条 documented command 从资产下载到 MIC prediction 成功，输出与模块级 reference 一致。

### R4：Guided generation quickstart

状态：待执行；为关键路径。

- 固定论文 BAA-3170/BAA-3197 中至少一个公开 target 及其 genome/text assets。
- root wrapper 调用 Generation、MDLM 与 Core 的固定 commits 和权重。
- 同时验证 smoke preset 与 paper preset 的参数解析；至少完成一个小 batch 的 SELFIES/RDKit contract。
- 记录 GPU 型号、显存、runtime、seed、完整 resolved config 和输出 manifest。

验收：fresh GPU environment 可产生完整 raw attempts 和派生 valid outputs；历史 outputs 不被覆盖。

### R5：数据、许可和完整 source release

状态：待执行。

- 发布 reviewer 承诺的公开 model-ready tables、strain mappings、splits 和 strain-description texts，
  或对不可再分发/私有部分给出明确 manifest、获取方式与排除说明。
- 完成第三方 source、model、data 和 checkpoint 的许可/NOTICE 审计。
- 生成展开 submodules 的 full-source archive，执行凭据和二进制膨胀检查。
- 用一台不依赖作者现有 cache 的环境完成 release-candidate fresh-clone QA。

验收：代码、数据和权重三类入口互相引用一致，所有公开文件可由 manifest 验证。

### R6：Canonical URL 切换与正式发布

状态：待执行。

- 将旧 public `ApexOracle` 改名归档为 `ApexOracle-Legacy`；不复用其肥大 history。
- 将新 super-repo 发布到 `DragonDescentZerotsu/ApexOracle`。
- 发布 tag、GitHub Release、full-source archive，并同步 Hugging Face/Zenodo cards 与论文链接。
- 只有两个端到端 quickstart、资产下载和 fresh-clone QA 完成后，才能把 reviewer response 的相关
  future tense 改为完成时。

验收：论文 canonical URL 指向新 super-repo；release tag 固定五个 module SHA 和全部资产版本。

## 6. 禁止项与变更控制

- 不把 dirty 本地目录或 legacy public repo 内的目录直接作为 submodule。
- 不把 checkpoint、embedding、dataset、raw output、W&B、cache 或编译产物提交到 Git。
- 不为统一目录而批量改写 module 内 import、Hydra config 或已验证 scientific protocol。
- 不在未验证等价前删除历史 source；恢复点与 source manifest 必须先完成。
- 不让 super-repo quickstart 依赖作者机器绝对路径、隐式 conda env 或未登记的手工复制。
- 本架构已冻结。改变 module 数量、目标 remote、submodule 策略或 canonical URL 切换方式，必须由
  作者再次确认，并同步更新 `REFACTOR_PLAN.md`、根 `AGENTS.md`、
  `docs/COMPUTE_AND_ASSET_MAP.md` 和本文档。

## 7. 当前事实、判断与仍待执行事项

### 已由 Git/文件系统验证的事实

- 当前 `Synergy` 为 private remote，已包含主要 Core 重构和 reviewer-facing code，但工作树仍有
  未提交 reviewer 工作。
- `ApexOracle-MDLM` public remote 已存在；source-only tag `legacy-code-snapshot-2026-08-09` 已 push，clean
  release 已进入默认 `master`，当前固定候选为 `c9d17c7f6f091234aaaebf5f08dbe23542f980c1`。Checkpoint/embedding I/O、shared
  guidance heads、candidate MIC/synergy scoring、interpretability、paper Fig. 3a、通用 peptide/small-molecule
  screens、molecule embedding producer、四个 peptide-classifier profiles 与五个 MIC-guidance profiles 已迁移；
  全仓 118 tests 与 14 项跨仓库 source contracts 通过；五个正式 guidance checkpoints 通过 schema/inactive
  cls-head strict load，Generation regression 正式 GPU parity exact。11 个 Core-owned hierarchical drivers 已
  handoff 后删除；两个 chemistry root utilities 已迁为通用 converter/catalogue matcher，并通过 11,401-row
  byte parity 与 5,887,458-row full-scan parity。
- Hugging Face `Kiria-Nozan/ApexOracle` 已从 72-file legacy tree 清为 18-file allowlist；正式 revision
  `77694f08c1d0664fdb24c5a7bab130c8a3bc2eda` 已从空 cache 下载，通过 MIT metadata、manifest/hash、
  `strict=True` load 和 integer-mask padded GPU inference。权重内容不变。旧 tracked HF duplicates 和三个
  classifier root trainers、六个 MIC-guidance root trainers、11 个 hierarchical duplicates、两个 chemistry
  utilities 与三个 synergy-guidance producers 已清除，source archive fresh install/import/CLI smoke 已通过；
  两个 synergy producer profiles 的正式 encoder/candidate GPU parity 与 checkpoint schema 已通过。远端
  shallow-clone wheel/install/import/CLI、118 tests、显式资产 20 checks 和 recovery-tag fetch 均通过。MDLM
  module-level source candidate 已就绪；Generation sampler integration 已完成，Core compatibility bridge 已删除。
- 合作者的 joint DLM+MTR 预训练源码已位于 public legacy `ApexOracle/DLM_pretrain/`，但仍含绝对
  路径和 small/medium config 不一致，尚不是 portable release。
- 本地 Evo-2 checkout dirty，且当前 commit 不能证明是 567 个 frozen tensors 的精确 producer。
- `ApexOracle-Generation` public remote 已创建，默认 `main` 为 `de6c1e5`；source-only recovery tag 已推送。
  从 GitHub shallow clone 后 release audit、14 tests 和双 strain dry-run 通过；外部历史 outputs 原地 ignored。
- 当前 public `ApexOracle` legacy history 已复制外部代码和大资产，不适合继续累加。

### 根据现有证据作出的判断

- Generation 的历史 source/config 冻结和 clean commit 已关闭；下一关键路径转为 Evo-2 clean fork。
- 采用 submodule 比重排模块内部代码更能降低科学行为变化和依赖冲突风险。

### 仍待执行而非待架构确认的事项

- 创建剩余 Core、DLM-Pretraining、Evo-2 remotes，并决定具体 visibility 切换时间。
- 完成剩余三个 clean module commits 及各自 smoke tests。
- 完成数据/模型再分发许可审计和稳定下载 URI。
- 完成两个端到端 quickstart、full-source archive 与 fresh-clone QA。

## 8. 从 2026-08-10 开始的下一阶段固定顺序

1. **MDLM source release——完成。** 默认 `master` 固定候选为 `c9d17c7`；source/HF asset/fresh-clone、
   guidance/scoring 与 Generation integration 已通过，legacy tag 保留。
2. **Generation clean fork——完成。** 默认 `main` 固定候选为 `de6c1e5`；paper preset、通用 grid、GPU smoke、
   remote fresh-clone 和 recovery tag 均通过，未推送上游 `kuleshov-group`。
3. **Evo-2 clean fork（下一关键路径）。** 在官方许可兼容基线上迁移通用 genome extraction CLI，固定 window/layer/pooling
   contract；以已有 567 tensor lineage 和小规模 shape/value test验收，不声称无法证明的历史 byte-exact producer。
4. **DLM-pretraining producer 与 Core 收口。** 合作者 pretraining 原则上原样归档并先做 synthetic
   train/save/load；只有 blocking portability failure 才做最小路径/文档/config 修补，不改变模型结构、
   objective 或训练行为。并行完成 Core 的 public-data/secret/license/fresh inference 审计。
5. **最后创建 super-repo。** 只有五个 module candidate SHA 均冻结后才创建 `.gitmodules`、
   `modules.lock.yaml`、资产 manifests 和两个 quickstarts；随后做 full-source archive、fresh-clone QA 和 canonical
   URL 切换。不得为了提前展示目录而先加入浮动 branch submodule。
