# Fig. 2b：相同数据与相同 fold 的 molecule encoder benchmark

本目录提供 molecular-representation benchmark：所有 encoder 使用原生实现能够处理的 DBAASP ID 交集，并共享一份按 molecule ID 预定义的五折划分。

本实验使用的权重身份、当前位置、SHA-256 和计划迁移路径统一登记在 [`configs/model_weights.yaml`](../../configs/model_weights.yaml)；维护说明见 [`MODEL_WEIGHTS.md`](../../MODEL_WEIGHTS.md)。

该 benchmark 使用共同 ID 交集和共享的 molecule-level 五折划分，没有独立 validation split，也不采用 scaffold split。各模型沿用原实现的 prediction head、训练模式和 checkpoint selection；模型间统一的是样本集合与 folds。

## 公共数据

原始表包含 11,401 个 molecule。按原脚本的 native preprocessing 规则：

- ChemBERTa-MTR：10,889；
- ChemBERTa-MLM：10,889；
- MolFormer：10,889；
- PeptideCLM：11,377；
- DLM MTR+DLM / DLM-only：11,082；
- 按固定 residue projection 规则、使用原模型的 APEX：11,321；
- 全部 encoder 的最终共同交集：**10,886**。

五个共享 fold 的大小是 2,178、2,177、2,177、2,177、2,177。进入 fold 后任何 encoder 都不得再删除 ID。

## APEX 保持原样

APEX 输入序列中，noncanonical residue 写成 `X`，D-residue 只保留 residue identity，cyclic、bond 和 multichain topology 使用确定性线性 residue 顺序。这只是在原模型可接受的 residue string 层面构造输入。

APEX 模型、23-token vocabulary 和 AAindex embedding 不做修改。原 APEX 没有 `X` token，因此
`X` 按已冻结在 `apex_adapter.py::legacy_onehot_encoding` 的原行为留在 index 0。不得增加
index 23、平均 AAindex embedding 或更换原 `512→256` regression head。

2026-07-19 已把原 `compare_APEX` 动态 import 的支持代码迁入
`src/apexoracle/benchmarks/molecule_encoders/{apex_model,apex_adapter,legacy_training}.py`。真实
checkpoint 以 `strict=True` 加载，固定四序列的 legacy/canonical `(4,128)` feature SHA-256
均为 `dd1ac1a9f359f60b55a83910cdd198b5aeca00788b2d5a8c6aa691d956643cb3`。三个 Fig. 2b
runner 和 capsule resource builder 已切换到 canonical module；详细证据见
`apex_legacy_migration_audit.json`。2026-07-20，checkpoint 已迁至统一 `weights/` 根并通过
`fig2b_apex_encoder` manifest ID 加载；AAindex 位于
`resources/reference/apex/aaindex1.csv`，其上游许可和安装说明见同目录 README。其余历史 checkpoint
与 W&B 记录完整迁入项目外 legacy archive，没有删除。

## 数据准备

先审计各原生 tokenizer/输入路径的可处理 ID：

```bash
python scripts/prepare_data/audit_fig2b_encoder_eligibility.py
```

再构建 10,886 个共同 ID 和唯一五折划分：

```bash
python scripts/prepare_data/build_fig2b_shared_dataset.py
```

默认产物位于`DataPrepare/Data/fig2b_shared_v1/`：

- `common_molecule_ids.csv`；
- `folds.csv`；
- `shared_molecules.csv`；
- `exclusions.csv`；
- `apex_projection_audit.csv`；
- `dataset_manifest.json`。

## 训练协议边界

- 不新增 10% inner validation；
- 不把所有 comparator 强制改成相同 head；
- 不把原在线 backbone 行为替换成统一 eval-mode feature cache；
- 不改变 APEX vocabulary、embedding、encoder 或 head；
- 唯一强制共享的是 10,886 个 molecule ID 和五个 fold。

旧脚本确实会在 held-out fold 上逐 epoch 评估并选择 best checkpoint，APEX validation 时还保持 head dropout。该协议存在 test-set reuse 的局限。采用独立 train/validation/test 划分的实验属于不同协议，应与这里的结果分别报告。

## 当前状态

- native-processability audit、10,886-ID intersection、共享五折和 manifest：已实现并验证；
- APEX adapter：已恢复原 23-token 模型并严格加载完整原 checkpoint；
- feature cache 代码只用于输入/encoder 审计，不作为 paper-compatible 正式训练入口；
- 各原始训练脚本读取共同 IDs/folds 的薄 wrapper：已实现并通过全量单 epoch smoke test；
- DLM-only 与 MTR+DLM 的精确 checkpoint 接入：已实现并验证 checkpoint 结构；
- 正式五折结果：7 个模型 × 5 folds 已于 2026-07-18 全部完成并通过完整性校验；结果见 [`results_shared_5fold.md`](results_shared_5fold.md)。

## Paper-compatible 正式训练入口

2026-07-17 已增加两个正式训练入口：

- `scripts/reproduce_fig2b_baselines_online_5fold.py`：ChemBERTa-MTR、ChemBERTa-MLM、MolFormer、PeptideCLM 和原版 APEX；
- `scripts/reproduce/run_fig2b_shared_mdlm_online.py`：12-layer DLM-only 与 24-layer MTR+DLM，必须在 `mdlm` conda 环境中运行。

两者都读取 `fig2b_shared_v1/folds.csv`，保持 200 epochs、batch size 200、Adam、`1e-4` learning rate、原 prediction head、frozen backbone 的 train/eval mode 和 held-out-fold checkpoint selection。为了减少不改变结果的重复计算，held-out fold 上 frozen backbone 的 `eval()` feature 每个 fold 只计算一次；每个 epoch 仍重新运行 head，APEX head dropout 在 held-out selection 时仍按原脚本保持开启。

正式四卡任务由 `scripts/reproduce/run_fig2b_gpu_queue.py` 调度，完成后由 `scripts/reproduce/summarize_fig2b_shared_results.py` 生成逐 fold 指标和新旧结果差异。当前运行输出位于`results/fig2b_shared_original_protocol/`。

MolFormer 明确固定为本地已有权重对应的历史 revision `7b12d946c181a37f6012b9dc3b002275de070314`。当前 Hugging Face `main` revision 引入了本环境 `transformers` 版本不存在的 `masking_utils`，而该历史 revision 可离线严格加载原 `MolformerModel` 权重；这个兼容性固定没有改变模型结构、权重或训练超参数。
