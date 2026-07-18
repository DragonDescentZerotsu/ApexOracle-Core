# Fig. 2b：相同数据与相同 fold 的 molecule encoder benchmark

本目录对应 reviewer 对 molecular-representation benchmark 的具体要求：所有 encoder 使用原生实现能够处理的 DBAASP ID 交集，并共享一份按 molecule ID 预定义的五折划分。

Reviewer 在一般性的 split 评论中提到 random split 不如 scaffold split 严格，但没有明确要求把这个 molecular-representation benchmark 改为 scaffold split；针对该 benchmark 的具体问题和回复承诺是共同 ID 交集加共享五折，也没有新增 validation split。新版正式 benchmark 只改变两件事：公共 molecule IDs 和公共 folds；APEX、各语言模型、模型特有 prediction head 以及原训练/选择行为均应保持原论文实现并如实记录。

## 公共数据

原始表包含 11,401 个 molecule。按原脚本的 native preprocessing 规则：

- ChemBERTa-MTR：10,889；
- ChemBERTa-MLM：10,889；
- MolFormer：10,889；
- PeptideCLM：11,377；
- DLM MTR+DLM / DLM-only：11,082；
- 按用户确认规则投影、但使用原模型的 APEX：11,321；
- 全部 encoder 的最终共同交集：**10,886**。

五个共享 fold 的大小是 2,178、2,177、2,177、2,177、2,177。进入 fold 后任何 encoder 都不得再删除 ID。

## APEX 保持原样

APEX 输入序列中，noncanonical residue 写成 `X`，D-residue 只保留 residue identity，cyclic、bond 和 multichain topology 使用确定性线性 residue 顺序。这只是在原模型可接受的 residue string 层面构造输入。

APEX 模型、23-token vocabulary 和 AAindex embedding 不做修改。原 APEX 没有 `X` token，因此 `X` 按 `compare_APEX/utils.py::onehot_encoding` 的原行为留在 index 0。不得增加 index 23、平均 AAindex embedding 或更换原 `512→256` regression head。

## 数据准备

先审计各原生 tokenizer/输入路径的可处理 ID：

```bash
python scripts/prepare_data/audit_fig2b_encoder_eligibility.py
```

再构建 10,886 个共同 ID 和唯一五折划分：

```bash
python scripts/prepare_data/build_fig2b_shared_dataset.py
```

默认产物位于被 Git 忽略的 `DataPrepare/Data/fig2b_shared_v1/`：

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

旧脚本确实会在 held-out fold 上逐 epoch 评估并选择 best checkpoint，APEX validation 时还保持 head dropout。这个做法存在 test-set reuse 的局限，但 reviewer 本轮并未要求改变它。正式复现应保持并披露该行为；如果以后增加严格 train/validation/test 版本，必须单独标为 sensitivity analysis，不能替代 paper-compatible benchmark。

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

正式四卡任务由 `scripts/reproduce/run_fig2b_gpu_queue.py` 调度，完成后由 `scripts/reproduce/summarize_fig2b_shared_results.py` 生成逐 fold 指标和新旧结果差异。当前运行输出位于被 Git 忽略的 `results/fig2b_shared_original_protocol/`。

MolFormer 明确固定为本地已有权重对应的历史 revision `7b12d946c181a37f6012b9dc3b002275de070314`。当前 Hugging Face `main` revision 引入了本环境 `transformers` 版本不存在的 `masking_utils`，而该历史 revision 可离线严格加载原 `MolformerModel` 权重；这个兼容性固定没有改变模型结构、权重或训练超参数。

GPU 在宿主机上正常：4 张 NVIDIA H100、driver 580.159.03。Codex 文件沙箱隐藏 `/dev/nvidia*`，所以在沙箱内运行 `nvidia-smi` 会误报无法连接；GPU 命令需要按项目约定在沙箱外执行。
