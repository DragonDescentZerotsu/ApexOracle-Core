# Fig. 1b 三菌株抗生素分类重构

本目录记录 Fig. 1b 三菌株分类代码家族的行为保持迁移。唯一入口为：

```bash
python scripts/reproduce/run_antibiotic_classification.py \
  --mode strict-zero-shot \
  --test-group 0 \
  --dry-run
```

`--test-group` 的固定顺序是：

0. `#004`（E. coli BW25113，genome + text）；
1. `17978`（A. baumannii ATCC 17978，genome + text）；
2. `Staphylococcus aureus RN4220`（text-only）。

## 三种模式

- `strict-zero-shot`：目标菌株全部样本只用于逐 epoch 评估和 checkpoint 选择，不进入训练；不接受 `--fold`。
- `fine-tune`：目标菌株在经过 512-token 过滤后执行 `KFold(n_splits=5, shuffle=True, random_state=42)`；必须指定 `--fold 0..4`。
- `molecule-only`：使用同一目标 KFold，但只训练 `768→384→128→1` molecular classification head；这是旧文件名中的 `wo_SAND` 对照，不是 strict zero-shot ApexOracle。

共享配置位于 `configs/antibiotic_classification/legacy_three_strain.yaml`。三种模式共享数据准备、target/fold adapter、feature loader、指标、AUROC best-tracker、checkpoint 命名和 ensemble orchestration；full-fusion 与 molecule-only 只在真实模型和训练行为不同处保留 mode adapter。

## 已由代码、日志和真实运行验证的事实

- 三个旧 MDLM driver 的前 834 行完全相同；Dataset、collate、fusion、head 和绝大部分训练逻辑属于复制代码。
- full-fusion 模式使用完整 MIC genome-text frame，同时再次把 genome-backed MIC 记录放入 text route；这会让这些 MIC 记录通过两个 route 各训练一次。统一 runner 明确保留该行为。
- batch size 为 70，默认 25 epochs、10 ensembles、Adam `1e-5`、CosineAnnealingLR、`freeze_epochs=3000`。
- strict zero-shot 的三个目标 checkpoint 网格完整，共 `3 × 10 = 30` 个文件，并有三个完整结束的日志。
- fine-tune 预期 `3 × 10 × 5 = 150` 个 checkpoint，现存 77 个；14 个日志中只有 6 个包含最终五折汇总。因此它只能标记为历史证据不完整。
- molecule-only 的 `3 × 10 × 5 = 150` checkpoint 网格和 15 个日志均完整，但它是 molecule-only 对照。
- strict zero-shot 日志最终 ensemble 指标为：`#004` 0.9360/0.5890，`17978` 0.7262/0.3243，RN4220 0.7679/0.1655（AUROC/AUPRC）。
- 真实数据 dry-run 与旧日志中的 Dataset 计数逐项一致。以 `#004` 为例，512-token 过滤后的两路 MIC 为 77,372/89,118，辅助 genome/text 分类为 7,684/39,311，held target 为 2,335。
- H100 上使用 group 0 / ensemble 0 的 capsule checkpoint、batch size 70 和全 eval checkpoint inference 时，canonical runner 与旧 capsule 的 2,335 个 molecule ID、label 和 logit 逐值完全相同；AUROC/AUPRC 均为 `0.9350169300225734 / 0.6120361666817208`。详见 `checkpoint_validation_group0_ensemble0.json`。
- 合成四路 batch 的一轮 CUDA 集成测试通过，覆盖 MIC genome、MIC text route、辅助 genome classification、辅助 text classification、scheduler 和 held-target evaluation。

checkpoint 文件网格和日志完成度来自 `checkpoint_family_audit.json`，生成命令为：

```bash
python scripts/audit/audit_antibiotic_classification_checkpoints.py \
  --output experiments/fig1b_antibiotic_classification/checkpoint_family_audit.json
```

## 必须保留的历史行为

- full-fusion 每个 epoch 只对 genome attention、text attention 和 regression head 调用 `train()`/`eval()`；classification head 在 held-target checkpoint selection 时仍保持 train mode，因此 dropout 参与 AUROC/AUPRC。
- molecule-only 在评估前显式调用 classification head 的 `eval()`。
- held target 直接用于逐 epoch best-AUROC checkpoint selection，没有独立 validation set。
- AUROC 提升时先保存 checkpoint，随后才更新 best AUPRC；因此 checkpoint 内的 `auprc` 可能落后于同 epoch 实际 AUPRC。
- strict zero-shot 的目录名仍包含历史遗留的 `10_fold`，但该模式没有 target KFold。
- RN4220 作为 target 时，两个 genome-backed auxiliary block 来自 Python set difference；其顺序仍受 `PYTHONHASHSEED` 影响。旧日志没有记录该环境变量，不能声称恢复了逐 batch 的精确历史顺序。

## 推理指标与训练日志的边界

历史训练日志中的 best prediction 来自 classification head 的 train-mode dropout。加载 checkpoint 后做确定性推理时，canonical `--evaluate-checkpoints` 会把全部消费模块设为 eval，这与 capsule 的推理契约一致，但不应被描述为逐 bit 重放训练时的随机 best predictions。

此外，bfloat16 attention 对分批方式敏感：旧 capsule 默认 batch size 64，而论文训练 driver 使用 70。group 0 / ensemble 0 在 64 与 70 下 AUROC 相同，AUPRC 相差约 `2.9e-5`；统一到 batch size 70 后逐样本 logit 完全一致。

## 仍待确认或不在本阶段声称的事项

- fine-tune 结果证据不完整，不应从现存 77 个 checkpoint 推断完整五折正式结果。
- 本阶段没有重新训练 30 个 strict zero-shot ensemble；保留并验证的是原行为和已有 checkpoint 推理。
- group 1/2 的其余 29 个 strict checkpoint 已完成文件网格审计，尚未逐文件做 H100 推理；高置信度正式日志数值保持不变。
