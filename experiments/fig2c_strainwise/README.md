# Fig. 1a / Fig. 2c strain-wise DLM ensemble

本目录迁移论文中高置信度的 strain-wise MIC regression 实验。历史 canonical
driver 为仓库根目录的
`DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py`；
现阶段其外层训练控制流保持不变，但 Dataset、四种 collate、cross-attention 和
regression/classification head 已切换到 `src/apexoracle/` 的共享实现。
四条单批次训练路径（regression/classification × genome+text/text-only）也已切换到
共享 forward、loss、backward、gradient clipping 和 optimizer-step 实现。epoch-0 baseline、
逐 epoch evaluation、不同长度 DataLoader 的 `zip_longest` 协调、cosine scheduler、prediction
accumulation、partition sentinel、best-metric tracker 和 checkpoint payload 也已抽取；legacy
driver 当前只保留 outer loop shell、日志和文件写入位置。

## 已由代码、日志和真实 checkpoint 验证的事实

- 三个历史日志的 ensemble R² 为 `0.4057`、`0.6889` 和 `0.6434`，平均值为
  `0.5793`。
- 本机和 node002 的最终 driver、MIC CSV、small-molecule auxiliary CSV、strain
  mapping 和 taxonomy mapping 的 SHA-256 完全一致；四类 embedding/FASTA 文件清单
  数量及文件名清单也一致。
- 历史 split 构造同时使用无序 `set` 和原地 taxonomy-alias list mutation。三个 GPU
  fold 是独立 Python 进程，日志没有记录各自的 `PYTHONHASHSEED`，因此仅由当前源码
  重新执行不能证明恢复了 2025 年的精确 strain membership。
- `legacy_protocol_manifest.json` 同时记录 `PYTHONHASHSEED=0` 的确定性候选 membership
  和历史日志中的权威样本计数。候选 membership 不得标注为历史精确 membership。
- group 0 / ensemble 0 checkpoint 已通过全部 state dict 的 `strict=True` 加载和两次
  独立 H100 固定批次验证；结果见
  `checkpoint_validation_group0_ensemble0.json`。
- 21 个 checkpoint 的公共消费契约和 `3 × 7` 网格已经全部扫描。group 0/2 的
  14 个文件只保存 fusion/heads；group 1 的 7 个文件额外保存名为
  `ChemBERTa_state_dict` 的 131-key、12-layer/768 MDLM backbone。三个 group 的
  optimizer 都只有相同的 5 个参数组和 49 个 state entries，证明额外 backbone
  没有在下游训练中更新。
- 四条共享单批次训练路径已经逐项比较 logits、loss、所有参数 gradient 和 Adam step
  后的参数，结果与 legacy 操作完全一致。测试强制触发了默认运行不会到达的 epoch-5000
  clipping 分支，确认 text attention 不裁剪，且 classification 历史上错误地裁剪
  `reg_head` 而不是 `cls_head`；本次为保持行为没有修正。
- 上述四路径严格比较使用 CPU float32；另在 H100 上用 CUDA autocast、GradScaler 和有限的
  合成 scale `128` 验证了 genome+text regression，logits、loss、有限 gradients 和 Adam
  更新后的参数逐位一致。合成小模型在默认初始 scale `65536` 下两侧都会 overflow 并跳过
  step，因此该 case 不能作为参数更新证据；正式 driver 的历史默认动态 scaler 未被修改。
- evaluation helper 不调用 `eval()`；原 driver 在初始和逐 epoch 评估期间继续保持模块
  `train()` mode，因此 dropout 仍参与 held-out-fold checkpoint selection。accumulator 保留
  genome+text/text-only loss 与预测分区、species 首次出现顺序、epoch-0 train-mean baseline，
  以及分区样本数不超过 1 时返回 `-1000` 的历史规则。
- loader orchestration 仍以最长 loader 为迭代长度并用 `None` 填充较短 loader；scheduler
  在四类训练 batch 全部处理后每个 epoch 调用一次。best R²、Spearman 和 Pearson 使用严格
  `>`，R² 相等时不会替换 best predictions 或 checkpoint。

## 根据现有证据作出的推断

- node002 上当前 driver 的修改时间是 2025-05-09 10:40；group 1 日志从
  2025-05-08 20:50 持续到次日 21:11。因此 group 1 进程很可能持有修改前的内存
  代码版本，这与其 checkpoint 多保存 backbone 相符。

## 仍待作者或旧源码确认

- group 1 当时是否在线调用 frozen MDLM 生成 molecule feature，还是只额外保存了
  一个未消费的 backbone。当前 loader 会显式登记该 payload，但只对实际消费的
  fusion/head state dict 做 `strict=True` 加载；不得把“未进入 optimizer”等同于
  “forward 中未使用”。

## 兼容入口

新入口仍调用经过审计的 legacy training loop，但实际使用共享核心模块：

```bash
python scripts/reproduce/run_fig2c_strainwise.py \
  --test-group 0 \
  --device 0 \
  --epochs 25 \
  --acknowledge-dynamic-legacy-split
```

`--acknowledge-dynamic-legacy-split` 是强制参数，避免把新进程生成的 fold 误称为
历史 checkpoint 的精确 fold。当前正式论文结果继续以保存的 checkpoint 和完整日志
为准；在精确 membership 进一步恢复前，不重新训练并替换正式数值。

如需重建已登记的确定性候选 manifest，必须在进程启动前显式固定 hash seed：

```bash
PYTHONHASHSEED=0 python scripts/prepare_data/build_fig2c_strainwise_manifest.py
```

该命令只重建候选 membership；不会将它升级为 2025 年历史 membership。

## 验证命令

```bash
pytest -q tests/test_strainwise_legacy_equivalence.py

python scripts/reproduce/validate_fig2c_strainwise_checkpoint.py \
  --checkpoint Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles/genome_text_learnable_emb_Strain_wise_best_R2_group_0_ensemble_0.pth \
  --device cuda:0
```

## 仍待完成

- 为其余 20 个 ensemble checkpoint 登记 SHA-256；公共 contract 扫描已经完成。
- 建立完全脱离根目录 legacy driver 的 config-driven runner；当前共享 helper 已覆盖
  DataLoader 协调、scheduler、evaluation accumulation 和 checkpoint selection，但 outer loop
  shell、日志与输出路径仍在 legacy driver。
- 在完成上述迁移及旧 checkpoint 等价检查后，才删除根目录中的重复 comparator 和
  pooling/eval 脚本。
