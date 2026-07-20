# ApexOracle Fig. 2b shared-benchmark audit capsule

该目录是 Fig. 2b 修订实验的轻量结果审计包。它读取正式共享数据实验保存的 35 个 fold
指标，重新计算七个模型的 mean R² 和 sample SD，并逐项校验论文使用的最终汇总。

审计协议固定为 `fig2b-shared-native-intersection-v2`：七个模型使用相同的 10,886 个
molecule IDs 和同一组五折划分。运行：

```bash
code/run
```

输出为 `results/fig2b_shared_metrics_verified.json`。

这个 capsule 不再携带旧 native-retained-set feature cache、regression head、mean-pooling
diagnostic 或根目录 `fix_*` driver。它也不声称能够在 Code Ocean 中重新训练 35 个模型；
完整训练入口、模型特有协议、权重 manifest 和迁移审计位于 `data/source/`，大型权重仍由
主仓库的外部资源策略管理。
