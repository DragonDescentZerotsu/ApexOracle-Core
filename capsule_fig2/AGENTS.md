# Fig. 2b capsule 维护约定

- `code/reproduce_fig2b_mic_regression.py` 只审计正式共享数据 35-fold 结果，不训练模型。
- `code/prepare_fig2b_mic_regression_resources.py` 只打包 canonical runner、配置、实验说明、
  migration audit 和最终小型指标文件。
- 不重新引入根目录 `fix_*`、外部 MDLM 原始 driver 副本或 `chemberta_mlm_mean`。
- `data/` 是 builder 生成物；更新正式 Fig. 2b 结果后必须重新运行 builder 和审计入口。
- 原始数据、模型权重、feature cache、head checkpoint 和 optimizer state 不进入该目录。
