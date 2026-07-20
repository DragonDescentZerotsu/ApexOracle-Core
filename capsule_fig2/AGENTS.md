# Fig. 2b capsule 维护约定

- `code/reproduce_fig2b_mic_regression.py` 只审计正式共享数据 35-fold 结果，不训练模型。
- `code/prepare_fig2b_mic_regression_resources.py` 只打包 canonical runner、配置、实验说明、
  migration audit 和最终小型指标文件。
- 不重新引入根目录 `fix_*`、外部 MDLM 原始 driver 副本或 `chemberta_mlm_mean`。
- `data/` 是 builder 生成物；更新正式 Fig. 2b 结果后必须重新运行 builder 和审计入口。
- 原始数据、模型权重、feature cache、head checkpoint 和 optimizer state 不进入该目录。
- 该 capsule 只做小型结果审计，CPU 即可；不要占用当前 Fig. 1b 的本机/node GPU，也不要在
  node001/node002 的共享 release 工作树中另建一套 capsule 训练任务。
