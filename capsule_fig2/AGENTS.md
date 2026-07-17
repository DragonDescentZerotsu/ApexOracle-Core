This folder is a trimmed Code Ocean capsule for Fig. 2b MIC regression only.

Runtime entrypoint:

- `code/run`

Required runtime resources:

- `data/fig2b_mic_regression/**`

The run path is inference-only. It reloads frozen feature caches and the
5-fold regression-head checkpoints, then writes Fig. 2b metrics under
`results/`.
