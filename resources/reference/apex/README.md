# APEX AAindex reference asset

`aaindex1.csv` 是 APEX encoder 使用的 566 维 amino-acid property table。发布仓库不再从
历史 `compare_APEX/` 目录读取该文件。

由于上游 APEX 许可仅明确允许非营利研究用途，并限制向商业第三方分发，本仓库不追踪该
CSV。请从 APEX 官方仓库获取 `aaindex1.csv`，在接受上游许可后放到本目录：

- 上游文件：<https://gitlab.com/machine-biology-group-public/apex/-/raw/main/aaindex1.csv>
- 上游许可：<https://gitlab.com/machine-biology-group-public/apex/-/raw/main/LICENSE>
- 期望 SHA-256：`478510fa4ee5e23c871558789a3f8450c20615b82dc294794a24bfe55e2bfaef`
- 期望大小：`64823` bytes

本地迁移自论文运行目录的文件已经过 SHA-256 核验，与上述身份一致。该文件仍受上游
许可约束，不因被放入此目录而改变授权条件。
