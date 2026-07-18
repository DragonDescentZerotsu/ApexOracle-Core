# 模型权重登记与迁移约定

`configs/model_weights.yaml` 是本仓库模型权重位置、身份和发布状态的唯一登记表。训练脚本中的路径和运行产物中的路径只能作为执行记录；后续重构应逐步改为通过该 manifest 和 `APEXORACLE_WEIGHTS_DIR` 解析权重。

## 当前约定

- 权重二进制不进入 Git；Git 只保存 manifest、下载说明、校验和与许可信息。
- 未来统一的本地根目录为 `${APEXORACLE_WEIGHTS_DIR:-weights}`。
- `future_storage.relative_path` 是权重迁移后的稳定相对路径；当前 `migration_status` 仍是 `planned_not_moved`，因此现阶段训练继续读取 `current_path` 或上游 Hugging Face model ID。
- 每个本地 checkpoint 在发布前必须具备 SHA-256、文件大小、稳定下载 URI、许可或再分发结论和至少一个加载 smoke test。
- 第三方 Hugging Face 权重不应默认复制再分发。优先固定 revision 并从上游下载；只有许可允许且确有离线发布需要时才镜像到统一存储。

## Fig. 2b 当前登记

| Manifest ID | 当前来源 | 已固定身份 | 未来相对路径 | 状态 |
| --- | --- | --- | --- | --- |
| `fig2b_dlm_only` | `/data2/tianang/projects/mdlm/Checkpoints_fangping/best_2.ckpt` | SHA-256 `fbbcc65…75e59` | `molecule_encoders/dlm_only/best_2.ckpt` | 作者已确认用于修订 benchmark；原论文精确身份仍是高置信度推断 |
| `fig2b_dlm_mtr_dlm` | `/data2/tianang/projects/mdlm/Checkpoints_fangping/best.ckpt` | SHA-256 `f8df1fb5…d8ca` | `molecule_encoders/dlm_mtr_dlm/best.ckpt` | 已验证用于修订 benchmark |
| `fig2b_apex_encoder` | `compare_APEX/APEX_ckpt/APEX_pretrained_encoder_state_dict_best.ckpt` | SHA-256 `a4b37338…b2b9` | `molecule_encoders/apex/APEX_pretrained_encoder_state_dict_best.ckpt` | 原实现和权重保持不变 |
| `fig2b_chemberta_mtr` | `DeepChem/ChemBERTa-77M-MTR` | revision 待固定 | `molecule_encoders/chemberta_mtr` | 发布前待处理 |
| `fig2b_chemberta_mlm` | `DeepChem/ChemBERTa-77M-MLM` | revision 待固定 | `molecule_encoders/chemberta_mlm` | 发布前待处理 |
| `fig2b_molformer` | `ibm/MoLFormer-XL-both-10pct` | revision `7b12d946…0314` | `molecule_encoders/molformer` | 已固定 |
| `fig2b_peptideclm` | `aaronfeller/PeptideCLM-23M-all` | revision 待固定 | `molecule_encoders/peptideclm` | 发布前待处理 |

## DLM-only 决策记录

2026-07-18，作者确认修订后的 Fig. 2b benchmark 使用 `best_2.ckpt`。该文件是 12-layer、hidden size 768 的纯 DLM checkpoint，不包含 MTR regression branch；正式五折运行从 `ema.shadow_params` 加载，五个 fold 均为 `missing_keys=[]`、`unexpected_keys=[]`。

这里必须保留两个不同的证据结论：

1. **已确认事实：** 修订后的共同数据 benchmark 使用 manifest 中 SHA-256 为 `fbbcc65f85013297212342e7d3286fc9b3ab6fbf0d9b28a0407e11d63b875e59` 的 `best_2.ckpt`。
2. **高置信度推断：** `best_2.ckpt` 最可能是旧论文 DLM MLM bar 对应的 checkpoint，但旧 W&B 日志没有保存 checkpoint 路径，因此不能声称已绝对证明旧论文使用了这个文件。

## 同容量 DLM-only 搜索结论

已核验联合模型的原始 checkpoint 目录：

`node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.05.06/112126/checkpoints`

该目录现存 `best.ckpt`、`last.ckpt` 以及 step 960000、970000、980000、990000、1000000 的 checkpoint；每个文件均为 5,268,558,165 bytes。原始训练源码 `/data1/fangping/mdlm/diffusion.py` 使用 `loss + 0.1*reg_mse`，`/data1/fangping/mdlm/models/dit.py` 为该模型构建 209-descriptor regression head。因此这些文件都是从训练开始就采用联合 DLM+MTR objective 的 24-layer、hidden size 1024 checkpoint。

截至 2026-07-18，已搜索本机与 node002 的 Tianang/Fangping checkpoint 和项目目录、对应 W&B projects，以及公开 Hugging Face repository，**没有找到同为 24-layer、hidden size 1024 的纯 DLM checkpoint**。现有 `best_2.ckpt` 与 `best.ckpt` 的比较是两个预训练模型版本的比较，不是严格控制容量后的 MTR objective ablation。不能通过从 `best.ckpt` 删除 regression head 来构造 DLM-only，因为 backbone 参数在预训练中也受到联合目标更新。

若后续论文必须将差异归因于 MTR objective，需要恢复尚未发现的旧备份，或使用相同训练数据、步数和 24-layer/1024 配置重新训练纯 DLM。找到新权重后必须先补录 SHA-256、大小、来源和加载验证，再替换当前解释。

## 后续迁移验收

迁移某个权重时，应依次完成：复制到统一根目录、复核 SHA-256、填写稳定下载 URI、记录许可、让加载代码改用 manifest ID、运行等价性 smoke test，最后才删除或归档旧路径。不得先移动文件再依靠全仓库搜索修复路径。
