# 模型权重登记与迁移约定

`configs/model_weights.yaml` 是本仓库模型权重位置、身份和发布状态的唯一登记表。训练脚本中的路径和运行产物中的路径只能作为执行记录；后续重构应逐步改为通过该 manifest 和 `APEXORACLE_WEIGHTS_DIR` 解析权重。

## 当前约定

- 权重二进制不进入 Git；Git 只保存 manifest、下载说明、校验和与许可信息。
- 未来统一的本地根目录为 `${APEXORACLE_WEIGHTS_DIR:-weights}`。
- `future_storage.relative_path` 是权重迁移后的稳定相对路径。APEX encoder 已迁移并由 manifest
  resolver 加载；其余本地权重仍读取已登记的 `current_path`。
- 每个本地 checkpoint 在发布前必须具备 SHA-256、文件大小、稳定下载 URI、许可或再分发结论和至少一个加载 smoke test。
- 第三方 Hugging Face 权重不应默认复制再分发。优先固定 revision 并从上游下载；只有许可允许且确有离线发布需要时才镜像到统一存储。
- Fig. 1b 当前加载的 `Pep_emb_dict_cls_wo_pad_eval.pt` 和 `SM_emb_dict_cls_wo_pad_eval.pt` 是
  预计算 molecule feature cache，不是可训练模型权重，因此不登记为 checkpoint manifest ID；
  它们的条目数、SHA-256、消费代码和机器位置记录在 `docs/COMPUTE_AND_ASSET_MAP.md`。

## Fig. 2b 当前登记

| Manifest ID | 当前来源 | 已固定身份 | 未来相对路径 | 状态 |
| --- | --- | --- | --- | --- |
| `fig2b_dlm_only` | `/data2/tianang/projects/mdlm/Checkpoints_fangping/best_2.ckpt` | SHA-256 `fbbcc65…75e59` | `molecule_encoders/dlm_only/best_2.ckpt` | 作者已确认用于修订 benchmark；原论文精确身份仍是高置信度推断 |
| `fig2b_dlm_mtr_dlm` | `/data2/tianang/projects/mdlm/Checkpoints_fangping/best.ckpt` | SHA-256 `f8df1fb5…d8ca` | `molecule_encoders/dlm_mtr_dlm/best.ckpt` | 已验证用于修订 benchmark |
| `fig2b_dlm_mtr_dlm_small_candidate` | `node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.04.29/165523/checkpoints/best.ckpt` | SHA-256 `3c612c9c…6c9d6` | `molecule_encoders/dlm_mtr_dlm_small/best.ckpt` | 12-layer 容量匹配候选；尚未运行共同数据 benchmark |
| `fig2b_apex_encoder` | `weights/molecule_encoders/apex/APEX_pretrained_encoder_state_dict_best.ckpt` | SHA-256 `a4b37338…b2b9` | `molecule_encoders/apex/APEX_pretrained_encoder_state_dict_best.ckpt` | 已迁移；通过 manifest ID 加载并严格验证 |
| `fig2b_chemberta_mtr` | `DeepChem/ChemBERTa-77M-MTR` | revision `66b895ca…04ca` | `molecule_encoders/chemberta_mtr` | 已用于 Fig. 2b/2c 重构；旧 run 未记录 revision |
| `fig2b_chemberta_mlm` | `DeepChem/ChemBERTa-77M-MLM` | revision `ed8a5374…15ff` | `molecule_encoders/chemberta_mlm` | 已用于 Fig. 2b/2c 重构；旧 run 未记录 revision |
| `fig2b_molformer` | `ibm/MoLFormer-XL-both-10pct` | revision `7b12d946…0314` | `molecule_encoders/molformer` | 已固定 |
| `fig2b_peptideclm` | `aaronfeller/PeptideCLM-23M-all` | revision `a0847d82…fdda` | `molecule_encoders/peptideclm` | 已用于 Fig. 2b/2c 重构；旧 run 未记录 revision |
| `fig2c_strainwise_dlm_7ensemble` | `Checkpoints/.../MDLM_MTR_fix_7_fold_ensembles` | 21 个文件；代表文件 SHA-256 `9b097645…e24e` | `strainwise/dlm_7ensemble` | 3×7 网格与公共消费契约已确认；group 1 的 7 个文件额外含 frozen MDLM payload；其余 20 个 SHA-256 待登记 |

Fig. 2c comparator 迁移使用上述固定 revision 来消除上游 `main` 漂移。三项新固定值均已与
现存 legacy checkpoint 的 state-dict 结构核对；但 2025 年原始训练脚本没有写入 Hugging Face
commit，因此这里只能称为“当前重构的固定复现 revision”，不能反向声称已证明旧 run 当时的
精确 upstream commit。MolFormer 的 revision 此前已因兼容性问题固定。

## Synergy 当前登记

| Manifest ID | 当前来源 | 已固定身份 | 状态 |
| --- | --- | --- | --- |
| `synergy_mic_base_epoch_100` | `Checkpoints/.../guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth` | SHA-256 `f24faf67…c3a4` | 三折候选加载的 base；不是已确认的 Methods 13-epoch 权重 |
| `synergy_three_fold_7ensemble_candidate` | `Checkpoints/.../MDLM_3_fold_ensembles_1_base_model_cls` | 21 个 member 的完整 manifest | 作者接受的论文高置信度复现候选；不是精确原始 checkpoint 声明 |

作者于 2026-07-19 将本仓库范围收缩为只复现论文已汇报结果，因此 post-paper all-data
guidance、prospective regression 和 screening 权重不再由本发布 manifest 管理；本次只从 Git
登记和代码中移除这些条目，没有删除本地 checkpoint 二进制。清理范围与恢复位置见
`reproducibility/synergy_paper_only_cleanup_2026-07-19.json`。

作者同日确认本机 mean AUROC/AUPRC `0.7598/0.7440` 可作为论文实现的高置信度复现候选；
与论文 `0.7539/0.7454` 的绝对差为 `0.0059/0.0014`。rank 和 base-epoch 差异继续记录为
provenance 限制，但 synergy 重构阶段已关闭。

## DLM-only 决策记录

2026-07-18，作者确认修订后的 Fig. 2b benchmark 使用 `best_2.ckpt`。该文件是 12-layer、hidden size 768 的纯 DLM checkpoint，不包含 MTR regression branch；正式五折运行从 `ema.shadow_params` 加载，五个 fold 均为 `missing_keys=[]`、`unexpected_keys=[]`。

这里必须保留两个不同的证据结论：

1. **已确认事实：** 修订后的共同数据 benchmark 使用 manifest 中 SHA-256 为 `fbbcc65f85013297212342e7d3286fc9b3ab6fbf0d9b28a0407e11d63b875e59` 的 `best_2.ckpt`。
2. **高置信度推断：** `best_2.ckpt` 最可能是旧论文 DLM MLM bar 对应的 checkpoint，但旧 W&B 日志没有保存 checkpoint 路径，因此不能声称已绝对证明旧论文使用了这个文件。

## DLM 容量匹配搜索结论

已核验联合模型的原始 checkpoint 目录：

`node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.05.06/112126/checkpoints`

该目录现存 `best.ckpt`、`last.ckpt` 以及 step 960000、970000、980000、990000、1000000 的 checkpoint；每个文件均为 5,268,558,165 bytes。原始训练源码 `/data1/fangping/mdlm/diffusion.py` 使用 `loss + 0.1*reg_mse`，`/data1/fangping/mdlm/models/dit.py` 为该模型构建 209-descriptor regression head。因此这些文件都是从训练开始就采用联合 DLM+MTR objective 的 24-layer、hidden size 1024 checkpoint。

截至 2026-07-18，已搜索本机与 node002 的 Tianang/Fangping checkpoint 和项目目录、对应 W&B projects，以及公开 Hugging Face repository，**没有找到同为 24-layer、hidden size 1024 的纯 DLM checkpoint**。当前正式 Fig. 2b 使用的 12-layer `best_2.ckpt` 与 24-layer `best.ckpt` 是两个预训练模型版本的比较，不是严格控制容量后的 MTR objective ablation。不能通过从 24-layer `best.ckpt` 删除 regression head 来构造 DLM-only，因为 backbone 参数在预训练中也受到联合目标更新。

不过，node002 上存在一个可与 `best_2.ckpt` 做容量匹配比较的 small joint checkpoint：

`node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.04.29/165523/checkpoints/best.ckpt`

该文件为 12-layer、hidden size 768，global step 650032，SHA-256 `3c612c9c68b9ee72c077dc1492153fa30d5c9fa4cb1753355bf146cff616c9d6`，大小 1,568,403,312 bytes，并包含 `768→768→209` regression head。它和纯 DLM `best_2.ckpt` 的 architecture、数据配置与 sequence length 相同，可以作为新的容量匹配 DLM+MTR 候选。

仍需保留一项限制：joint run 使用 learning rate `1e-4`、global batch size 480、最佳 step 650032；DLM-only run 使用 `3e-4`、global batch size 768、最佳 step 621036。因此新配对消除了模型容量差异，但还不是仅改变 objective 的完全单变量预训练实验。该 checkpoint 尚未运行共同数据五折 benchmark，不能提前替换当前 Fig. 2b 数值。

## 后续迁移验收

迁移某个权重时，应依次完成：复制到统一根目录、复核 SHA-256、填写稳定下载 URI、记录许可、让加载代码改用 manifest ID、运行等价性 smoke test，最后才删除或归档旧路径。不得先移动文件再依靠全仓库搜索修复路径。
