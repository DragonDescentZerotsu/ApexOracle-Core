# Remasking reviewer peptide-label structure audit

> 日期：2026-07-29
> 状态：已完成 checkpoint identity、amide QC、`[SEP]` suffix 和 reviewer-retrained
> head 交叉检查。该审计不修改生成结果、论文或 reviewer response。

> **2026-07-29 后续决定：** 本文记录的是发现历史 classifier/structure 冲突时的审计状态。
> 作者随后冻结了“窄结构筛选 + first `[SEP]` 后正确 padding 的历史 v1 classifier”联合口径，
> 并将 `figures/remasking_structure_qualified_peptides_with_mic_control` 确认为最终 reviewer
> 图。该联合口径仍是 preliminary narrow criterion，不是通用 peptide ground truth。历史
> 四面板和 yield-only 图均保留为 legacy；完整索引见 `figures/README.md`。

## 1. 直接结论

当前 `peptide_classifier_positive` 只能解释为历史 v1 classifier 的 operational label，
不能用于声称生成结构真实是 peptide，也不能把 classifier-negative 直接称为 small molecule。

用户提出的疑问成立：对于普通的多残基 peptide，至少应存在一个 amide/peptide bond。少数单个
amino acid、特殊 peptidomimetic 或非典型连接形式可能没有普通 amide bond，但本次高置信度
classifier-positive 且无 amide 的例子包括 `CC`、`O=[Ti+2]`、`N#[O+]`、`[N+]=CN` 和
`N[CH]F`，显然不能用这些例外解释。

因此，现有四面板图中的 panel a、c、d 若把 classifier-positive 直接标为 “peptide”，会误导
reviewer。在建立独立的 structure-based peptide criterion 并重算之前，这些 panel 只能标成
“v1-classifier-positive/negative”，不能作为 peptide/small-molecule composition。

## 2. 已由代码、checkpoint 和输出验证的事实

### 2.1 权重身份

- 本次 generation 和 clean-input post-hoc evaluation 使用的历史 v1 checkpoint 均为：
  `/data2/tianang/projects/mdlm/cls-guide-pad-no-mask-checkpoints/epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt`
- 文件大小为 `393,226,092` bytes，SHA-256 为
  `40f638ca5668f20a641a538035015b1741ab69cded300cba27f7148cc291945b`。
- 正式 reviewer classifier retrain 确实从同一个 checkpoint 提取 backbone；但是该实验冻结
  backbone 后，**重新初始化并训练了两个新的 `768→384→128→1` classification heads**。
- 因而 reviewer retrain 的近乎完美 clean AUROC/AUPRC 验证的是
  “frozen v1 features + newly trained heads”在 molecule/sequence-disjoint **来源标签**测试集上的
  可分性，不是历史部署 head 的性能，也不是生成分子上的 structure-based peptide identity。
- 用审计脚本重新计算历史 head 的全部 3,600 个概率，与冻结
  `evaluated_attempts.csv` 的最大绝对差为 `8.49e-7`。这证明本次异常不是误加载了另一个
  checkpoint 或错误重建了 head。

### 2.2 原 amide QC 的确偏窄，但不是主因

generation runner 使用：

```text
[NX3][CX3](=O)[#6]
```

该 SMARTS 要求 carbonyl carbon 还连接 carbon，因此比 RDKit
`rdMolDescriptors.CalcNumAmideBonds` 的 general amide 定义更窄。

在 2,355 个 runner 判为 RDKit-valid 的结构中：

- runner SMARTS 检出 622 个；
- RDKit general amide 检出 702 个；
- runner SMARTS 漏掉 80 个 general-amide-positive 结构。

所以原先“无 amide”的 QC 数量有轻度高估，但修正 SMARTS 后仍不能解决 classifier/structure
冲突。

### 2.3 历史 v1 head 与 amide 的冲突

所有 2,355 个 valid structures 中：

- 历史 full-token v1 head 判 1,343 个 positive；
- 其中可重新解析的 1,341 个 positive 里，只有 596 个有 general amide；
- 745/1,341（55.6%）连一个 general amide 都没有。

当前论文 window `0.55--0.45` 的 395 个 valid structures 中：

- 历史 full-token v1 head 判 213 个 positive；
- 其中 88 个有 general amide，125 个没有；
- 即 58.7% 的 classifier-positive structures 没有 general amide。

### 2.4 `[SEP]` 后 token 泄漏存在，但只解释一部分

runner 实际只解码 first `[SEP]` 之前的 token；原 evaluation 却把完整固定长度 token tensor
送入 classifier，其中可能包含 first `[SEP]` 后不会进入 molecule 的非-PAD token。

把 first `[SEP]` 后全部位置替换为 PAD 后：

- 所有 valid structures 的 positive 数从 1,343 降至 1,209；
- 166 个 positive 变 negative，32 个 negative 变 positive；
- 当前 window 的 positive 数从 213 降至 191；
- 当前 window 的 191 个 positive 中仍有 105 个没有 general amide。

因此 suffix leakage 是真实评估缺陷，但修复后仍有 55.0% 的 current-window positive structures
没有 general amide，不能解释主要冲突。

### 2.5 高性能 reviewer-retrained heads 也没有解决生成 OOD 问题

把两个已完成 reviewer-retrained heads 应用于相同的 SEP-padded generated inputs，并按正式
reviewer summary 的方式平均 logits：

- 2,355 个 valid structures 中判 1,353 个 positive；
- 其中 731/1,353（54.0%）没有 general amide；
- current window 判 220 个 positive，其中 130/220（59.1%）没有 general amide。

`N[CH]F`、`N#[O+]` 和 `[N+]=CN` 等结构仍可得到接近 1 的 ensemble probability。这说明
reviewer retrain 的高测试性能主要回答来源标签分布内的可分性，不能外推为 generated-OOD
structure identity。

### 2.6 同一 canonical molecule 的 label 可因 SELFIES 表示不同而变化

2,355 个 valid rows 对应 1,949 个 canonical molecules；其中 109 个 canonical molecules 出现
多次。即使 first `[SEP]` 后已统一 PAD：

- 历史 v1 head 在 45/109 个重复 canonical molecule groups 内给出互相矛盾的 binary labels；
- reviewer-retrained ensemble 在 41/109 个 groups 内也出现 label disagreement。

例如同一个 canonical `N#[O+]` 或 `N[CH]F` 的不同 generated SELFIES 表示会跨过 0.5 threshold。
这进一步说明 token classifier score 不是 canonical molecular structure invariant，不能单独用作
结构类别真值。

## 3. 根据现有证据作出的解释

1. v1 label 是按数据来源赋值：SmProt2、UniProt/UniRef、PeptideCLM-generated 为正，
   PubChem 为负，而不是由独立 structure parser 标注。模型可以利用来源、tokenization、长度或
   其他数据集特征完成这一任务。
2. 本次生成输出明显偏离训练分布，包含 radical、unusual element/charge 和极短结构。来源标签
   分类器在这种 OOD 区域的概率没有经过校准。
3. reviewer retrain 的 near-perfect metric 与上述现象不矛盾：它证明 held-out source-label
   molecules 很容易区分，不证明任意新生成结构具有 peptide backbone。

以上是由标签机制、生成样例和交叉检查共同支持的解释；尚未通过专门的 shortcut-attribution
实验确定模型具体依赖哪一种 token/source feature。

## 4. 对现有 reviewer 实验的影响

### 仍可使用

- 3,600 attempts 的 generation completion；
- RDKit-valid counts 只能作为 **RDKit parseability**，不能自动称为 chemically usable 或
  peptide-valid candidates；
- 已保存的 clean predicted MIC 可以保留作模型输出血缘，但 reviewer-facing activity summary
  应在 independent structure filter 后对合格候选重新汇总；对明显非 peptide/OOD 结构的 MIC
  prediction 不能解释成可信 activity evidence；
- current 与 `gamma_peptide=0` 的同协议 direct control；
- remasking window 对 RDKit parseability 的 sensitivity；
- “classifier-positive”作为与实际 guidance checkpoint 一致的 operational score，但必须明确
  限定含义。

### 暂时不能使用

- `53.9% peptides / 46.1% small molecules` 作为真实结构 composition；
- “valid peptide yield”作为无需限定的 peptide yield；
- 用当前 classifier proportion 直接回答 Reviewer 2 的
  “what proportion ... are peptides vs. small molecules?”；
- panel d 当前的 “Peptide/Small molecule”标签。
- 用全部 RDKit-valid outputs 的 predicted MIC trade-off 直接为 peptide-generation window
  背书；该结论需在 structure-qualified subset 上复核。

现有 figure 和 `RESULTS.md` 中相关数字保留作审计血缘，不应在修订信中引用为 structural peptide
proportion。

## 5. 最小下一步

无需重新生成 3,600 个 attempts。最小修复是对现有 2,355 个 valid structures 做一次独立、
canonical-structure-based peptide classification，并据此重算各 window 与
current-vs-control 的 peptide yield/composition。

候选规则需在正式使用前固定并验证：

1. general amide count 只能作为必要性 QC，不能单独作为 peptide 定义，因为许多 small-molecule
   drugs 也含 amide；
2. PepLink v0.1.2 reverse parser 可以高精度识别其 contract 内的 standard linear/head-to-tail
   peptides，但 coverage 较窄，不能把 unsupported 自动等同于 non-peptide；
3. 更稳妥的最终口径应组合 peptide-backbone topology、至少两个 residue-like units、元素/价态
   合理性，并用已知 peptide/small-molecule controls 验证 precision 与 coverage。

在该定义冻结前，建议不修改正式论文或 reviewer response，也不继续润色当前
peptide-composition figure。

## 6. 可复核入口

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/audit_remasking_peptide_classifier_structure.py

python -m pytest -q tests/test_remasking_schedule_reviewer.py
```

输出：

- compact report：
  `analysis/peptide_structure_audit/summary.json`
- local row-level audit：
  `analysis/peptide_structure_audit/audited_valid_attempts.csv`

本次 script SHA-256：
`38f52d51c223fb44019a2898f1f37d46a4b63c7c608ad89d0257720d21d40332`

本次 compact summary SHA-256：
`adcb913475f118f8c96db70b4525cda475595e9e2ac6710cc187eb58769104ef`
