# PeptideCLM tokenizer vendor boundary

这里只保留 ApexOracle 论文比较实验实际需要的 PeptideCLM SMILES SPE tokenizer、vocabulary
和 merges。训练示例、notebook、clustered dataset 与其他未被调用的 tokenizer 不属于发布运行时。

- 上游：<https://github.com/aaronfeller/PeptideCLM>
- 模型：`aaronfeller/PeptideCLM-23M-all`
- 固定 revision：`a0847d8231d236645a2c4f629590118716c6fdda`
- 许可：本目录中的 `LICENSE`（MIT）
- `new_vocab.txt` SHA-256：`96af9b4e0aa1fee93a9123720c1a46f977269ecc4ff81ac1cd4e0385b8f4aa2e`
- `new_splits.txt` SHA-256：`90446eadc9a1d8722be58b6b8e31c2756e0d321cd448cb5fae7538a33e3e3af4`

迁移验收要求是 representative SMILES 的 token IDs、attention mask、special-token mask、decode
结果与旧 `PeptideCLM/tokenizer/my_tokenizers.py` 完全一致。
