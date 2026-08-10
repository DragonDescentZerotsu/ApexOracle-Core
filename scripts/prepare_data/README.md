# Data preparation entrypoints

本目录只保留可参数化、对输入只读且具有测试或 manifest contract 的公共数据准备入口。论文时期的
硬编码 producer 不属于公共 API；它们的源码由 Core recovery tags 恢复。

## Strain-text embeddings

Canonical CLI：

```bash
python scripts/prepare_data/embed_strain_texts.py \
  --input-dir /path/to/strain_descriptions \
  --output-dir /path/to/text_embeddings \
  --device cuda:0 \
  --local-files-only
```

安装 package 后也可运行 `apexoracle-embed-strain-texts`。默认固定模型
`YBXL/Med-LLaMA3-8B` revision
`567e7e71d8b6b433d8bc494f8112176bec4afccf`，提取倒数第二层并保存 `[tokens, 4096]`
float32 tensor。CLI 自动识别两套 paper-era filename encoding：ATCC 文件名的 `_` 表示空格；
text-only 文件名的 `～` 和 `^` 分别表示空格与 `/`。两种模式均在 tokenization 前把精确 strain 名替换为
`This strain`。

默认 `--existing skip` 不覆盖已存在 tensor；也可显式选择 `error` 或 `overwrite`。每次运行写出
`strain_text_embedding_manifest.json`，记录 source/normalized-text/output SHA-256、model revision、
hidden-state index、shape、dtype 与 replacement warnings。数据、文本、tensor 和模型权重不进入 Git。

验证命令：

```bash
PYTHONPATH=src python -m pytest -q tests/test_strain_text_embeddings.py
```

两条真实历史资产的 GPU parity 记录位于
`reproducibility/strain_text_embedding_parity_2026-08-10.json`：text-only 样本逐元素完全一致；
ATCC 样本 shape/dtype 一致且在 `rtol=1e-5, atol=1e-4` 下 allclose。
