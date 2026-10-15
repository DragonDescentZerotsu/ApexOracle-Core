# ReMDM remasking-schedule figures

## Canonical final figure

三面板图使用以下 stem：

`remasking_structure_qualified_peptides_with_mic_control`

默认入口：

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp MPLBACKEND=Agg \
  conda run --no-capture-output -n mdlm \
  python scripts/audit/plot_remasking_structure_qualified_peptides.py
```

canonical 交付包含 PDF、SVG、PNG、`*_caption.md`、`*_plotted_data.csv` 和
`*_manifest.json`。manifest 冻结输入、脚本和输出 SHA-256。panel a 与 panel c 的
`Peptide yield` 使用完全相同的联合定义：通过窄结构筛选，并且在 first `[SEP]` 后正确
padding 的历史 v1 classifier score 不低于 0.5。图稿状态为 final；该联合判据本身仍是
preliminary narrow criterion，不能扩展声称为通用 peptide ground truth。

canonical panel c：在同一个三行 dot-and-interval
坐标区中展示 RDKit-valid yield、peptide yield，以及 current guidance 与 no peptide
guidance 的 all-RDKit-valid median predicted MIC（37.9 vs 56.2 $\mu\mathrm{M}$）。数值轴压缩
5--25 区间，每行标签单独标明 `%` 或 $\mu\mathrm{M}$ 单位。两类横向
error bars 分别为三个 seed-level pooled rates 和三个 seed-level pooled median MIC 的
sample s.d.。此修订原地更新 canonical stem 及其 caption/plotted-data/manifest，legacy stems
未改动。

## Legacy figures

历史版本使用以下 stem：

- `remasking_structure_qualified_peptides`：最终三面板图之前的 yield-only 版本。
- `remasking_schedule_reviewer`：structure audit 之前、以历史 classifier label 为主的
  四面板 pooled-median 版本。
- `remasking_schedule_reviewer_violin`：上述历史四面板图的 MIC violin 变体。
