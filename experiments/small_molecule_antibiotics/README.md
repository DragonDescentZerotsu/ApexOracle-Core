# 三菌株小分子抗生素数据入口

本目录记录 Fig. 1b 三菌株小分子二分类数据从三个公开来源到论文 token cache 的血缘。
canonical 代码只读取原始数据，拒绝覆盖输入或已存在输出；论文复现继续消费冻结文件，不会在训练
时重新生成数据。

## 已由代码和真实数据验证的事实

- `DataPrepare/DataCheck.ipynb` 保存了三个来源的历史转换单元，并非转换规则完全缺失。
- E. coli BW25113：`Activity == "Active"` 为阳性，得到 2,335 行、120 阳性。
- A. baumannii ATCC 17978：原始 `Mean` 小于全表均值减一个 sample standard deviation
  (`ddof=1`) 为阳性；冻结阈值为 `0.8057681044789169`，得到 7,684 行、480 阳性。
- S. aureus RN4220：直接复制 `ACTIVITY`，得到 39,312 行、512 阳性。
- 三个 block 的冻结顺序为 E. coli → A. baumannii → S. aureus。canonical builder 生成的
  49,331 行 CSV 与论文输入逐字节一致，SHA-256 为
  `4dabc0f8ac808d33ede3eacb47bacf7b55b2a900fcf78fd3d45a89c2037f3dc2`。
- IBM SELFIES tokenizer 固定到 revision
  `55e83392264cb998f7aa5014847df29868aefeb8` 后，重建的 49,330 行 token CSV 也逐字节一致，
  SHA-256 为 `d8e6391bfae3c35fe8d311461565df177fc75044cbc40204bf74f7ecf1fe7f27`。
  唯一排除记录是 `na_12751`，原因是 SELFIES tokenization 含 unknown token；没有 invalid
  SMILES 或超过 1,024 tokens 的记录。
- classification runner 现在从 versioned config 显式读取 token table 路径，不再依赖 runner
  内部的隐藏默认值。

## 历史实现的风险与发布边界

历史 notebook 用 processed 目录的未排序遍历结果做 merge，而且会把 merge 输出写回同一目录；
重复运行可能把输出自身再次读入。canonical builder 显式列出三个输入和冻结顺序，因此保留当前
论文行为，同时消除原地覆盖与自包含输入风险。旧 tokenizer driver 还使用硬编码绝对路径、未固定
Hugging Face revision 并允许覆盖输出，已由 legacy tag 保存后删除。

论文数据的完整 SHA-256、行数、标签规则和写入策略见
`configs/data_pipeline/small_molecule_antibiotics_paper.yaml`；机器可读验收记录见
`reconstruction_audit.json`。

## 只读重建示例

输出必须使用新的独立路径，例如：

```bash
python scripts/prepare_data/build_small_molecule_antibiotic_dataset.py build \
  --ecoli 'DataPrepare/Data/small_molecule/raw/cell~Escherichia_coli_BW25113~#004.csv' \
  --abaumannii DataPrepare/Data/small_molecule/raw/chem_bio_relative_growth~Acinetobacter_baumannii_ATCC_17978.csv \
  --saureus DataPrepare/Data/small_molecule/raw/nature_1_positive~Staphylococcus_aureus_RN4220.csv \
  --output results/small_molecule_rebuild/small_molecule_Evo_binary_data.csv

python scripts/prepare_data/build_small_molecule_antibiotic_dataset.py tokenize \
  --input DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data.csv \
  --output results/small_molecule_rebuild/small_molecule_Evo_binary_data_SELFIES.csv \
  --revision 55e83392264cb998f7aa5014847df29868aefeb8 \
  --local-files-only
```
