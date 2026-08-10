# ReMDM remasking reviewer 审计与补实验计划

> 状态：正式补实验、clean-model evaluation、structure audit、canonical reviewer figure、
> 正式论文修改及对应 reviewer response 均已完成。第 11--12 节保留为交互定稿过程，正式落稿
> 记录见第 13 节；跨仓库改动、GitHub 状态、提交白名单和统一公共仓库建议见
> `PUBLICATION_HANDOFF.md`。外部 sampler、配置和历史结果均未修改；本轮代码和紧凑证据已按主题
> 直接推送到 `Synergy/main`。
> 生成结果见 `RESULTS.md`，classifier/amide 冲突与当前使用边界见 `STRUCTURE_AUDIT.md`。
> 建立日期：2026-07-28。
> 目标：用同一套数学符号澄清 ReMDM、Campbell et al. (2024) 与实际 ApexOracle sampler 的关系，
> 并以预先定义的补实验回答 remasking interval、token selection 和 peptide yield 问题。

## 1. 当前结论

> **2026-07-29 重要更正与最终口径：** 本文档最初把 reviewer 补实验中的 v1
> classifier-positive operational label 简称为 peptide。独立审计显示，current window 的
> 213 个 full-token classifier-positive valid structures 中有 125 个没有 RDKit general amide
> bond；修正 first `[SEP]` 后 token leakage 后仍有 105/191 没有 general amide。reviewer-retrained
> 新 heads 也存在相同 generated-OOD 问题。因此，最终 reviewer figure 使用保守联合口径：
> 候选必须同时通过窄结构筛选和 first `[SEP]` 后正确 padding 的历史 v1 classifier 阈值。
> current/no-guidance 下分别为 20/600（3.3%）和 10/600（1.7%）。不满足该联合口径的候选只能
> 称为 “other generated structures”或“did not meet the conservative peptide criterion”，
> 不能自动等同于 small molecules。详见 `STRUCTURE_AUDIT.md` 和 `figures/README.md`。

### 1.1 对 reviewer 意见的判断

- **合理的部分：**
  1. 原稿没有给出选择 $t_{\rm on}=0.55,t_{\rm off}=0.45$ 的可复现实验证据；
  2. 原稿没有准确说明“哪些 token 有资格被 remask”以及 guidance 如何改变最终 remask 概率；
  3. 原稿没有量化 generated candidates 中 peptide 与 small-molecule 的比例，因此 Reviewer 2
     的 effectiveness
     问题目前确实没有被实验回答。
- **不构成正文缺陷的部分：**
  1. 正文已经给出 ReMDM posterior 并引用原方法，没有必要在正文重复证明 marginal preservation；
  2. Campbell/ReMDM 的数学联系是对 reviewer 比较的回应，放在 Reviewer 1 response 即可，不必
     扩写成正文 related work。
- **需要纠正的推论：**
  1. 使用统一的全局 loop window，不等价于假设每个分子或每个 token 都在同一时刻需要同等程度的
     correction。窗口只规定何时允许 reversible transition；实际 transition 是逐 token 随机采样，
     且 ApexOracle 中的最终概率依赖当前部分序列和 peptide guidance。
  2. 当前 base kernel 对每个 eligible token 独立使用同一个 $r_t$。是否 remask 是逐 token
     决定的，分子长度不是这个 Bernoulli decision 的输入，也不存在按 molecule 固定的 remask
     quota。因此，使用统一 window 并不能推出不同大小的分子被要求进行相同 correction。
  3. Campbell et al. (2024) 不是唯一有数学依据的 remasking 方法。Campbell 的 continuous-time
     detailed-balance construction 和 ReMDM 的 finite-step marginal-preserving posterior
     都是概率上自洽的构造，只是参数化、离散化和 token ordering 策略不同。

### 1.2 推荐的总体应对

不更换当前 ReMDM sampler。修订应完成三件事：

1. 在 **Reviewer 1 回复**中，用论文现有符号简短说明 ReMDM posterior 保持 MDLM marginal、
   与 Campbell construction 的联系和区别，以及它为何自然适用于 pretrained MDLM。正文没有必要
   展开 Campbell/ReMDM 的方法比较或重复 ReMDM 论文中的完整证明。
2. 在 Methods 中只补足本工作实际需要复现的信息：token eligibility、base remask proposal、
   peptide-guided reweighting 和完整 schedule，并删除“predicting a fixed proportion”等不准确
   表述。
3. 只做 reviewer 直接要求的两类实验：
   - remasking window sweep，用 peptide proportion 和 clean-model predicted MIC 说明区间如何选择；
   - stage-2 effectiveness evaluation，量化加入 peptide-guided correction 前后/有无该 correction
     时的 peptide proportion。

Reviewer 使用了 “mathematically decent manner”这一措辞，回复中可以直接说明
“our ReMDM-based construction is likewise mathematically decent”，随后用 marginal preservation
和 pretrained-MDLM compatibility 支撑这句话；无需回避或替换 reviewer 的用词。

## 2. 已由论文、论文 PDF 和代码验证的事实

### 2.1 当前 manuscript 已经写了什么

正式 TeX：

`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`

- Eq. 9 已写出 ReMDM-style finite-step posterior：

  $$
  q_r(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}) =
  \begin{cases}
  \operatorname{Cat}((1-r_t)\mathbf{x}_t+r_t\mathbf m),
  &\mathbf{x}_t\ne\mathbf m,\\
  \operatorname{Cat}\!\left(
  \dfrac{\beta_1\mathbf m+\beta_2\mathbf x}{1-\alpha_t}
  \right),&\mathbf{x}_t=\mathbf m,
  \end{cases}
  $$

  其中

  $$
  \beta_1=1-\alpha_{t-1}-r_t\alpha_t,\qquad
  \beta_2=\alpha_{t-1}-(1-r_t)\alpha_t.
  $$
- Sampling Strategy 已列出
  $t_{\rm on}=0.55,t_{\rm off}=0.45,\alpha_{\rm loop}=0.5,r_t=0.02$。
- Implementation Details 重复列出了 256 steps、target MIC 1、
  $\gamma_{\rm MIC}=15$、$\gamma_{\rm pep}=15$ 和三阶段 guidance。

### 2.2 当前 manuscript 仍需修正或补充的地方

| 位置/问题                     | 当前状态                                                                         | 后续处理                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Eq. 9 denominator             | TeX 中出现`1-a_t`，其余均用 $\alpha_t$                                       | 核对并改为$1-\alpha_t$                                                           |
| marginal preservation         | 正文已给 posterior，但没有展开证明                                               | 正文不重复证明；在 Reviewer 1 回复中用两行代数解释                                 |
| pretrained MDLM compatibility | 正文只说 adopted Loop                                                            | Campbell/ReMDM 比较和兼容性解释放 Reviewer 1 回复，不扩写正文                      |
| token eligibility             | 没有准确写出 decoded、non-padding positions 等范围                               | 按实际代码补充                                                                     |
| “2% probability”            | 被写成最终 remask probability                                                    | 改为 unguided/base ReMDM proposal probability                                      |
| “fixed proportion”          | 代码并不固定 token 数                                                            | 删除；实际是 independent categorical draws                                         |
| correction timing             | “early token-level errors”可能被误解成 denoising trajectory 的 early-time 区间 | 可保留原意，但明确写成“token-level errors introduced in earlier denoising steps” |
| exact interval rationale      | 当前只有直觉性陈述                                                               | 等补实验后按结果写，不把历史 pilot 夸大为系统搜索                                  |

### 2.3 实际 sampler 如何选择 remask token

论文生成路径已经由代码核验为：

`main.py::guide_sample_AMP`
$\rightarrow$
`diffusion.py::sample_AMP`
$\rightarrow$
`_cbg_denoise_antibiotic_remdm_loop`

在 loop window 内：

1. 只有当前已经 decoded 的位置可在“保留当前 token”和“回到
   $\texttt{<MASK>}$”之间选择；
2. padding 后的位置被固定为 padding，CLS 最终也被固定，不属于普通 remask 选择；
3. unguided ReMDM base kernel 对每个 eligible token 给出
   $1-r_t=0.98$ 的 keep mass 和 $r_t=0.02$ 的 mask mass；
4. 实际 paper sampler 随后用 D-CBG 的 peptide-classifier 项重加权两个候选，再逐位置做
   categorical sampling。

因此，对于 eligible 位置 $\ell$，实际概率可概念性写成

$$
\begin{aligned}
P(x_{t-1}^{\ell}=\mathbf m\mid\mathbf x_t,y_{\rm pep})
&\propto r_t\exp\{\gamma_{\rm pep}g_\ell(\mathbf m;\mathbf x_t)\},\\
P(x_{t-1}^{\ell}=x_t^\ell\mid\mathbf x_t,y_{\rm pep})
&\propto (1-r_t)\exp\{\gamma_{\rm pep}g_\ell(x_t^\ell;\mathbf x_t)\},
\end{aligned}
$$

其中 $g_\ell$ 是代码使用的 first-order D-CBG classifier approximation。也就是说：

- **base proposal 是随机的、逐 token 的 $r_t=0.02$**；
- **最终 guided remask probability 不是所有位置固定 2%**；
- **没有先固定要 remask 的 token 数，也没有按分子大小设置 quota**；
- **本工作没有使用 ReMDM-conf 的 denoiser-confidence ranking**。

在忽略 guidance 的示意情况下，宽度约为 0.10 的 256-step window 约含 26 次 loop update，
单个 eligible token 至少被 base kernel remask 一次的概率约为
$1-(1-0.02)^{26}\approx 0.41$。这只是帮助解释 correction budget 的计算，不应冒充实际
guided transition probability。

### 2.4 当前历史输出不能回答 peptide proportion

- `outputs/generated_mol_SELFIES-new-test` 当前只保留 BAA-3170 的 518 条 post-filtered records
  （24、306、188 三个文件中的行数）。
- 当前 generation 代码在 SELFIES 完整性和分子有效性过滤后才写出这些文本文件。
- 同名输出可被后续运行覆盖；历史不同 batch 的全部 raw attempts、失败样本和逐步 transitions
  没有完整保存。

因此，不能用“现存有效分子中 classifier-positive 的比例”冒充 Reviewer 2 所问的
“全部生成候选中 peptide vs small molecule 的比例”。新实验必须从 raw attempts 开始记录分母。

### 2.5 关于 reviewer 所称 90%/10%

Methods 列出的过滤前五来源为：

- PubChem：111,378,206；
- SmProt：825,632；
- UniRef：6,972,866；
- UniProt：3,749,540；
- PeptideCLM/CycloPS：10,000,000。

按这些原始来源计数，PubChem 为 83.79%，peptide sources 合计为 16.21%，并非 90%/10%。
Methods 随后称 token-length filtering 和 UniRef/UniProt dedup 后共有 121.6M molecules。
现存 DLM Arrow dataset 只保存 token IDs 和 descriptors，没有 source ID；过滤后的精确来源比例
尚不能直接从该 dataset 恢复。

这与论文 peptide classifier 的 82,795,051-row 数据是两套不同数据，不能混用。classifier
正类的 12.6065% 是来源标签比例，不是 DLM pretraining 的 peptide 比例。

正式回复前应：

1. 从 DLM preprocessing 的上游文件或 producer notebook 恢复过滤后逐来源计数；
2. 如果无法精确恢复，明确报告可验证的过滤前计数和最终总数，不猜测过滤后比例；
3. 不因 reviewer 使用 90%/10% 就直接接受这一数字。

## 3. Reviewer 1 回复中使用的 ReMDM–Campbell 数学解释

本节用于组织 Reviewer 1 response，不计划把 Campbell/ReMDM 的方法比较或下面的完整推导加入
论文正文。正文只保留复现本工作 sampler 所需的 ReMDM posterior、schedule 和 token-selection
说明。

### 3.1 ReMDM 为什么在数学上成立

MDLM 的 absorbing marginal 为

$$
q(\mathbf x_t\mid\mathbf x)
=\operatorname{Cat}\!\left(
\alpha_t\mathbf x+(1-\alpha_t)\mathbf m
\right).
$$

把第 2.1 节的 ReMDM posterior 对 $\mathbf x_t$ 边缘化：

$$
\begin{aligned}
P(\mathbf x_{t-1}=\mathbf x\mid\mathbf x)
&=\alpha_t(1-r_t)+\beta_2=\alpha_{t-1},\\
P(\mathbf x_{t-1}=\mathbf m\mid\mathbf x)
&=\alpha_t r_t+\beta_1=1-\alpha_{t-1}.
\end{aligned}
$$

所以加入 remasking 后仍有

$$
q_r(\mathbf x_{t-1}\mid\mathbf x)
=\operatorname{Cat}\!\left(
\alpha_{t-1}\mathbf x+(1-\alpha_{t-1})\mathbf m
\right),
$$

即每个时刻的 marginal 与原始 MDLM 相同。有效概率还要求

$$
0\le r_t\le
\min\left(1,\frac{1-\alpha_{t-1}}{\alpha_t}\right).
$$

在 Loop plateau 中 $\alpha_{t-1}=\alpha_t=\alpha_\ast$。当
$\alpha_\ast=0.5,r_t=0.02$ 时，unguided kernel 在 decoded 和 masked 两侧均产生约 2% 的
反向交换，维持 50% masked / 50% decoded 的 marginal，同时允许已有 token 被重新预测。

ReMDM 的 NELBO 与 MDLM diffusion loss 的差别是时间权重发生变化；denoiser 的预测对象仍是
clean token $\mathbf x$。因此使用 pretrained MDLM 的 $x_\theta(\mathbf x_t,t)$ 做
inference-time ReMDM sampling 是自然且由 ReMDM 本文直接论证的 plug-and-play 选择。

### 3.2 Campbell et al. (2024) 的对应构造

Campbell 使用 continuous-time masking interpolant。把其时间方向映射到本文的
$\alpha$ 后，同样可写成

$$
p_\alpha(\mathbf x_\alpha\mid\mathbf x)
=\alpha\,\delta_{\mathbf x}+(1-\alpha)\,\delta_{\mathbf m}.
$$

它先构造能产生该 marginal flow 的最小 rate matrix
$R_\alpha^\ast$，再加入满足 detailed balance 的 rate：

$$
p_\alpha(i\mid\mathbf x)R_\alpha^{\rm DB}(i,j\mid\mathbf x)
=p_\alpha(j\mid\mathbf x)R_\alpha^{\rm DB}(j,i\mid\mathbf x),
$$

$$
R_\alpha^\eta=R_\alpha^\ast+\eta R_\alpha^{\rm DB}.
$$

对 mask interpolant，其一个具体 detailed-balance pair 是

$$
R_\alpha^{\rm DB}(\mathbf x,\mathbf m)=\eta,\qquad
R_\alpha^{\rm DB}(\mathbf m,\mathbf x)
=\eta\frac{\alpha}{1-\alpha}.
$$

该附加流的流入和流出相互抵消，所以改变 trajectory 的 jump/stochasticity 数量，却不改变
目标 marginal。其 Euler discretization 中，已 unmasked positions 以
$\Delta t\,\eta$ 被随机、均匀地 remask；masked positions 的额外 unmask rate 带有
$\eta\alpha/(1-\alpha)$ 补偿。

这与 ReMDM 的结构对应为：

- Campbell 的 $\eta\Delta t$ 对应 ReMDM finite-step 的 $r_t$；
- Campbell 的 $\eta\alpha/(1-\alpha)$ compensation 对应 ReMDM masked branch 中为保持
  marginal 增加的 $\alpha r_t/(1-\alpha)$ mass；
- 二者都是“增加可逆 churn，同时保持预先指定的 mask/data marginal”。

### 3.3 真正的差别

| 维度                   | Campbell et al. (2024)                                       | ReMDM                                                                          |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 数学语言               | continuous-time CTMC / rate matrix / detailed balance        | finite-step non-Markovian posterior                                            |
| 自由度                 | stochasticity$\eta$                                        | per-step remask probability$r_t$ 或 $\sigma_t$                             |
| marginal 保持          | detailed-balance flow 不改变 Kolmogorov marginal             | posterior 显式代数保证 MDLM marginal                                           |
| 与现有 MDLM 的接口     | 需要把 denoiser 转成 unconditional rates 并 Euler sampling   | 直接把 pretrained MDLM prediction 代入 posterior                               |
| concrete remask choice | Appendix 实现为对 eligible decoded positions 随机均匀 remask | base schedules 可随机；ReMDM-conf 可按低 denoising confidence 增大 remask 概率 |
| “purity”作用         | 对当前 masked positions 决定优先 unmask 的顺序               | 不是基础 ReMDM 的必需部分                                                      |
| 本论文实际使用         | 未使用                                                       | Loop + constant base$r_t$ + D-CBG peptide reweighting                        |

因此，不能把 Campbell 的 purity 描述成一种“更高级的 remask-token selection”：
其论文中的 purity 排序用于选择 **unmask** 哪些 masked positions，而 concrete remask 仍是随机均匀
选择。ReMDM-conf 才是 ReMDM 中明确按低 denoising confidence 提高 **remask** 概率的可选扩展，
但 ApexOracle 没有使用它。

## 4. Reviewer 1 指定句子的回复逻辑

对

> This assumes that all molecules, regardless of their size or structural complexity, require
> syntactical correction at the exact same phase of the denoising trajectory.

建议直接澄清：

1. 统一 window 只限定**何时允许** reversible correction，不表示每个 molecule 都会被 correction，
   更不表示每个 token 都会在同一步被 correction。
2. 在 unguided base kernel 中，每个 eligible token 独立使用相同的 $r_t$。是否 remask 是
   token-level Bernoulli decision，分子长度不参与该决定，也没有为每个 molecule 预先指定
   correction token 数。
3. 实际 ApexOracle sampler 还会根据当前 partial molecule 的 peptide-classifier guidance，
   分别重加权各位置的 keep-versus-mask probability；所以最终 transition 甚至不是所有 token
   统一的 2%。
4. 当前 window 本身确实是全局超参数，但 reviewer 从这一点推出“所有 molecules require
   syntactical correction at the exact same phase”并不成立。我们只需要用 window sweep 解释为何
   采用该全局计算窗口，无需增加 size/complexity interaction 实验。

## 5. 对当前 Reviewer 2 schedule 回复的核验

当前草稿中的这部分是正确的：

- $r_t=0$ / $0.02$ / $0$ 的 piecewise schedule；
- $t_{\rm on}=0.55,t_{\rm off}=0.45,\alpha(t_{\rm on})=0.5$；
- remasking 使已经 decoded 的 token 可以重新进入 mask state。

但草稿不能原样提交，原因如下：

1. `t_off < t < t_on` 应与代码/Methods 的边界约定统一为
   $t_{\rm off}<t\le t_{\rm on}$；
2. $r_t=0.02$ 是 unguided ReMDM base proposal，不是 peptide-guided sampler 的所有位置最终
   remask probability；
3. 应明确 eligible decoded non-padding positions、CLS/padding 约束和逐位置 categorical draw；
4. “early token-level errors”的原意可以保留，但应避免把 `early` 误读成 early-time window；
   最清楚的写法是“token-level errors introduced in earlier denoising steps”；
5. 还需要说明 masked branch 的 compensation，避免让读者误以为只是单向随机遮盖；
6. 这段只能回答“schedule 是什么”，不能回答“为什么选择该 interval”或“stage 2 是否有效”。

在补实验完成前，Reviewer 2 的 schedule 回复应保持为“定义与机制澄清”；interval rationale 和
effectiveness 数值应留待实验结果补入。

## 6. 联合补实验设计

实验只回答 reviewer 明确提出的两个问题，不增加其他机制性小 ablation。

### 6.1 实验 A：remasking window 如何选择

目标：在固定其他参数的情况下，只改变
$(t_{\rm on},t_{\rm off})$，比较哪个 window 生成的 peptide proportion 更高、clean-model
predicted MIC 更低。

固定：

- paper checkpoint、tokenizer、strain/genome/text embeddings；
- target MIC 1、256 steps、$\alpha_{\rm loop}=0.5$；
- base $r_t=0.02$；
- $\gamma_{\rm MIC}=15,\gamma_{\rm pep}=15$；
- 两个 paper target strains；
- 每个条件相同 seeds 和 attempted-sample count。

建议的最小 window grid：

| 目的            | $(t_{\rm on},t_{\rm off})$ |
| --------------- | ---------------------------- |
| earlier window  | $(0.75,0.65)$              |
| current window  | $(0.55,0.45)$              |
| later window    | $(0.35,0.25)$              |
| narrower window | $(0.525,0.475)$            |
| wider window    | $(0.55,0.25)$              |

这五个条件分别检验 window 的位置和宽度，同时保持其余参数不变。运行前还应核对这些 boundary
在 256-step implementation 中对应的实际 loop-step 数。

每个 window 只比较：

1. peptide proportion；
2. clean MIC reporting model 的 predicted MIC；
3. 作为分母说明所必需的 attempted、completed/valid molecule counts。

不为本 reviewer 问题额外加入 $r_t$、$\alpha_{\rm loop}$、guidance strength、confidence
schedule、diversity 或 molecular-complexity grid。

window 的结论应按两个目标共同解释。例如，不能只因为 peptide proportion 更高就忽略 predicted
MIC 明显变差。当前 window 如果处于两项指标的合理折中区域，就可以据此解释最终选择；如果另一
本次结果应完整展示当时经验比较的选择逻辑：current setting 是否处于 yield 与 predicted MIC
的合理 trade-off 区域，以及相邻位置或更宽窗口带来什么变化。修订中不声称理论唯一最优，也不
改变当年 wet-lab candidates 的生成 provenance。

### 6.2 实验 B：stage-2 peptide correction 是否有效

Reviewer 2 的原句已经把 effectiveness 具体化为“generated candidates 中 peptides vs small
molecules 的比例”。因此 primary answer 是在 current paper setting 下报告这个比例。

为使“correction 有效”不只是用单个最终比例作描述，增加一个最小直接对照：

| 条件                  | current loop/remasking | phase-2 peptide guidance |
| --------------------- | ---------------------: | -----------------------: |
| Full/current          |                   保留 |  $\gamma_{\rm pep}=15$ |
| No peptide correction |                   保留 |   $\gamma_{\rm pep}=0$ |

两组除 phase-2 peptide guidance 外完全相同。比较最终 peptide proportion，即可直接量化
stage-2 peptide correction 带来的增量。不再增加 no-remask、monotone-MDLM、ReMDM-conf 或其他
小 ablation。

如果实现成本很低，可以同时记录同一 full trajectory 在 $t_{\rm on}$ 与 $t_{\rm off}$ 的
peptide-classifier score，作为“loop 前后”的机制说明；它不是独立实验，也不替代最终 molecule
proportion。

### 6.3 peptide proportion 和 MIC 的最小口径

- peptide proportion 必须说明分母：
  - all attempted sequences；
  - completed、chemically valid generated molecules。
- v1 peptide classifier 是 pipeline 实际优化目标，可以保留为 operational score；后续 structure
  audit 已证明它不能作为生成结构的 peptide identity。
- RDKit-valid 和 amide bond 最初只作为透明 QC；后续审计显示 classifier/structure 冲突足以影响
  reviewer 结论，因此现在必须补充独立 canonical-structure-based criterion。仍不能把“至少一个
  酰胺键”单独冒充严格 peptide 定义。
- MIC 只使用论文的 clean reporting checkpoint，报告各条件的 median predicted MIC；必要时再给
  IQR。不要用 noisy guidance regressor 自评。

### 6.4 raw output 的最低保存要求

新实验必须保存所有 attempts，而不是只保存过滤后的成功分子。每条至少记录：

- strain、window/condition、seed 和 sample index；
- raw final token IDs/SELFIES；
- completion/validity；
- peptide classification；
- clean predicted MIC；
- exact config 和 checkpoint/input hashes。

这是回答比例分母和保证运行不覆盖历史输出所必需的记录，不扩展为大型 sampler diagnostics 实验。

## 7. 执行阶段与写作门槛

### 7.0 2026-07-28 冻结并启动的正式协议

以下为已由 task manifest、smoke 日志和实时 GPU 状态验证的事实：

- 冻结 manifest：
  `experiments/remasking_schedule_reviewer/task_manifest.json`，SHA-256
  `84463b4dd09b7c6e6e021468fba47dca850bf26eecbab7847767fe424c07b6e7`。
- 共 36 个独立任务：6 个 condition × 2 个 strain × 3 个 seed；每任务 4 batches ×
  25 raw attempts，共 100 attempts，总分母为 3,600。
- seeds 固定为 `20260728/20260729/20260730`。BAA-3170 使用论文配置长度 368；
  BAA-3197 使用其历史生成配置长度 232。其余共同参数为 256 steps、`r_t=0.02`、
  `alpha_on=0.5`、`gamma_MIC=15` 和 target MIC 1 micromolar。
- 五个 window 为：
  `earlier=0.75--0.65`、`current=0.55--0.45`、
  `later=0.35--0.25`、`narrower=0.525--0.475`、
  `wider=0.55--0.25`。唯一 effectiveness control 是 current window 下
  `gamma_peptide=0`；不增加其他细小 ablation。
- 本机 batch-size smoke 已逐级验证 1、10、25；batch 25 在 H100 上峰值约 45.5 GiB，
  单 batch sampler 用时约 100 秒，因此冻结为每卡一次运行一个 batch-25 task。
- 本机 4 张 H100 各顺序负责 3 个任务，共 12 个任务；node002 8 张 A100 各顺序负责
  3 个任务，共 24 个任务。每个 condition 固定为 2 个 local 和 4 个 node002 tasks，且两个
  strain 都跨 host，避免 condition 与机器混杂。
- 本机正式队列于 2026-07-28 启动并正常完成：12/12 tasks、48/48 batches、
  1,200/1,200 raw attempts。tmux session `remasking_reviewer_local_20260728` 已退出，
  状态文件为 `runs/queue_status_local.json`。
- node002 使用隔离的、与本机关键源码逐文件同 hash 的 producer 快照和 conda-pack 环境；
  BAA-3197 batch-1 A100 smoke 已通过。正式 tmux session 为
  `remasking_reviewer_node002_20260728`，状态文件为 node002 同实验目录下
  `runs/queue_status_node002.json`。原 GPU guard 于启动前停止，全部正式任务退出后恢复。
- 首次简单 round-robin allocation 在任何 task 完成前停止，因为它会造成 condition-host
  混杂；两个 3/4-batch partial task 和旧 queue/log 被保留在
  `superseded_allocation_smoke_20260728/`，明确排除在正式 evaluator 输入之外。
- evaluator 已在 1,000 个本机正式 attempts 的执行中快照上端到端通过：全部得到 v1 clean-input
  peptide probability，647 个 RDKit-valid molecules 全部得到 finite clean-MIC prediction；
  smoke 输出位于被忽略的 `smoke/evaluator_local_validation/`。首次 smoke 暴露并修复了 checkpoint
  Hydra config 的 eager resolver 问题；该修正只影响后处理配置读取，不影响任何 generation task。

### MDLM clean scoring integration（2026-08-10）

`evaluate_remasking_schedule_reviewer.py` 的 clean MIC evaluation 已从动态导入 MDLM 根目录
`judge_generated_mols_MIC.py` bridge，切换为直接调用 `apexoracle_mdlm.scoring` 的
`load_condition_embedding_banks` 与 `load_candidate_mic_regressor`。CLI 仍显式接收 `--mdlm-root`；该路径只用于
定位 sibling submodule 的 `src/`、upstream runtime/config 和本地 ignored assets，不再把 root legacy filename
当作 API。

正式 clean MIC checkpoint、最终 no-copy Generation sampler 的两个真实 token，以及 BAA-3170/BAA-3197 两个
conditions 下，旧 bridge 与 direct API 共四个 logits 均 `torch.equal`，最大绝对差 `0.0`。精确输入/output、
checkpoint hash 和解释边界记录在 `mdlm_scoring_bridge_parity.json`。Focused source contract：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_remasking_schedule_reviewer.py -k canonical_mdlm_scoring_api
```

完整 reviewer test 文件仍需要本地 ignored `analysis/evaluated_attempts.csv` 才能运行 seed-level MIC error-bar
测试；fresh worktree 缺少该资产时的单个 `FileNotFoundError` 不属于本次 bridge 切换回归。

Canonical 入口：

```bash
# 冻结/复核 task manifest
python scripts/reproduce/prepare_remasking_schedule_reviewer_tasks.py \
  --output experiments/remasking_schedule_reviewer/task_manifest.json

# 单任务原始生成（通常由 orchestrator 调用）
CUDA_VISIBLE_DEVICES=0 /home/tianang/anaconda3/envs/mdlm/bin/python \
  scripts/reproduce/run_remasking_schedule_reviewer.py --help

# 每台机器按 manifest 启动一条顺序队列/GPU
/home/tianang/anaconda3/envs/mdlm/bin/python \
  scripts/reproduce/orchestrate_remasking_schedule_reviewer.py --help

# 全部任务完成后，用同一 v1 classifier 和 clean MIC checkpoint 评估
CUDA_VISIBLE_DEVICES=0 /home/tianang/anaconda3/envs/mdlm/bin/python \
  scripts/reproduce/evaluate_remasking_schedule_reviewer.py --help

# 冻结协议与 host/GPU allocation 回归测试
/home/tianang/anaconda3/bin/python -m pytest -q \
  tests/test_remasking_schedule_reviewer.py

# 从冻结 summary 重建历史四面板图（structure audit 后暂不用于 reviewer）
/home/tianang/anaconda3/bin/python \
  scripts/audit/plot_remasking_schedule_reviewer.py

# 保留 median bar 版本，另生成 panel b 为完整 MIC distribution 的 violin 版本
/home/tianang/anaconda3/bin/python \
  scripts/audit/plot_remasking_schedule_reviewer.py --panel-b-style violin

# 作者确认的最终 reviewer 三面板图；独立输出，不覆盖 legacy 图
MPLBACKEND=Agg \
  /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/plot_remasking_structure_qualified_peptides.py

# 如需复现 legacy yield-only 版本
MPLBACKEND=Agg \
  /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/plot_remasking_structure_qualified_peptides.py \
    --layout yield-only
```

`run_remasking_schedule_reviewer.py` 不写外部 sampler 的历史 output 目录。每个 task 独占
`runs/<task_id>/`，逐 batch 原子写出全部 attempted token sequences、SELFIES/SMILES 解码、
RDKit validity 与结构代理，并保存 resolved config、启动日志和 batch SHA-256。原评估的
classifier-positive proportion 使用生成时同一个 v1 peptide classifier 在 clean input 的
`p(peptide)>=0.5`；后续已确认该值不是 structure peptide truth。predicted MIC 使用
`guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth`，只对
RDKit-valid 分子计算，且不用于改变 raw-attempt 分母。

### 7.1 2026-07-29 完成状态

- generation：36/36 tasks、144/144 batches、3,600/3,600 unique raw attempts；
- evaluation：3,600/3,600 v1 peptide probabilities；2,355/2,355 RDKit-valid molecules
  获得 finite clean-MIC prediction；
- current setting 的历史 operational labels 为 213 classifier-positive / 182
  classifier-negative（53.9% / 46.1%），不得解释成真实 peptide/small-molecule composition；
- current vs `gamma_peptide=0` 的 valid classifier-positive yield 为 35.5% vs 33.0%，但
  classifier-positive/all-attempt proportion 仅为 47.7% vs 47.5%，因此 correction 的
  composition effect 应描述为 modest；
- wider `0.55--0.25` 的历史 valid classifier-positive yield 为 48.5%；原
  all-RDKit-valid/classifier-positive predicted-MIC trade-off 需在 structure-qualified subset
  上复核；
- node002 guard 已在实验结束后恢复并复核为每卡约 73.3 GiB、7--8% utilization。

完整表格、strain-specific 结果、reviewer response 含义和产物 SHA-256 见
`experiments/remasking_schedule_reviewer/RESULTS.md`。

历史四面板图及其 exact plotted-data CSV、SHA-256 manifest 位于
`experiments/remasking_schedule_reviewer/figures/`，已标记为 legacy，不再用于 reviewer。图中只呈现冻结 summary 的描述性结果，
不会重新计算 classifier label 或 predicted MIC。默认版本保留 panel b 的 pooled median bar；
`*_violin.*` 版本只把 panel b 换成 log-scale predicted-MIC distribution。panel a 的 error bars
为三个 seed-level pooled rates 的 sample s.d.；panel c 的 reviewer-facing 图只显示
descriptive effect sizes。六个 matched `strain × seed` tasks 的 two-sided exact paired
sign-flip p-values 只保留在 CSV/manifest/内部结果报告中，不在 figure 上标注。

窄结构口径图读取
`analysis/peptide_structure_audit/audited_valid_attempts.csv`、冻结 summary 和 raw SMILES。
B 与卤素明确允许；常见金属候选单列为 manual review，不自动计入 peptide；异常元素直接排除。
默认三面板图另外保留原 window-wise all-RDKit-valid median predicted MIC 和
current-vs-no-guidance direct control；panel a 仅显示同时通过 structure screen 和 SEP-padded
classifier 的严格单系列。三个 panel 单行排列、标题居中，不绘制总标题或灰色 panel
descriptions。panel c 删除 v1 classifier-positive 独立行，在同一个三行 dot-and-interval
坐标区中保留 RDKit-valid 与 peptide yield，并增加 all-RDKit-valid median predicted MIC；
其中 peptide yield 与 panel a 使用相同的 structure screen + SEP-padded classifier
联合定义。yield 误差条为三个 seed-level pooled rates 的 sample s.d.，predicted-MIC
误差条为三个 seed-level pooled median 的 sample s.d.；
唯一图例位于 panel c，且只显示简短组名；`gamma_peptide` 取值在 caption 中解释。panel c 的
5--25 数值区间用显式断轴压缩，每行标签分别标明 `%` 或 $\mu\mathrm{M}$ 单位。panel b 以三个
seed-level pooled median MIC 的 sample
s.d. 为纵向误差条，同时保留 all-RDKit-valid pooled median 作为柱高。完整说明写入同 stem 的
`*_caption.md`。
输出 stem 为 `figures/remasking_structure_qualified_peptides_with_mic_control*`。
该 stem 已于 2026-07-29 被作者确认为 canonical final reviewer figure。`--layout yield-only`
只用于复现 legacy `figures/remasking_structure_qualified_peptides*`，不会被默认三面板图覆盖。
两种 layout 均生成 caption、exact plotted-data CSV 和 manifest。这里的 final 指图稿版本；
联合 peptide criterion 仍属 preliminary，不能在 control-set 验证前称为通用 peptide ground
truth。canonical/legacy 索引及避免二进制重复的存储规则见 `figures/README.md`。

### Phase 0：只读资产冻结

- [ ] 恢复/界定 DLM 过滤后逐来源计数；
- [X] 核对 BAA-3197 historical generation config、两个 strain input embeddings 和 checkpoint；
- [X] 核对 clean MIC reporting checkpoint 与 noisy guidance checkpoint；
- [X] 记录本机和 node002 GPU 可用性并冻结 task ownership；
- [X] 验证 wrapper 可保存 raw attempts；没有修改外部 sampler 或其历史 output；
  如果未来必须改外部 sampler，仍需先获得作者对最小
  instrumentation 的明确授权。

### Phase 1：实现与小规模验证

- [X] 在 Synergy 中建立 canonical launcher、task manifest、evaluator 和协议测试；
- [X] 外部 sampler 以关键文件 SHA-256 和 node002 隔离只读快照固定，所有输出写入新目录；
- [X] 用 batch 1/10/25 smoke 验证无历史覆盖、raw denominator 和 completion hash；
- [X] 验证五个 window 的实际 loop-step boundary，以及 full/no-peptide-correction 两个条件只相差
  phase-2 $\gamma_{\rm pep}$；
- [X] 同步更新根 `AGENTS.md`、`experiments/README.md`、
  `docs/COMPUTE_AND_ASSET_MAP.md` 和本实验 README；本次没有新增 `scripts/audit/` 入口。

### Phase 2：运行 Reviewer 直接要求的实验

- [X] 运行五个固定 window；
- [X] 运行 current full 与 no-peptide-correction 的最小 effectiveness 对照；
- [X] 汇总 peptide proportion、clean predicted MIC 和必要的 attempt/valid denominators；
- [X] 冻结 raw CSV、compact summary、provenance、manifest、输入/代码 SHA-256 和
  reviewer-facing `RESULTS.md`；
- [X] 生成并由作者确认 canonical 三面板 reviewer figure，legacy 图原地保留。

### Phase 3：修改文稿与回复（已完成）

- [X] 修正 Eq. 9 notation、token-selection 描述、Sampling Strategy 和 Implementation Details；
- [X] 在 Reviewer 1 回复中加入 Campbell/ReMDM 数学对照；正文未增加 Campbell related work；
- [X] 在回复中加入 peptide-yield / window-sensitivity 数值，并使用作者确认的 canonical
  三面板 reviewer figure；
- [X] 更新 Reviewer 1 与 Reviewer 2 的三处对应回复，完成的动作均改为完成时态；
- [X] 在临时目录编译并逐页核验 TeX，渲染核验 DOCX；正式论文 PDF 未覆盖；
- [X] 按作者最终决定，从 Results 删除 window/effectiveness 两段细节；这些内容保留在 Methods、
  Supplementary Fig. C4/caption 和 reviewer response；
- [X] 确认正式论文和这三处 reviewer 回复均未展开内部 peptide 判定标准。

## 8. 预期 reviewer 回复结构

### Reviewer 1

1. 感谢 reviewer 指出 interval rationale 与 token-selection 描述不足；
2. 用同一 absorbing marginal 展示 ReMDM 的两行 marginal-preservation；
3. 说明 Campbell 的 detailed-balance CTMC 与 ReMDM finite-step posterior 的对应及区别；
4. 说明 pretrained MDLM 兼容性；
5. 准确说明 base $r_t$、eligible positions 和 D-CBG reweighting；
6. 说明相同的是 per-token base probability / expected fraction，而不是每个 molecule 的固定
   correction token 数；
7. 报告五个 window 的结果和最终选择依据。

### Reviewer 2：domain imbalance / peptide yield

1. 先给经血缘核验的 DLM source composition，不混入 classifier dataset；
2. 报告 all-attempt 和 valid-molecule 分母下的 peptide proportion；
3. 对外直接报告统一口径的 peptide proportion；详细判定规则只保留在内部审计记录；
4. 用 full vs no-peptide-correction 的单一对照量化 stage-2 增益。

### Reviewer 2：$r_t$ schedule

1. 给完整 piecewise equation 和 boundary；
2. 给 $t_{\rm on},t_{\rm off},\alpha_{\rm loop},r_t$；
3. 说明 base proposal、masked-side compensation、guided effective probability；
4. 引用 schedule sweep，而不是只写“chosen empirically”。

## 9. 已解决的原待确认事项

1. peptide 主口径已冻结为窄结构筛选与正确 SEP-padding classifier 的联合标准；其余候选不自动
   二分为 small molecules；
2. raw logging 由 Synergy-side wrapper 完成，没有修改外部 sampler 或历史 outputs；
3. 预算已冻结并完成：每个 `window × strain × seed` 100 attempts，共 3,600 attempts；
4. 保留 wet-lab generation 使用的 current schedule；作者确认原始选择本来就经过了相近的
   empirical window comparison，只是过程和结果没有被完整记录。本次五窗口实验用于把该选择
   过程与 trade-off 逻辑以可复核形式展示出来，不改变原研究实际采用的生成 provenance。

## 10. 验收标准

- 数学说明能明确验证 marginal，而不是只引用方法名；
- Campbell purity 不被错误写成 remask ranking；
- $r_t=0.02$ 不再被写成 guided sampler 的统一最终概率；
- 不再使用“fixed proportion of tokens”；
- peptide yield 有 all-attempt denominator 和独立结构口径；
- interval 结论来自预先固定的五个 window，而不是事后增加或删除条件；
- effectiveness 只使用 full vs no-peptide-correction 的直接对照，不扩展无关 ablation；
- 所有新结果有 task manifest、seed、checkpoint/input/code hash 和唯一输出目录；
- Campbell/ReMDM 数学对照只进入 Reviewer 1 回复，不进入正文 related work；
- 正式论文和对应 reviewer response 已完成相应修改后，才采用 “we have added”表述。

## 11. Reviewer response draft v1（历史交互稿，已迁移）

> **使用边界：** 以下保留定稿过程，不能代替正式 response letter。2026-08-02 已完成论文修改、
> 插图、编译和回复迁移；正式文件及验证记录见第 13 节。

### 11.1 Reviewer 1：global window、token selection 与 Campbell et al. (2024)

> **Reviewer comment**
>
> ApexOracle's error-correcting “remasking” step relies on a hardcoded, static schedule where
> remasking is exclusively activated between diffusion timesteps $t=0.55$ and $t=0.45$.
> This assumes that all molecules, regardless of their size or structural complexity, require
> syntactical correction at the exact same phase of the denoising trajectory. The paper fails to
> describe why choose this time interval and how to choose the remark tokens. Recent paper
> (Campbell et al. 2024. *Generative flows on discrete state-spaces: enabling multimodal flows
> with applications to protein co-design*) provides mathematically decent manner to handle mask
> and remarking.

**Draft response**

We thank the reviewer for identifying that the original manuscript did not explain the empirical
choice of the remasking interval or the token-level remasking rule with sufficient precision. We
agree that $t_{\rm on}=0.55$ and $t_{\rm off}=0.45$ constitute a global, non-adaptive
hyperparameter choice. However, a shared interval specifies when reversible token transitions are
enabled; it does not require every molecule, or every token within a molecule, to be corrected at
the same timestep.

More specifically, ApexOracle uses the Loop sampler of Remasking Discrete Diffusion Models
(ReMDM). For each eligible, already decoded molecular-token position, with padding and fixed
special-token positions excluded, the unguided ReMDM kernel assigns base masses $1-r_t$ and
$r_t$ to retaining the current token and returning it to $\texttt{<MASK>}$, respectively.
These are independent token-level categorical draws; the sampler does not prescribe a fixed
number or proportion of remasked tokens for each molecule, and molecular length is not an input to
this decision. In the ApexOracle sampler, the D-CBG peptide guidance further reweights the
keep-versus-mask alternatives using the current partially decoded molecule. Consequently,
$r_t=0.02$ is the base ReMDM proposal probability inside the loop, not a fixed final remasking
probability shared by all positions. The implemented base schedule is
$r_t=0.02$ for $t_{\rm off}<t\leq t_{\rm on}$ and $r_t=0$ otherwise, with
$\alpha_t=0.5$ held constant during the loop.

The ReMDM construction used here is likewise mathematically decent. With the MDLM absorbing
marginal

$$
q(\mathbf{x}_t\mid\mathbf{x})=
\operatorname{Cat}\!\left(\alpha_t\mathbf{x}+(1-\alpha_t)\mathbf{m}\right),
$$

the ReMDM posterior in Eq. 9 uses
$\beta_1=1-\alpha_{t-1}-r_t\alpha_t$ and
$\beta_2=\alpha_{t-1}-(1-r_t)\alpha_t$. Marginalizing over the two possible states at time $t$
gives

$$
\Pr(\mathbf{x}_{t-1}=\mathbf{x}\mid\mathbf{x})
=\alpha_t(1-r_t)+\beta_2=\alpha_{t-1},
$$

$$
\Pr(\mathbf{x}_{t-1}=\mathbf{m}\mid\mathbf{x})
=\alpha_t r_t+\beta_1=1-\alpha_{t-1}.
$$

Thus, the remasking transition introduces reversible token updates while preserving the prescribed
MDLM mask/data marginal at every step. This is also why ReMDM is a natural choice for our
pretrained MDLM: the denoiser continues to predict the clean token under the same absorbing
marginals and can be used at inference time without retraining the pretrained model.

Campbell et al. (2024) and ReMDM are closely related probability-preserving constructions rather
than a principled-versus-ad-hoc contrast. Campbell et al. formulate the process as a
continuous-time Markov chain and add a detailed-balance rate that increases reversible
mask/unmask transitions without changing the target marginal flow. ReMDM expresses the same core
idea as a finite-step posterior whose compensation terms explicitly preserve the MDLM marginal.
In Campbell et al.'s concrete masking implementation, already unmasked positions are also selected
uniformly at random for remasking. Their purity heuristic prioritizes which currently masked
positions to unmask; it is not a confidence-based rule for selecting remask tokens. Therefore,
Campbell et al. do not provide a molecule-size-adaptive remasking interval or a structurally ranked
remask-token rule that is absent from our sampler.

During the original method development, we selected the $0.55$–$0.45$ interval by empirically
comparing candidate remasking windows, but the original manuscript did not fully report this
selection process or its supporting results. We now present this selection process and rationale
through a controlled evaluation of five windows while fixing the checkpoint, guidance strengths,
base $r_t=0.02$, target strains, sampling steps, seeds, and number of attempts. Each window
comprised 600 raw attempts
(2 strains $\times$ 3 seeds $\times$ 100 attempts). The peptide yields for the earlier
$0.75$–$0.65$, current
$0.55$–$0.45$, later $0.35$–$0.25$, narrower $0.525$–$0.475$, and wider
$0.55$–$0.25$ windows were 2.2%, 3.3%, 5.0%, 2.7%, and 11.0%, respectively. The corresponding
median predicted MIC values among all RDKit-valid outputs were 37.2, 37.9, 44.5, 40.3, and
50.0 $\mu\mathrm{M}$, where lower values are preferred. These results show that the current
window is not a unique optimum: later or wider correction increases peptide yield but is
accompanied by worse predicted activity, whereas the earlier window gives nearly identical
predicted MIC but lower peptide yield. We therefore retain the $0.55$–$0.45$
setting as a reasonable empirical trade-off and report the complete sensitivity analysis rather
than claiming universal optimality across molecular sizes or complexities
(Supplementary Fig. C4).

In the revised Sampling Strategy section, we have defined the complete piecewise schedule,
eligible token positions, independent token-level sampling rule, masked-side compensation, and
guidance-dependent reweighting; and have explicitly described the interval as an empirical global
hyperparameter supported by our sensitivity analysis.

**历史迁移清单（已完成）**

1. 将 Eq. 9 的 `1-a_t` 改为 $1-\alpha_t$；
2. 把 $r_t=0.02$ 写成 unguided/base proposal，不写成 guided final probability；
3. 删除 “predicting a fixed proportion”；
4. 准确定义 eligible decoded non-padding positions 和 special-token constraints；
5. 插入 canonical 三面板图并确定 Supplementary figure 编号；
6. 最后一段已在动作完成后由 “we will” 改成 “we have”。

### 11.2 Reviewer 2：pretraining imbalance 与 stage-2 effectiveness

> **Reviewer comment**
>
> Major domain imbalance in pretraining data: 90% PubChem small molecules, 10% peptides. The
> generative model is heavily biased toward small molecules. The peptide classifier in stage 2
> must overcome this bias within a narrow time window ($t_{\rm on}=0.55$ to
> $t_{\rm off}=0.45$). The effectiveness of this correction is never evaluated: what proportion
> of generated candidates are peptides vs. small molecules?

**Draft response**

We thank the reviewer for requesting a direct evaluation of the stage-2 peptide correction. We
agree that the molecular pretraining corpus is imbalanced toward PubChem and that the original
manuscript did not quantify the correction's effectiveness. We note that the source counts
reported in the Methods correspond, before token-length filtering and cross-source deduplication,
to 111,378,206 PubChem molecules and 21,548,038 molecules from the four peptide sources, i.e.,
83.8% and 16.2%, rather than exactly 90% and 10%. This numerical correction does not alter the
reviewer's qualitative concern.

We therefore performed a direct controlled comparison under the paper's $0.55$–$0.45$
remasking window. The full condition used $\gamma_{\rm peptide}=15$, whereas the control used
$\gamma_{\rm peptide}=0$; all other checkpoints, guidance and sampling parameters, target
strains, seeds, and attempt counts were identical. Each condition comprised 600 raw attempts
(2 strains $\times$ 3 seeds $\times$ 100 attempts), so invalid generations were retained in
the denominator rather than being silently discarded.

With peptide guidance, 395/600 attempts (65.8%) produced RDKit-valid molecules and 20/600 (3.3%)
were peptides; peptides therefore accounted for 20/395 RDKit-valid outputs (5.1%). Without
peptide guidance, 355/600 attempts (59.2%) were RDKit-valid and 10/600 (1.7%) were peptides;
peptides accounted for 10/355 RDKit-valid outputs (2.8%). Thus, enabling stage-2 peptide guidance
doubled the number of peptide candidates (20 versus 10), increased peptide yield from 1.7% to
3.3%, and increased the RDKit-valid yield from 59.2% to 65.8%. Notably, this increase in peptide
yield did not come at the expense of predicted activity: the median clean-model predicted MIC
among RDKit-valid candidates decreased from 56.2 to 37.9 $\mu\mathrm{M}$ when peptide guidance
was enabled. We refer to the remaining valid outputs as other RDKit-valid structures rather than
automatically assigning all of them to the small-molecule class. Overall, the direct control shows
a measurable but incomplete correction of the pretraining-domain bias. We have added these
denominated results and the with-versus-without-guidance comparison to the revised manuscript
(Supplementary Fig. C4).

**历史迁移清单（已完成）**

1. 在 Methods 中定义 denominator，并在 Supplementary Fig. C4/caption 与回复中给出精确计数和比例；
2. 明确 83.8%/16.2% 是过滤和跨来源去重前的 source counts，不冒充 retained 121.6M 的精确比例；
3. 加入 current 与 $\gamma_{\rm peptide}=0$ 的 600-vs-600 direct control；
4. 只把 predicted MIC 称为 clean-model prediction，不写成 wet-lab activity；
5. canonical figure 编号已确认为 C4，“we will add” 已改成 “we have added”。

### 11.3 第一版需要作者重点确认的措辞

1. Reviewer 1 中保留 reviewer 原词 “likewise mathematically decent”，后面立即用 marginal
   preservation 支撑，不与 reviewer 争论形容词；
2. 说明原始 $0.55$–$0.45$ 选择本来就经过了相近的 empirical window comparison，只是原稿
   没有完整展示过程和结果；本次以可复核的五窗口结果明确呈现选择过程和 trade-off 逻辑；
3. Reviewer 2 直接报告 peptide yield 和 RDKit-valid yield，不把其余 valid structures 强制
   二分为已验证的 small molecules；
4. 不写 “statistically significant”，也不加入“不显著”的句子；只报告冻结样本量、effect size
   和精确分母；
5. 不声称 stage-2 guidance 完全克服 pretraining bias，使用 “measurable but incomplete
   correction”。

## 12. Manuscript revision draft v1（历史交互稿，已迁移）

> **状态与边界：** 本节保留迁移前的逐段交互稿。正式内容已于 2026-08-02 写入
> `sn-article.tex`，并以 `\rev{}` 标红；正式文件和编译结果见第 13 节。

### 12.1 修改范围与不修改内容

拟修改五处：

1. 修正 **Remask during generation** 中 Eq. 9 的 $1-a_t$ 为 $1-\alpha_t$，并准确解释两个
   posterior branches；
2. 重写 **Sampling Strategy**，加入完整 piecewise schedule、eligible positions、逐 token
   sampling 和 guidance reweighting；
3. 在 **Implementation Details** 中统一 guided-generation 参数，删除与 Sampling Strategy
   重复或含混的说法；
4. 最初拟在 Results 中加入 window-selection 和 direct-control 结果；作者最终决定不采用，
   以保持 prospective discovery Results 的原有叙事；
5. 在 Supplementary Information 中加入 canonical 三面板图及可复现实验定义。

明确不做两件事：

- 不在正文加入 Campbell et al. 与 ReMDM 的展开比较；该比较只用于直接回答 Reviewer 1；
- 不在正文重复 ReMDM marginal-preservation 的两行证明；正文保留 posterior、参数定义和
  必要的 probability-preserving 说明即可。

### 12.2 `Remask during generation`：拟替换正文

建议保留 Eq. 9 的位置，但用下面的内容替换当前该段和公式后的定义句：

> In the standard MDLM reverse sampler, a position remains unchanged after it has been decoded.
> We therefore adopted the Loop sampler from Remasking Discrete Diffusion Models
> (ReMDM)\cite{remask}, which allows an already decoded molecular token to return to the
> $\texttt{<MASK>}$ state and be denoised again. For a decoded token $\mathbf{x}_t$, let $r_t$
> denote the base probability of proposing a return to $\mathbf{m}$, the mask state. The reverse
> transition is

$$
p_{\theta}\!\left(\mathbf{x}_{t-1}\mid\mathbf{x}_t,t\right)=
\begin{cases}
\operatorname{Cat}\!\left(
\mathbf{x}_{t-1};(1-r_t)\mathbf{x}_t+r_t\mathbf{m}
\right),
& \mathbf{x}_t\neq\mathbf{m},\\[6pt]
\operatorname{Cat}\!\left(
\mathbf{x}_{t-1};
\dfrac{
\beta_1\mathbf{m}+\beta_2\operatorname{NN}_{\theta}(\mathbf{x}_t,t)
}{
1-\alpha_t
}
\right),
& \mathbf{x}_t=\mathbf{m},
\end{cases}
$$

> where
> $\beta_1=1-\alpha_{t-1}-r_t\alpha_t$ and
> $\beta_2=\alpha_{t-1}-(1-r_t)\alpha_t$. For an already decoded position, the first branch
> assigns probability mass to retaining the current token or returning it to the mask state. For
> a currently masked position, the second branch supplies the compensating denoising transition
> required to preserve the prescribed MDLM mask/data marginal. The schedule $r_t$ and its
> interaction with predictor guidance are specified below.

这里不加入完整 marginal proof；Reviewer 1 回复中的两行代数已经足够回答
“mathematically decent manner”。

### 12.3 `Sampling Strategy`：拟完整替换正文

建议用下面三段替换当前从 “We adopted the Loop sampling strategy...” 开始的两段：

> We combined the ReMDM Loop sampler with three-stage predictor guidance. The ApexOracle MIC
> regressor and a peptide classifier trained on the molecular pretraining data were used as the
> two guidance predictors. We denote their guidance strengths by $\gamma_1$ and $\gamma_2$,
> respectively. The base remasking schedule was

$$
r_t=
\begin{cases}
0.02, & t_{\rm off}<t\leq t_{\rm on},\\
0, & \text{otherwise},
\end{cases}
\qquad
t_{\rm on}=0.55,\quad
t_{\rm off}=0.45.
$$

> During the loop interval, the noise schedule was held at
> $\alpha_t=\alpha(t_{\rm on})=0.5$. For $1\geq t>t_{\rm on}$, remasking was disabled and only
> MIC guidance was active $(\gamma_1=15,\gamma_2=0)$. For
> $t_{\rm off}<t\leq t_{\rm on}$, the ReMDM loop and peptide guidance were active
> $(\gamma_1=0,\gamma_2=15)$. For $t\leq t_{\rm off}$, remasking was again disabled, MIC
> guidance was restored, and the remaining masked positions were denoised with the MDLM
> posterior.

> Within each loop step, the base ReMDM proposal was applied independently at every decoded
> molecular-token position within the preset target length. Such a position could either retain
> its current token with base mass $1-r_t=0.98$ or return to $\texttt{<MASK>}$ with base mass
> $r_t=0.02$. Positions beyond the target length were fixed to $\texttt{[PAD]}$, and the first
> position was fixed to $\texttt{[CLS]}$. Before categorical sampling, the peptide-classifier
> term in Eq. 7 reweighted the keep-versus-mask alternatives according to the current partially
> decoded molecule. Therefore, $r_t=0.02$ is the unguided ReMDM proposal probability, not a
> position-independent final probability after guidance. No fixed number or proportion of tokens
> was selected for remasking, and no molecule-level remasking quota was used.

再接一段 interval-selection rationale：

> During method development, we empirically compared five candidate remasking windows with all
> other sampling parameters fixed (Supplementary Fig. C4). We selected the $0.55$–$0.45$ interval
> because it provided a practical trade-off between peptide yield and clean-model predicted MIC.
> We treated the interval as a global sampling hyperparameter rather than a theoretically unique
> or molecule-specific optimum.

这套改法会删除当前正文中的：

- “It is believed that there may be certain periods...”；
- “predicting a fixed proportion of the generated tokens”；
- 把每个位置的最终 guided remask probability 直接写成固定 2% 的含混说法。

### 12.4 `Implementation Details`：拟替换 guided-generation 参数段

建议用下面一段替换当前以 “For guided generation, we set a total of 256 steps...” 开始的段落：

> Guided generation used 256 reverse steps and a target MIC value of 1. The scale parameter
> $\sigma$ in $p(y_1\mid\mathbf{x}_t^{1:L},t)$ was linearly annealed from 0.5 to 0.2 as $t$
> decreased from 1 to 0. Unless otherwise specified, we used
> $t_{\rm on}=0.55$, $t_{\rm off}=0.45$, $\alpha(t_{\rm on})=0.5$, and a base loop-remasking
> probability of $r_t=0.02$. MIC guidance used $\gamma_1=15$ in stages 1 and 3, whereas peptide
> guidance used $\gamma_2=15$ in stage 2. Both predictors used for guidance were trained on
> noised molecular sequences. The clean MIC model was used only for post-generation property
> reporting and did not alter the generated samples.

这里不再重复“middle of the generation process”或“correct mistakes”等直觉性表述；机制已由
Sampling Strategy 精确定义。

### 12.5 Methods 新增：window sensitivity 与 peptide-guidance control

建议紧接 Implementation Details 的 guided-generation 参数段加入一个新的粗体段落
**Remasking-window sensitivity analysis**：

> We evaluated five remasking windows:
> earlier $(t_{\rm on},t_{\rm off})=(0.75,0.65)$, current $(0.55,0.45)$, later
> $(0.35,0.25)$, narrower $(0.525,0.475)$, and wider $(0.55,0.25)$ (Supplementary Fig. C4). We additionally evaluated a
> no-peptide-guidance control using the current window with $\gamma_{\rm peptide}=0$. All other
> checkpoints, guidance parameters, target-strain inputs, and sampling settings were held fixed.
> For each condition, we generated 600 raw attempts across the two prospective target strains,
> *E. coli* AR-0349 (ATCC BAA-3170) and *P. aeruginosa* PA5257
> (ATCC BAA-3197), using three random seeds and 100 attempts per strain and seed. The target
> sequence lengths were 368 and 232 tokens, respectively.

> Peptide yield and RDKit-valid yield used all raw attempts as the denominator. Predicted MIC was
> evaluated only for RDKit-valid outputs using the clean MIC reporting model and was not treated
> as a wet-lab measurement. Window-wise bars report pooled values across both strains and all
> three seeds. Error bars for yields are the sample standard deviation across three seed-level
> pooled rates; error bars for predicted MIC are the sample standard deviation across the three
> seed-level pooled median MIC values.

具体 peptide 判定口径不写入论文；完整结构筛选、SEP-padding、classifier threshold
和元素/金属处理规则继续保留在内部审计文档及可复现代码中。

### 12.6 Results 新增历史草稿（最终未采用）

> **最终决定：** 以下两段未写入最终正文。结果细节只保留在 Methods、Supplementary Fig. C4/
> caption 和 reviewer response，以避免打断 Results 的 prospective discovery 主线。

原交互稿曾建议在 Results 的
**ApexOracle discovers antimicrobials targeting unseen drug-resistant strains**
中，在首次介绍两种 prospective target strains 和 generated candidates 后加入下面两段：

> We next made the empirical selection of the remasking interval explicit by comparing five
> candidate windows under an otherwise fixed generation protocol (Supplementary Fig. C4). The
> peptide yields for the earlier, current, later, narrower, and wider windows were
> 2.2%, 3.3%, 5.0%, 2.7%, and 11.0% of all raw attempts, respectively. The corresponding
> median predicted MIC values among all RDKit-valid outputs were 37.2, 37.9, 44.5,
> 40.3, and 50.0 $\mu\mathrm{M}$. Thus, extending the loop to later or wider intervals increased peptide
> yield but also increased predicted MIC, whereas the earlier interval gave a similar predicted
> MIC but a lower peptide yield. These results support the selected $0.55$–$0.45$ interval as a
> reasonable empirical trade-off rather than a theoretically unique optimum.

> We also evaluated the effect of stage-2 peptide guidance using the same $0.55$–$0.45$ interval.
> With peptide guidance $(\gamma_{\rm peptide}=15)$, 395 of 600 attempts (65.8%) were RDKit-valid
> and 20 of 600 (3.3%) yielded peptide candidates; peptides accounted for 20 of 395 RDKit-valid
> outputs (5.1%). With peptide guidance disabled $(\gamma_{\rm peptide}=0)$, 355 of 600 attempts
> (59.2%) were RDKit-valid and 10 of 600 (1.7%) yielded peptide candidates; peptides accounted for
> 10 of 355 RDKit-valid outputs (2.8%). Notably, the increase in peptide yield did not come at
> the expense of predicted activity: the median clean-model predicted MIC among RDKit-valid
> candidates decreased from 56.2 to 37.9 $\mu\mathrm{M}$ when peptide guidance was enabled.
> Stage-2 peptide guidance also produced more RDKit-valid candidates, although peptides remained
> a minority of the valid outputs.

这里不写 statistical significance，也不把其余 RDKit-valid structures 自动称为 small
molecules。

### 12.7 Supplementary figure：拟加入位置与 caption

建议将 canonical PDF 以单一正式文件名
`Fig_SI_remasking_schedule.pdf` 放入论文目录，在 Supplementary Information 中作为新的
双栏 `figure*`，位置放在当前 **MIC distributions in held-out evaluation sets** 之后。暂用标签
`fig:remasking_schedule`，编译编号已确认为 Supplementary Fig. C4。

建议 caption：

> **Remasking-window sensitivity and peptide-guidance effectiveness.**
> **a,** Peptide yield across five remasking windows. Bars are pooled rates from 600 attempts per
> window; vertical error bars are
> the sample s.d. across three seed-level pooled rates (200 attempts per seed).
> **b,** Pooled median predicted MIC among all RDKit-valid outputs from the clean MIC reporting
> model; vertical error bars are the sample s.d. across the three seed-level pooled median MIC
> values. Lower values indicate stronger predicted activity.
> **c,** Current peptide guidance (blue circles; $\gamma_{\rm peptide}=15$) versus no peptide
> guidance (open squares; $\gamma_{\rm peptide}=0$) under the same $0.55$–$0.45$ window. The three
> rows show RDKit-valid yield, peptide yield, and the pooled median clean-model predicted MIC
> among RDKit-valid outputs; row labels specify percentage or micromolar units. The numerical x
> axis omits the unused 5–25 interval, as indicated by the axis break. Horizontal error bars are
> the sample s.d. across three seed-level pooled rates for yields and across three seed-level
> pooled median MIC values for predicted MIC. Predicted MIC is model-based and is not a wet-lab
> measurement.

### 12.8 正式修改落地状态

- [X] 在正式 TeX 中完成 Eq. 9、Sampling Strategy、Implementation Details 和 Methods；
- [X] Results 中曾加入的两段 remasking 细节已按作者最终决定删除；
- [X] 按 SHA-256 核验并复制 canonical figure 为 `Fig_SI_remasking_schedule.pdf`；
- [X] 加入 Supplementary figure、caption 和交叉引用，编译编号为 Supplementary Fig. C4；
- [X] 用 `\rev{}` 标出新增或修改文字；
- [X] 临时编译并逐页检查公式、revision 标记、figure 浮动位置、编号和引用；
- [X] 把对应 Reviewer 1/2 回复中的计划时态改为完成时态并写入正式 response letter。

## 13. 正式落稿与验证记录（2026-08-02）

### 13.1 已由文件和渲染验证的事实

- 正式 TeX：
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`。
  `Remask during generation`、`Sampling Strategy`、guided-generation `Implementation Details`、
  `Remasking-window sensitivity analysis` 和 Supplementary caption 已同步修改；Results 最终未
  加入 remasking sensitivity/effectiveness 细节。Campbell/ReMDM 的展开比较只进入回复信，
  没有加入正文 related work。
- 正式图：
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Fig_SI_remasking_schedule.pdf`。
  它与 canonical 源图的 SHA-256 均为
  `23ce3a58f82b82f1fb1f458efd08152fbf1284f9b2e2f6b97cd2e1030e9bb847`。
- 在独立临时目录编译成功，输出 31 页；新增图编号为 Supplementary Fig. C4，已有
  Supplementary Fig. C3 和 Appendix Table D1 编号保持不变。日志没有 LaTeX error、未解析引用
  或新增 schedule 公式溢出；正式 `sn-article.pdf` 未覆盖。
- 正式回复：
  `/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/Response to reviewers letter.docx`。
  Reviewer 1 remasking、Reviewer 2 domain imbalance/effectiveness 和 Reviewer 2 `$r_t$ schedule`
  三处回复已更新为完成时态；内部中文 drafting note 已从 `$r_t$` comment 移除。
- DOCX 修改前备份：
  `Response to reviewers letter_before_remasking_revision_20260802.docx`。LibreOffice 独立渲染为
  29 页，相关回答未出现截断、异常居中或分页覆盖。
- 正式论文与上述三处回复均未写入 general amide、residue-like motif、SEP-padding、classifier
  threshold 或元素/金属规则；这些判定细节只保留在本内部 README、`STRUCTURE_AUDIT.md`、代码
  和 manifest 中。
- 本轮 peptide-guidance/remasking reviewer 问题至此暂时收束。代码与紧凑证据位于 Synergy，
  producer 仍只读位于外部 `mdlm` 和 `discrete-diffusion-guidance` checkout，正式文稿位于非 Git
  的 `ApexOracle_cleaned`。发布交接、remote 审计和防膨胀白名单见 `PUBLICATION_HANDOFF.md`；
  reviewer 代码与紧凑证据已按主题直接推送 `Synergy/main`，没有创建 fork 或 PR。

### 13.2 表述边界

- 五窗口结果支持 `$0.55$--$0.45$` 是兼顾 peptide yield 与 predicted MIC 的经验 trade-off，
  不支持其为理论唯一、按分子复杂度自适应或对所有分子全局最优。
- current-vs-control 结果按精确分母描述 effect size；不加入 statistical-significance 声称，也不
  加入“不显著”的表述。
- 37.9 与 56.2 $\mu\mathrm{M}$ 均为 clean-model predicted MIC，不是 wet-lab measurement。
- 作者随后进一步精简了 Sampling Strategy 的 interval-selection 文字：正文只按方法学口吻说明
  五窗口比较和 trade-off 选择，不再使用“为了展示选择过程/支持证据而报告”的 reviewer-response
  口吻；Methods 的 `Remasking-window sensitivity analysis` 已直接引用 Supplementary Fig. C4。
- 作者最终决定删除 Results 中曾新增的两段 window/effectiveness 结果，以保持
  `ApexOracle discovers antimicrobials targeting unseen drug-resistant strains` 的原有主线。正式
  reviewer response 的落点说明已同步改为“protocol 位于 revised Methods，denominated results
  位于 Supplementary Fig. C4”。
