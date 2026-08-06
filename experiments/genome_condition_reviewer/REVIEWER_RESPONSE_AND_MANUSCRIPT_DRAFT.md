# Genome-condition reviewer response and manuscript revision record

> **Approved and implemented on 2026-08-06.**
> This file records the agreed response and manuscript edits. The new validation figure was
> assigned Supplementary Fig. C6, the three new software citations were verified against
> Google Scholar and the primary journal records, and the text below was transferred to the
> formal TeX, bibliography, and response DOCX.

## 1. Reviewer comment

> The manuscript states that bacterial whole-genome sequences are embedded and incorporated
> as conditional inputs. However, given the length and complexity of bacterial genomes, this
> process presumably requires fragmentation, pooling, and/or substantial compression. The
> authors should clarify how complete genomes were processed, how fragment-level embeddings
> were aggregated, and what final representation was used as the model condition. More
> importantly, the authors should provide evidence that this compression preserves
> biologically relevant information, including resistance determinants, mobile genetic
> elements, and strain-specific genomic signatures. It would also be important to demonstrate
> that the resulting embeddings can distinguish closely related strains, rather than merely
> capturing species-level differences.

## 2. Implemented reviewer response

We thank the reviewer for raising this important point. Processing, aggregation, and the
final model condition are described in the revised Methods,
“Genome Embeddings” and “Molecule–Strain Knowledge Fusion.” Genome-assembly FASTA files were
processed using 11,000-nt windows at a 10,000-nt step. Each saved window was passed through
Evo-2-40B, and its nucleotide-level layer-46 hidden states were length-normalized mean-pooled
to produce one 8,192-dimensional fragment vector. The fragment vectors were stacked
**without any further cross-fragment pooling** and used directly as the genome-derived
key–value bank for cross-attention. Thus, compression occurs within each 11-kb fragment, but
the model does not collapse all saved fragments into a single fixed-size strain vector.

We would like to clarify a premise underlying the last part of the comment. ApexOracle
does not produce a single fixed-size genome vector per strain. Its genome condition is the
fragment-indexed matrix

$$
\mathbf{E}=[\mathbf{e}_1,\ldots,\mathbf{e}_M]\in\mathbb{R}^{M\times d_g},
$$

where \(M\) is the number of saved fragments for that strain and \(d_g=8{,}192\).

To test directly whether the within-fragment mean pooling preserves strain-level sequence variation,
we added a homologous-fragment analysis across all eligible bacterial embeddings
(Supplementary Fig. C6a). We first identified nearest same-species strain pairs using
whole-genome average nucleotide identity (ANI), which measures the mean nucleotide identity
across genomic regions that can be aligned between two assemblies. Candidate strain pairs
were required to have ANI \(\geq95\%\) and aligned fractions \(\geq50\%\) in both
directions; the aligned fractions quantify how much of each assembly participates in the
alignment and therefore complement the identity measured by ANI. Among the 166 strain pairs
that ultimately contributed homologous fragments, ANI had a median of 99.34% (range,
95.51%–100%), and the smaller of the two directional aligned fractions had a median of 91.65%
(range, 72.42%–100%). Thus, the analyzed pairs generally shared high nucleotide identity
across most of both assemblies. We then compared mutual-best homologous 11-kb fragments within
the genomic segments actually encoded. These 166 strain pairs from 53 species yielded 6,625
homologous fragment pairs, including 1,976 sequence-identical pairs and 4,649 pairs containing
sequence variation. All 4,649 variable fragment pairs had non-zero cosine distance.
Across the variable fragments, sequence divergence and embedding cosine distance were
strongly positively associated (Spearman \(\rho=0.695\)). Importantly, the association was
maintained after restricting the analysis to the most closely related strain pairs with
whole-genome ANI \(\geq99\%\) (4,156 variable fragments; \(\rho=0.714\)). In addition,
for each homologous pair, we used a reproducibly sampled different-index fragment from the
genome containing the homolog as a within-genome reference. In 99.94% of comparisons, the
true homologous pair was closer in embedding space than this reference pair. These results
demonstrate that the frozen representation responds systematically to sequence variation
between closely related, same-species strains, rather than only separating species.

We agree that whether the downstream predictor uses this resolution is a related but distinct
question. The strain-wise generalization benchmark holds out strains within species
(Fig. 2c; seven-model ensemble \(R^2=0.5814\)), and the cross-attention case study provides
complementary qualitative evidence: the same peptide, conditioned on two same-species
*E. coli* strains, attended to different strain-unique surface-polysaccharide regions and yielded
divergent activity predictions. We have softened the original wording so that these analyses
are not presented as an isolated causal test of the genome channel. The new homologous-fragment
analysis is the direct test of representation-level strain discrimination; the benchmark and
attention analysis provide downstream context. We consider this more faithful to the way the
unpooled fragment matrix is actually used than clustering an artificial single pooled vector
that is never supplied to ApexOracle.

On preserving “resistance determinants, mobile genetic elements, and strain-specific
signatures,” each 11-kb fragment spans approximately ten average-length bacterial genes and
therefore represents gene-cluster-scale context rather than a single gene. Reducing the
window to approximately 1 kb would increase the number of cross-attention key–value pairs by
roughly an order of magnitude. Because the genome-derived key–value bank is retained without
cross-fragment pooling, this would increase both key–value storage and the principal
downstream genome cross-attention compute and memory costs by approximately the same order of
magnitude. At the selected resolution, we do not claim single-gene attribution or a complete
catalogue of resistance determinants or mobile elements.

We therefore tested whether information associated with these annotations remains present in
the fragment embeddings after nucleotide-level mean pooling, using deliberately simple
linear readouts rather than a complex task-specific decoder. Using the 264 bacterial genomes
for which the saved windows, FASTA/GenBank sequence and record order, and saved tensor shapes
were exactly compatible, we assigned conservative
fragment labels to 96,716 encoded fragments from existing GenBank gene, product, and feature
annotations. Because negatives for each probe were sampled only from genomes containing at
least one corresponding positive fragment, the AMR-associated evaluation set comprised 7,302
fragments from 249 genomes, whereas the mobile-element-associated evaluation set comprised
44,730 fragments from 263 genomes. We trained two independent L2-regularized logistic
regressions on the frozen fragment vectors, using five-fold
cross-validation grouped by genome so that overlapping fragments from the same genome never
appeared in both training and test sets. For AMR-associated fragments, pooled out-of-fold
AUPRC was 0.203 versus a positive prevalence of 0.167 in its evaluation set, and AUROC was
0.578. For mobile-element-associated fragments, pooled out-of-fold AUPRC was 0.446 versus a
prevalence of 0.198, and AUROC was 0.741 (Supplementary Fig. C6b,c). We interpret these
results conservatively: the simple linear readout detects a weak but above-baseline
AMR-associated signal and a more substantial mobile-element-associated signal. The labels
were derived from a conservative dictionary applied to existing GenBank annotations and do
not constitute a complete resistome or mobile-element catalogue.

The attention case study remains complementary to these quantitative controls. It shows
attention concentrating on strain-unique, multi-gene regions with biologically interpretable
surface-access functions—an O-antigen biosynthesis region versus a capsule biosynthesis
region—rather than establishing single-gene attribution to a canonical resistance determinant.
These are not necessarily canonical resistance genes, but they are relevant to antimicrobial
access; strain-level MIC differences can arise from surface and permeability biology rather
than resistance genes alone.

Together with the fragment-level ablation in Fig. 2c, in which replacing Evo-2 fragment
embeddings with coarse k-mer composition vectors reduced \(R^2\) by 11.6%, the new analyses
support the narrower conclusion that biologically meaningful annotation-associated and
strain-level sequence information is retained in the encoded fragment representations.

## 3. How this modifies the current response rather than replacing it

The implemented response retains the original response’s three-part logic:

1. **Processing and final condition:** retains the 10,000/11,000-nt fragmentation,
   within-fragment mean pooling, no cross-fragment pooling, and direct cross-attention
   key–value bank description.
2. **Closely related strains:** retains the explanation that ApexOracle does not use one
   fixed-size strain vector and retains the strain-wise benchmark and attention example, but
   adds the direct homologous-fragment experiment and stops calling the downstream analyses
   an isolated test of genome-channel use.
3. **AMR/MGE and biological content:** retains the gene-cluster-scale interpretation,
   computational rationale, surface-polysaccharide case study, and k-mer ablation, while
   adding held-out probes that keep all fragments from the same genome in one fold and
   explicitly reporting that the AMR signal is weak.

Necessary corrections to the current wording are:

- Replace “nothing in the pipeline compresses a strain’s genome” with “nucleotide states are
  mean-pooled within each fragment, but no pooling is performed across saved fragments.”
- Do not state that the strain-wise benchmark and attention case study alone prove that the
  genome embeddings distinguish closely related strains. Present the homologous-fragment
  analysis as the direct evidence that the frozen embeddings preserve within-species sequence
  variation, and retain the strain-wise benchmark and attention case study only as
  complementary evidence from the downstream model.
- Correct the strain-wise ensemble result from \(R^2=0.579\) to the manuscript value
  \(R^2=0.5814\).
- Remove the unsupported claim that 1-kb windows necessarily increase Evo-2
  “self-attention cost” by 100-fold. Retain the defensible approximately tenfold increase in
  fragment count, key–value storage, and the principal downstream genome cross-attention
  compute and memory costs.
- Replace “prevent genes from being truncated” with “provide a 1-kb overlap that reduces
  boundary truncation.”
- Replace “gene-level representations are critical” with the narrower “fragment-level
  contextual representations provide additional predictive value over coarse k-mer
  composition.”

## 4. Implemented manuscript modifications

These edits were applied to the formal TeX on 2026-08-06.

**Citation-preservation rule for implementation:** retain every existing citation attached to
text that remains in the manuscript, and include the corresponding primary citation whenever
a newly added sentence names a method, software package, model, or external data resource.
Any new citation key introduced below must be added to `sn-bibliography.bib` at the same time
as the formal TeX edit.

### 4.1 Architecture overview

**Current sentence requiring correction**

> A genomic encoder built on Evo-2 transforms a pathogen’s complete genome sequence into
> dense embeddings that capture resistance determinants and strain-specific genomic
> signatures.

**Implemented replacement**

> A genomic encoder built on Evo-2\cite{evo2} transforms encoded genomic fragments into a
> fragment-indexed representation that retains local sequence context, including measurable
> signals associated with annotated resistance determinants and mobile genetic elements, as
> well as strain-specific genomic signatures.

### 4.2 Methods: “Genome Embeddings”

Apply the following edits to the existing `Genome Embeddings` block; this is not a replacement
of the equations or the entire Methods subsection.

1. Replace the prose beginning “Pathogen genomes downloaded from ATCC...” and ending
   “...embedding of the entire genome” (the paragraph before the equations) with the text
   below.
2. Keep the two existing equations defining \(\mathbf{e}_i\) and \(\mathbf{E}\).
3. In the prose introducing the equations, replace “Suppose the genome yields \(M\)
   fragments” with “Suppose the saved genome condition contains \(M\) encoded fragments,”
   and replace “full-genome embedding” with “fragment-indexed genome condition.”
4. Keep the existing post-equation explanation that \(\mathbf{E}\) is not pooled across
   fragments and is used directly as the genome key–value bank.
5. Keep the existing detailed explanation of the fixed \(10^{14}\) rescaling, including its
   empirical magnitude audit and the fact that it was fixed across training, validation, and
   test splits.

**Replacement for the pre-equation prose**

> Genome assemblies were supplied as FASTA files and processed using a sliding-window
> implementation with a window length of 11,000 nt and a step of 10,000 nt.
> The 1,000-nt overlap reduces the likelihood that a feature crossing a window boundary is
> represented only as a truncated sequence.
>
> The 11-kb window spans approximately ten average-length bacterial genes and was selected as
> a computationally tractable gene-cluster-scale condition. Reducing the window length to
> approximately 1 kb would increase the number of fragment key–value pairs by roughly an
> order of magnitude. Because no pooling is performed across fragments, this would increase
> key–value storage and the principal downstream genome cross-attention compute and memory
> costs by approximately the same order of magnitude. This representation therefore targets
> fragment-level biological context and is not intended for single-gene attribution.
>
> Each encoded fragment was passed independently through Evo-2-40B\cite{evo2}, and
> nucleotide-level hidden states were extracted from layer 46. The choice of layer 46 was guided by the Evo-2
> feature-extraction tutorial, which uses a late, non-final Hyena-short block from the 7B model
> for embedding extraction. Consistent with the empirical practice that near-top, non-final
> hidden states often provide higher-quality general-purpose representations than final
> output representations, we selected the analogous near-top, non-final Hyena-short layer in
> Evo-2-40B. The nucleotide-level hidden states were
> length-normalized mean-pooled to obtain one \(d_g=8{,}192\)-dimensional vector per fragment.
> No pooling was performed across fragment vectors.

Also remove the sentence claiming use of Evo-2’s 1-million-nt context to contextualize each
11-kb fragment. Each saved representation was computed from an individual fragment, so the
statement does not describe additional sequence context used for these embeddings.

### 4.3 Methods: “Genome-representation validation,” placed immediately after “Genome Embeddings”

Place this as a separately labelled paragraph immediately after `Genome Embeddings` within
the same Methods area. The adjacent placement keeps all genome-representation material
together, while the separate label distinguishes how the production embeddings were created
from how the new reviewer-requested validation analyses were performed.

> To test whether the frozen fragment representations preserve sub-species sequence
> variation (Supplementary Fig. C6a), we identified nearest same-species bacterial strain
> pairs using skani\cite{shaw2023skani}, requiring
> whole-genome average nucleotide identity (ANI) \(\geq95\%\) and aligned fractions
> \(\geq50\%\) in both directions. ANI measures mean nucleotide identity within aligned
> regions, whereas aligned fraction measures how much of each assembly participates in those
> alignments. We
> deduplicated reciprocal selections into unordered pairs. Within the saved encoded genomic
> segments, we aligned full-length 11-kb windows
> in both directions using minimap2\cite{li2018minimap2} (`asm5`, `-c`, `--eqx`, `-N 5`) and
> retained
> same-orientation, mutual-best one-to-one matches with at least 80% alignment coverage at
> each end and mapping quality \(\geq20\). Global sequence divergence was calculated as the
> edlib\cite{sosic2017edlib} Needleman–Wunsch edit distance divided by 11,000, and pairs with
> divergence \(>5\%\)
> were excluded. Embedding difference was measured as cosine distance after the same fixed
> \(10^{14}\) rescaling used by ApexOracle. Spearman correlation was calculated across
> variable fragments, with a prespecified analysis restricted to strain pairs with
> whole-genome ANI \(\geq99\%\). As a distance-scale control, for each homologous pair, we also
> measured the distance from the first fragment to a reproducibly sampled different-index
> fragment from the genome containing its homolog. This provided a within-genome reference
> distance for a comparison in which the fragment correspondence was broken.
>
> To test whether the frozen fragment embeddings retained information associated with
> antimicrobial resistance (AMR) and mobile elements after nucleotide-level mean pooling
> (Supplementary Fig. C6b,c), we started from 264 bacterial genomes for which the saved
> windows, FASTA/GenBank sequence and
> record order, and embedding tensor shape were exactly compatible (96,716 fragments). A
> conservative frozen dictionary was applied to existing GenBank feature types, gene names,
> and product descriptions to label fragments
> overlapping AMR-associated or mobile-element-associated annotations. For each label, all
> positive fragments and at most five negative fragments per positive within each
> positive-bearing genome were included, yielding 7,302 fragments from 249 genomes for the
> AMR-associated probe and 44,730 fragments from 263 genomes for the mobile-element-associated
> probe. Separate L2-regularized logistic regressions
> (`C=1`, `class_weight="balanced"`, `liblinear`) were trained on the frozen 8,192-dimensional
> fragment embeddings after fixed \(10^{14}\) rescaling. These deliberately simple linear
> readouts test whether annotation-associated signals can be recovered from the frozen
> embeddings without a nonlinear or task-specific decoder. Five-fold stratified
> cross-validation was grouped by genome ID so that fragments from one genome could not occur
> in both training and test sets. We report pooled out-of-fold AUPRC and AUROC, the positive
> prevalence of each evaluation set as the AUPRC random baseline, and fold-level mean ± sample s.d. No probe
> hyperparameters were tuned.

### 4.4 Results: concise paragraph after the strain-wise benchmark

Insert this paragraph immediately after the current strain-wise benchmark paragraph ending
with the Fig. 2c k-mer ablation result, and before the exact-peptide-overlap sensitivity
paragraph.

> To test whether the fragment representation retained biologically relevant information
> beyond species identity, we performed two representation-level validations (Supplementary
> Fig. C6). Across 4,649 variable homologous 11-kb fragment pairs from closely related
> same-species strains, sequence divergence correlated with embedding cosine distance
> (Spearman \(\rho=0.695\); \(\rho=0.714\) for strain pairs with whole-genome ANI
> \(\geq99\%\)). Simple linear readouts also detected AMR-associated and
> mobile-element-associated signals, with AUPRCs of 0.203 and 0.446 compared with
> evaluation-set prevalences of 0.167 and 0.198, respectively. These results indicate that the
> frozen fragment embeddings retain strain-level sequence variation and annotation-associated
> information.

### 4.5 Existing strain-wise Results and Fig. 2c caption

Revise the current strong wording:

> replacing our Evo-2-based genome embeddings with the k-mer composition vectors ... reduced
> strain-wise \(R^2\) by 11.6%, confirming that fine-grained, gene-level genomic
> representations are critical ...

to:

> replacing the Evo-2 fragment embeddings with k-mer composition vectors reduced strain-wise
> \(R^2\) by 11.6%, consistent with additional predictive value from fragment-level contextual
> representations relative to coarse genome-composition features.

Apply the same change in the Fig. 2c caption. Do not describe an 11-kb vector as a gene-level
representation.

### 4.6 Attention-analysis Results

Retain the O-antigen/capsule case study, but revise the final interpretation so that attention
is explicitly hypothesis-generating:

> These strain-specific attention patterns provide a biologically plausible example of how
> different encoded genomic regions may contribute to divergent model outputs. These
> attention patterns are hypothesis-generating and should not be interpreted as evidence that
> the highlighted genomic regions caused the different model predictions.

Avoid “the model learned” as a causal conclusion unless supported by an intervention.

### 4.7 Supplementary Fig. C6 caption

Use the current reproducible caption as the starting point:

> **Encoded genome-fragment representations retain sub-species variation and
> annotation-associated signals. a,** Relationship between global sequence divergence and
> cosine distance for 4,649 variable mutual-best homologous 11-kb fragment pairs from 165
> nearest same-species strain pairs. Blue circles denote 4,156 fragments from 116 strain pairs
> with whole-genome average nucleotide identity (ANI) ≥99%, and grey crosses denote the
> remaining 493 fragments. Identical fragment pairs are omitted. Spearman correlations were 0.695 across all variable fragments and 0.714 within
> the ANI ≥99% subset. **b,** Five-fold held-out AUPRC for simple linear readouts of
> AMR-associated and mobile-element-associated fragment annotations; all fragments from each
> genome were kept in the same fold. Points denote individual folds, diamonds the fold mean ±
> sample s.d., and dashed lines the evaluation-set prevalence. Pooled out-of-fold AUPRC was
> 0.203 and 0.446, respectively. **c,** AUROC for the same held-out predictions. Points and
> diamonds are defined as in **b**; the dashed line denotes random AUROC of 0.5. Pooled
> out-of-fold AUROC was 0.578 and 0.741, respectively. Labels were derived from existing
> GenBank annotations and do not constitute complete resistome or mobile-element catalogues.

### 4.8 Bibliography updates applied with the formal manuscript edit

The following three verified software-paper entries were added to the formal manuscript's
`sn-bibliography.bib`.
The existing `evo2` citation key must be retained rather than duplicated.

```bibtex
@article{shaw2023skani,
  author  = {Shaw, Jim and Yu, Yun William},
  title   = {Fast and robust metagenomic sequence comparison through sparse chaining with skani},
  journal = {Nature Methods},
  year    = {2023},
  volume  = {20},
  pages   = {1661--1665},
  doi     = {10.1038/s41592-023-02018-3}
}

@article{li2018minimap2,
  author  = {Li, Heng},
  title   = {Minimap2: pairwise alignment for nucleotide sequences},
  journal = {Bioinformatics},
  year    = {2018},
  volume  = {34},
  number  = {18},
  pages   = {3094--3100},
  doi     = {10.1093/bioinformatics/bty191}
}

@article{sosic2017edlib,
  author  = {{\v{S}}o{\v{s}}i{\'c}, Martin and {\v{S}}iki{\'c}, Mile},
  title   = {{Edlib}: a {C/C++} library for fast, exact sequence alignment using edit distance},
  journal = {Bioinformatics},
  year    = {2017},
  volume  = {33},
  number  = {9},
  pages   = {1394--1395},
  doi     = {10.1093/bioinformatics/btw753}
}
```

## 5. Resolved implementation decisions

1. The validation figure is Supplementary Fig. C6; panel references are C6a and C6b,c.
2. The 99.94% homologous-versus-different-fragment control is retained in the reviewer response
   and Methods but omitted from the concise main-text Results paragraph.
3. The strain-wise benchmark and attention case study remain as downstream context in the
   response; the homologous-fragment experiment is identified as the direct
   representation-level test.
4. The conservative labels “AMR-associated” and “mobile-element-associated” are used for the
   probes, without upgrading them to complete resistance-determinant or mobile-element
   catalogues.
5. The formal TeX, bibliography, Supplementary Fig. C6 asset, and response DOCX were updated
   and independently rendered on 2026-08-06.
