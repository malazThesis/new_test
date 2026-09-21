# A Practical Evaluation of Oversmoothing and Oversquashing in Graph Neural Networks

## 1. Introduction

### Motivation
Increasing GNN depth enlarges the receptive field, but deeper message passing can also degrade predictive performance. This thesis investigates when such degradation is associated with oversmoothing, when normalization can mitigate it, and how these observations relate to optimization and graph structure.

### Research Questions

**RQ1.** How does increasing GNN depth affect predictive performance across real-world and controlled graph-learning tasks?

**RQ2.** When is depth-related degradation accompanied by oversmoothing-related representation collapse, and when does PairNorm mitigate both?

**RQ3.** How do source-target distance and structural bottlenecks affect long-range information propagation, and how do normalization and structural interventions alter this behavior?

**RQ4.** Under which conditions are oversmoothing and oversquashing useful explanations for deep-GNN failure, and when are optimization, task, or dataset properties more plausible explanations?

---

## 2. Background

### 2.1 Message Passing and Receptive Fields
References:
- Gilmer et al. 2017
- Kipf and Welling 2017
- Hamilton et al. 2017

**Figure 1:** `figures/fig01_receptive_field.png`

### 2.2 Depth versus Hidden Dimension

**Figure 2:** `figures/fig02_depth_vs_width.png`

### 2.3 Homophily and Heterophily

**Figure 3:** `figures/fig03_homophily_heterophily.png`

References:
- Zhu et al. 2020
- Platonov et al. 2023

### 2.4 Oversmoothing

Discuss:
- repeated propagation
- representation similarity
- Laplacian-smoothing perspective
- distinction between representation collapse and predictive failure
- diagnostic limitations

**Figure 4:** `figures/fig04_oversmoothing.png`

References:
- Li et al. 2018
- Rusch et al. 2023

### 2.5 PairNorm

Explain centering and scaling in PairNorm.

Important:
Center-only and scale-only variants used later are thesis-specific diagnostic ablations.

**Figure 5:** `figures/fig05_pairnorm.png`

Reference:
- Zhao and Akoglu 2020

### 2.6 Oversquashing

Discuss:
- long-range dependencies
- exponentially growing receptive fields
- compression into finite-dimensional representations
- distance
- structural bottlenecks
- topology
- rewiring

**Figure 14:** `figures/fig14_oversquashing_graph_structure.png`

The crossed-ring graph is a thesis-specific schematic.
This figure is conceptual and does not constitute an empirical result.

References:
- Alon and Yahav 2021
- Topping et al. 2022
- Di Giovanni et al. 2023

---

## 3. Experimental Methodology

### 3.1 Real-World Datasets

**Table 1:** `tables/table01_dataset_overview.csv`

15 datasets:
Cora, CiteSeer, PubMed, Roman-Empire, Amazon-Photo,
Amazon-Computers, Coauthor-CS, Coauthor-Physics,
Actor, Chameleon, Squirrel, Cornell, Texas, Wisconsin,
Amazon-Ratings.

Dataset references are given directly in Table 1.

### 3.2 Evaluation Protocol

General rules:
- checkpoints selected using validation accuracy
- hyperparameters selected using validation performance
- test accuracy used only for final reporting
- matched hyperparameters used for mechanism comparisons where required

### 3.3 Oversmoothing Diagnostics

Primary:
- normalized Dirichlet energy
- effective rank / effective-rank ratio

Supporting:
- pairwise cosine distance
- edge cosine distance

Representation diversity is treated as a diagnostic, not as a sufficient condition for predictive performance.

---

## 4. Broad Real-World Screening

### Experimental Design

Phase 1:
- L = 2, 4

Phase 2:
- L = 8, 16, 32

Models:
- GCN
- GAT
- GraphSAGE
- corresponding PairNorm variants

Selection is based on mean validation accuracy.

**Table 2:** `tables/table02_broad_screening_validation_selected.csv`

### Results

**Figure 6a:** `figures/fig06a_realworld_performance_depth.png`

**Figure 6b:** `figures/fig06b_realworld_pairnorm_effects.png`

Main observation:
PairNorm effects are heterogeneous rather than universally beneficial.

---

## 5. Selected Real-World Cases

Selected datasets:
- Amazon-Ratings
- Squirrel
- Roman-Empire
- Actor

Models:
GraphSAGE and GraphSAGE + PairNorm

Depths:
L = 4 and L = 8

Hidden dimension:
128

Learning-rate search:
{0.0003, 0.001, 0.003, 0.005, 0.01, 0.03}

Training:
800 epochs, 10 identical splits.

**Table 3:** `tables/table03_selected4_design.csv`

### Depth Interaction

Depth Loss:
Acc(L4) - Acc(L8)

Difference-in-differences:
(PN_L8 - PN_L4) - (GS_L8 - GS_L4)

Positive values indicate reduced depth degradation under PairNorm.

**Figure 7:** `figures/fig07_selected4_depth_interaction.png`

**Table 4:** `tables/table04_selected4_results.csv`

Main contrast:
- Amazon-Ratings and Squirrel: positive interaction
- Roman-Empire and Actor: negative interaction

### Representation Diagnostics

**Figure 8:** `figures/fig08_selected4_representation_ratios.png`

Figure 8 reports PairNorm / GraphSAGE ratios for:
- normalized Dirichlet energy
- effective-rank ratio

Actor provides a key counterexample:
representation-diversity diagnostics improve while the depth interaction worsens.

---

## 6. Controlled Homophily

### Controlled Graph Construction

Source datasets:
- PubMed
- Roman-Empire

Node features and labels are preserved.

Topology is modified using degree-preserving double-edge swaps on a simple undirected graph.

Preserved:
- number of undirected edges
- complete degree sequence

Targets:
h in {0.1, 0.5, 0.9}

The rewiring target is raw edge homophily.

Adjusted homophily is diagnostic only.

Final graphs are required to remain connected.

Low-label regime:
- 5% train
- 10% validation
- 85% test

**Table 5:** `tables/table05_controlled_homophily_design.csv`

### Endpoint Results

**Figure 9:** `figures/fig09_controlled_homophily_overview.png`

**Table 6:** `tables/table06_controlled_homophily_results.csv`

Accuracy:
test accuracy at validation-selected checkpoint.

Representation diagnostics:
epoch 2400.

### Temporal Dynamics

**Figure 10:** `figures/fig10_temporal_recovery.png`

Contrasting cases:
- PubMed, h = 0.1, L = 8
- Roman-Empire, h = 0.9, L = 8

Metrics:
- normalized Dirichlet energy
- effective-rank ratio

Goal:
distinguish persistent collapse from temporal recovery.

---

## 7. Controlled cSBM Experiments

Two balanced classes.

Nodes:
2000

Features:
64

Target average degree:
20

Target homophily:
{0.1, 0.5, 0.9}

Feature signal:
{0.25, 0.50}

Features consist of isotropic Gaussian noise plus a class-dependent displacement along one random unit direction.

**Table 7:** `tables/table07_csbm_design.csv`

Reference:
Deshpande et al. 2018 for contextual SBM framework.

### Results

**Figure 11:** `figures/fig11_csbm_controlled_summary.png`

Displayed condition:
L = 8.

Results use test accuracy from validation-selected checkpoints.

---

## 8. Learning-Rate Sensitivity

Controlled conditions:
- PubMed h = 0.1
- Roman-Empire h = 0.1
- GraphSAGE / GraphSAGE + PairNorm
- L = 8
- H = 128

Learning rates:
{0.001, 0.003, 0.01, 0.03}

Five graph seeds × five initialization seeds.

**Figure 12:** `figures/fig12_pubmed_roman_lr_sensitivity.png`

Aggregation:
initializations are first averaged within graph seed;
mean ± SD is then computed across graph-level means.

Stars indicate learning rates selected exclusively by mean validation accuracy.

Interpretation:
PubMed GraphSAGE remains near collapse throughout the tested range,
while Roman-Empire exhibits stronger optimization sensitivity.

Do not generalize beyond the tested LR range.

---

## 9. PairNorm Component Ablation

Variants:
- Baseline
- Center only
- Scale only
- Full PairNorm

Center-only and scale-only are thesis-specific diagnostic variants.

Conditions:

PubMed:
- h = 0.1
- LR = 0.01
- epoch 2400

Roman-Empire:
- h = 0.9
- LR = 0.01
- epoch 2400

cSBM:
- h = 0.1
- feature signal = 0.50
- LR = 0.03
- epoch 200

Five graph seeds × five initialization seeds.

**Figure 13:** `figures/fig13_pairnorm_component_ablation.png`

**Table 8:** `tables/table08_pairnorm_component_ablation.csv`

Figure aggregation:
initializations are averaged within graph;
mean ± SD is then shown across graph seeds.

Main observations:
- PubMed: scaling is the dominant successful component.
- Roman-Empire: representation changes do not produce a meaningful positive performance benefit.
- cSBM: centering alone is sufficient for near-perfect performance; scaling also performs strongly.

There is no universal PairNorm component explaining performance across domains.

---

## 10. Oversmoothing Synthesis

Answer RQ2 explicitly.

Main conclusions:

1. Depth degradation is heterogeneous.
2. PairNorm can reduce or increase depth degradation depending on the condition.
3. Persistent representation collapse can coincide with severe predictive failure.
4. Increased NDE or effective rank does not guarantee improved accuracy.
5. Homophily modulates the phenomenon but does not determine it alone.
6. Some apparent failures are strongly optimization-sensitive.
7. The relevant PairNorm mechanism is domain-dependent.

Working criterion:

Oversmoothing is most useful as an explanation when depth degradation,
persistent representation collapse, and successful intervention against that
collapse occur together.

Representation diagnostics alone are insufficient to establish a causal explanation.

---

## 11. Oversquashing

Current status:
theory and methodology only.

No unfinished oversquashing experiments should be used as empirical thesis results yet.

Planned controlled dimensions:
- source-target distance
- structural bottleneck width
- depth
- hidden dimension
- PairNorm
- targeted rewiring

No empirical conclusion until these experiments are completed and audited.

---

## 12. Discussion

Discuss:
- conditional rather than universal mechanisms
- predictive utility versus representation geometry
- topology / homophily / optimization interaction
- limitations of observational real-world comparisons
- value of controlled interventions
- metric limitations
- current status of the oversquashing experiments

---

## 13. Conclusion

Deep-GNN performance degradation should not be interpreted as one universal phenomenon.

PairNorm can prevent severe representation collapse and recover predictive performance in some conditions, while in others it increases representation diversity without improving predictive performance.

Controlled experiments show that homophily, topology, optimization, and normalization interact strongly.

A practical diagnosis therefore requires combining:
- predictive performance
- representation diagnostics
- controlled interventions
- optimization checks
- graph-structural analysis
