New:
-Additional oversmoothing experiments for deeper GCN and GraphSAGE models

-Comparison of baseline models with PairNorm

-Representation analysis using normalized Dirichlet energy, effective rank, and cosine-distance based measures

-Temporal analysis of how hidden representations change during training

-Controlled homophily experiments on PubMed and Roman-Empire

-Additional graph × initialization experiments to separate graph effects from random initialization

-Learning-rate sensitivity tests for the main GraphSAGE/PairNorm results

-PairNorm component ablation with Baseline, Center-only, Scale-only and full PairNorm.

-Component experiments on PubMed at h=0.1 and Roman-Empire at h=0.9 using 5 graph seeds × 5 initialization seeds.

-Additional matched Baseline and full PairNorm runs with oversmoothing tracking to obtain 25 representation measurements for every component.

-Final matched component masters for PubMed, Roman-Empire and cSBM.

-Final learning-rate robustness analysis for PubMed and Roman-Empire at h=0.1 using lr = 0.001, 0.003, 0.01 and 0.03.

-Final thesis plots for component ablation, component effects, learning-rate sensitivity and representation metrics.

Current results:

-The results so far suggest that deep-model degradation is closely related to representation collapse in some settings, especially on PubMed.

-PairNorm often preserves a less collapsed representation and can strongly reduce the performance loss with depth. However, this is not universal: on Roman-Empire the baseline representations can recover during training, and PairNorm gives much smaller or sometimes no performance benefit.

-The results therefore support a more nuanced conclusion: PairNorm can mitigate depth-related representation collapse, but its benefit depends strongly on the graph structure, homophily, and optimization setting.

-On PubMed at h=0.1, GraphSAGE reaches 39.9% test accuracy. Center-only also reaches 39.9%, Scale-only 80.8%, and full PairNorm 83.3%. Scaling therefore explains most of the PairNorm improvement in this setting.

-PubMed baseline and Center-only representations remain strongly collapsed, while Scale-only and full PairNorm substantially increase normalized Dirichlet energy and effective rank.

-On Roman-Empire at h=0.9, the baseline already reaches 98.4%. Center-only reaches 98.2%, Scale-only 97.9%, and full PairNorm 98.2%. Better representation geometry therefore does not necessarily lead to better predictive performance.

-In the matched cSBM component experiment at h=0.1 and feature signal 0.50, Baseline reaches 54.0%, while Center-only, Scale-only and full PairNorm all reach about 99.7–99.9%. The relative importance of PairNorm components therefore differs between cSBM and PubMed.

-The PubMed collapse remains stable across the tested learning rates, showing that the main result is not caused only by the choice of lr=0.01.

-Overall, PairNorm can prevent harmful representation collapse, but improved representation geometry is not sufficient to guarantee better accuracy across all graph settings.
