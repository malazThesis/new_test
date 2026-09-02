New:
-Additional oversmoothing experiments for deeper GCN and GraphSAGE models
-Comparison of baseline models with PairNorm
-Representation analysis using normalized Dirichlet energy, effective rank, and cosine-distance based measures
-Temporal analysis of how hidden representations change during training
-Controlled homophily experiments on PubMed and Roman-Empire
-Additional graph × initialization experiments to separate graph effects from random initialization
-Learning-rate sensitivity tests for the main GraphSAGE/PairNorm results

Current results:
-The results so far suggest that deep-model degradation is closely related to representation collapse in some settings, especially on PubMed.

-PairNorm often preserves a less collapsed representation and can strongly reduce the performance loss with depth. However, this is not universal: on Roman-Empire the baseline representations can recover during training, and PairNorm gives much smaller or sometimes no performance benefit.

-The results therefore support a more nuanced conclusion: PairNorm can mitigate depth-related representation collapse, but its benefit depends strongly on the graph structure, homophily, and optimization setting.
