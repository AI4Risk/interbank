# GPL and GSL baseline implementation

GPL denotes graph prompt learning; GSL denotes graph structure learning.

**Baseline adaptation.** 
For the GPL baselines, taking GPF-plus and DPTAP as representative examples, we introduce a prompt-based virtual augmentation generator into each method. The generator learns the correspondence between the original features and observed augmentations on bank nodes $\mathcal{V}_{u}$ through an alignment objective jointly optimized with the original task objective. The learned correspondence is then transferred to nodes without supplementary information, enabling the generation of their virtual augmentations from learnable prompt tokens. The observed and virtual augmentations are subsequently incorporated into the original prompt-learning pipeline for credit assessment.

For the GSL baselines HES-GSL, GSSC, and PROSE, the observed augmentations are incorporated into their native graph structure learning pipelines. The augmentations available on $\mathcal{V}_{u}$ are combined with the original node features, while the corresponding augmentation dimensions of the remaining nodes $\mathcal{V}_{r}$ are zero-filled. Each baseline uses the resulting node representations within its original structure learner to update the complete network, allowing the supplementary signals from $\mathcal{V}_{u}$ to benefit $\mathcal{V}_{r}$ through the learned topology. Credit assessment is then performed using the original financial features and the updated network.

**Experimental configurations.** The configuration table below reports the graph encoder configurations and key method-specific settings used in the baseline experiments. All baselines follow the same rolling-quarter protocol. For the GSL baselines with augmentation, the structure learner takes the concatenation of the original financial features and text embeddings derived from supplementary information as input, while credit assessment is performed using the original financial features over the modified network structure.

## Encoder configurations and hyperparameters

| Method | Encoder configuration | Key hyperparameters |
| --- | --- | --- |
| GPPT | 2-layer GraphSAGE | Prompt centers = 4; prompt orthogonality coefficient = 0.01; negative-edge ratio = 2 |
| MultiGPrompt | 2-layer GCN | Edge-drop ratio = 0.1; link-prediction negatives per node = 200; link contrastive temperature = 1.5 |
| PSP | 2-layer GCN | Contrastive temperature = 0.5; class-prototype prompt nodes = 4 |
| GPF-plus | 2-layer Graph Transformer | Prompt tokens = 5; augmentation-alignment coefficient = 1.0 |
| DPTAP | 2-layer Graph Transformer | Input-level prompt tokens = 5; augmentation-alignment coefficient = 1.0 |
| HES-GSL | 2-layer denoising autoencoder and 2-layer classifier | Top-$k$ neighbor selection $k=20$; feature mask ratio = $1/10$; reconstruction weight $\alpha=10$; homophily weight $\beta=1$; propagation hops $S=5$ |
| GSSC | 2-layer neighborhood MLP, edge sampler, and mixup scaler | Edge-sampler fusion factor $\alpha=0.1$; Relaxed-Bernoulli temperature = 0.5; mixup scale = 0.8 |
| PROSE | 2-layer stage learner GNN | Top-$k$ selection $k=20$; stage ratios = [0.7, 0.7, 0.7]; graph threshold $\epsilon=0.7$ |

In this table, “2-layer” specifies the encoder depth, and “Top-$k$” specifies the number of selected neighbors.
