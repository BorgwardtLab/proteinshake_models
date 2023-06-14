# ProteinShake Models

We build a graph neural network ("Graph"), PointNet++ ("Point"), and a 3D convolution network ("Voxel") as baseline methods and perform evaluation on the [ProteinShake](https://github.com/BorgwardtLab/proteinshake) tasks. See the paper for more information on the architecture of the models.

## Results

Table 1: *Results of baseline models/representations (columns) on the ProteinShake tasks (rows). Best marked in bold, values are given as mean and standard deviation over 4 random seeds. The optimal choice of representation depends on the task. Results were obtained on the random split, see the paper supplemental for the other splits.*

<div align="center">
  
| Task                      | Graph                 | Point                 | Voxel                 |
|:--------------------------|:----------------------|:----------------------|:----------------------|
| Binding Site              | **0.721 $\pm$ 0.010** | 0.609 $\pm$ 0.006     | -         |
| Enzyme Class              | **0.790 $\pm$ 0.007** | 0.712 $\pm$ 0.016     | 0.656 $\pm$ 0.012     |
| Gene Ontology             | **0.704 $\pm$ 0.001** | 0.580 $\pm$ 0.002     | 0.609 $\pm$ 0.004     |
| Ligand Affinity           | 0.670 $\pm$ 0.019     | 0.683 $\pm$ 0.003     | **0.690 $\pm$ 0.015** |
| Protein Family            | **0.728 $\pm$ 0.004** | 0.609 $\pm$ 0.004     | 0.543 $\pm$ 0.007     |
| Protein-Protein Interface | 0.883 $\pm$ 0.050     | **0.974 $\pm$ 0.003** | -         |
| Structural Class          | **0.495 $\pm$ 0.012** | 0.293 $\pm$ 0.013     | 0.221 $\pm$ 0.014     |
| Structure Similarity      | 0.598 $\pm$ 0.018     | **0.627 $\pm$ 0.006** | 0.620 $\pm$ 0.010     |
  
</div>

Figure 2: *Comparison of random, sequence, and structure splits across tasks and representations.
Models generalize less well to sequence and structure splits, respectively.*

<img src="https://raw.githubusercontent.com/BorgwardtLab/ProteinShake_eval/main/figures/2_Splits.svg">

Figure 3: *Relative improvement due to pre-training across tasks and representations. Performance is
substantially improved by pre-training with AlphaFoldDB. Tasks are abbreviated with their initials.
Values are relative to the metric values obtained from the supervised model without pre-training.*

<img src="https://raw.githubusercontent.com/BorgwardtLab/ProteinShake_eval/main/figures/3_Pretraining.svg">


## Installation

One can use `conda`, `mamba` or `pip` to download required packages. The main dependecies are:

```bash
proteinshake
pytorch
pyg
pytorch-lightning
hydra
```

An example for installing `ProteinShake_eval` with `mamba` (similar but faster than `conda`):

```bash
mamba create -n proteinshake
mamba activate proteinshake
mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install pyg -c pyg
mamba install lightning
pip install hydra-core --upgrade
pip install proteinshake
pip install -e .
```

## Model weights

The weights for pre-trained models are available [in the repository](https://github.com/BorgwardtLab/ProteinShake_eval/tree/main/pretrained).
## Training

#### Supervised training/Finetuning

Train a graph neural network from scratch for the Enzyme Class prediction task:
```bash
python experiments/train.py task=enzyme_class representation=graph
```

Finetune a PointNet++ for the Ligand Affinity prediction task:
```bash
python experiments/train.py task=ligand_affinity representation=point_cloud pretrained=true
```

Use `python experiments/train.py` to see more details.

#### Pretraining with masked residue prediction

```bash
python experiments/pretrain_mask_residues.py representation=graph
```

## License

Code in this repository is licensed under BSD-3, the model weights are licensed under CC-BY-4.0.
