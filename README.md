# ProteinShake Evaluation

We build GNN, PointNet, and VoxelNet as baseline methods and perform evaluation on the proteinshake tasks.

## Results

The results are obtained by taking the average of 4 runs with different random seeds.

#### Tasks

| Task                 | Short Name | Metric    |
|:---------------------|:-----------|:----------|
| EnzymeClass          | EC         | accuracy  |
| ProteinFamily        | pfam       | accuracy  |
| LigandAffinity       | LA         | spearmanr |
| BindingSiteDetection | BSD        | MCC       |
| StructuralClass      | StructC    | accuracy  |
| StructureSimilarity  | StructSIM  | spearmanr |

#### Without pretraing

| Model    | Split  | EC       | pfam | LA | BSD | StructC | StructSIM |
|:---------|:-------|:---------|:-----|:---|:----|:--------|:----------|
| GNN      | rand   | 80.5±0.3 |      |    |     |         |           |
| GNN      | seq    | 75.8±1.2 |      |    |     |         |           |
| GNN      | struct | 61.6±1.5 |      |    |     |         |           |
| PointNet | rand   | 56.9±1.6 |      |    |     |         |           |
| PointNet | seq    |          |      |    |     |         |           |
| PointNet | struct |          |      |    |     |         |           |
| VoxelNet | rand   | 65.6±1.2 |      |    |     |         |           |
| VoxelNet | seq    | 60.0±1.4 |      |    |     |         |           |
| VoxelNet | struct | 55.0±3.4 |      |    |     |         |           |

#### With pretraining

| Model    | Split  | EC       | pfam | LA | BSD | StructC | StructSIM |
|:---------|:-------|:---------|:-----|:---|:----|:--------|:----------|
| GNN      | rand   | 84.4±1.0 |      |    |     |         |           |
| GNN      | seq    | 81.2±1.6 |      |    |     |         |           |
| GNN      | struct | 69.3±3.0 |      |    |     |         |           |
| PointNet | rand   | 57.2±1.6 |      |    |     |         |           |
| PointNet | seq    |          |      |    |     |         |           |
| PointNet | struct |          |      |    |     |         |           |
| VoxelNet | rand   | 70.6±1.4 |      |    |     |         |           |
| VoxelNet | seq    | 61.7±1.6 |      |    |     |         |           |
| VoxelNet | struct | 60.5±1.9 |      |    |     |         |           |

## TODO

#### Tasks

- [x] EnzymeClass
- [x] LigandAffinity
- [x] BindingSiteDetection
- [x] ProteinFamily
- [ ] StructureSimilarity
- [ ] StructuralClass
- [ ] GeneOntology

## Installation

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

## Training

```bash
python experiments/train.py task=enzyme_class
```