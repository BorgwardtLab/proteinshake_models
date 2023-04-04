# ProteinShake Evaluation

We build GNN, PointNet, and VoxelNet as baseline methods and perform evaluation on the proteinshake tasks.

## Results

The results are obtained by taking the average of 4 runs with different random seeds.

#### Without pretraing

| Model    | Split  | EC (acc) | pfam | LA | BSD | StructC | StructSIM |
|:---------|:-------|:---------|:-----|:---|:----|:--------|:----------|
| GNN      | rand   | 80.5±0.3 |      |    |     |         |           |
| GNN      | seq    | 75.8±1.2 |      |    |     |         |           |
| GNN      | struct | 61.6±1.5 |      |    |     |         |           |
| PointNet | rand   |          |      |    |     |         |           |
| PointNet | seq    |          |      |    |     |         |           |
| PointNet | struct |          |      |    |     |         |           |
| VoxelNet | rand   |          |      |    |     |         |           |
| VoxelNet | seq    |          |      |    |     |         |           |
| VoxelNet | struct |          |      |    |     |         |           |

#### With pretraining

| Model    | Split  | EC (acc) | pfam | LA | BSD | StructC | StructSIM |
|:---------|:-------|:---------|:-----|:---|:----|:--------|:----------|
| GNN      | rand   | 84.4±1.0 |      |    |     |         |           |
| GNN      | seq    |          |      |    |     |         |           |
| GNN      | struct |          |      |    |     |         |           |
| PointNet | rand   |          |      |    |     |         |           |
| PointNet | seq    |          |      |    |     |         |           |
| PointNet | struct |          |      |    |     |         |           |
| VoxelNet | rand   |          |      |    |     |         |           |
| VoxelNet | seq    |          |      |    |     |         |           |
| VoxelNet | struct |          |      |    |     |         |           |

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