# ProteinShake Evaluation

We build GNN, PointNet, and VoxelNet as baseline methods and perform evaluation on the proteinshake tasks.

## Results

| Model    | Split  | EC | pfam | LA | BSD | StructC | StructSIM |
|:---------|:-------|:---|:-----|:---|:----|:--------|:----------|
| GNN      | rand   |    |      |    |     |         |           |
| GNN      | seq    |    |      |    |     |         |           |
| GNN      | struct |    |      |    |     |         |           |
| PointNet | rand   |    |      |    |     |         |           |
| PointNet | seq    |    |      |    |     |         |           |
| PointNet | struct |    |      |    |     |         |           |
| VoxelNet | rand   |    |      |    |     |         |           |
| VoxelNet | seq    |    |      |    |     |         |           |
| VoxelNet | struct |    |      |    |     |         |           |


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