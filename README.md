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

#### Without pretraing on AlphaFold.SwissProt

| Model    | Split  | EC           | pfam         | LA              | BSD         | StructC  | StructSIM |
|:---------|:-------|:-------------|:-------------|:----------------|:------------|:---------|:----------|
| GNN      | rand   | __80.5±0.3__ | __77.4±0.4__ | 0.658±0.009     | 0.691±0.015 | 55.3±0.6 |           |
| GNN      | seq    | __75.8±1.2__ | 72.7±1.1     | 0.505±0.016     | 0.369±0.017 | 59.5±1.2 |           |
| GNN      | struct | __61.6±1.5__ | 48.8±1.7     | __0.377±0.016__ | 0.156±0.020 | 55.9±2.4 |           |
| PointNet | rand   | 56.9±1.6     | 40.2±0.9     | __0.696±0.021__ |             |          |           |
| PointNet | seq    | 51.1±3.0     |              | __0.553±0.033__ |             |          |           |
| PointNet | struct | 53.7±2.6     |              | 0.364±0.031     |             |          |           |
| VoxelNet | rand   | 65.6±1.2     | 54.3±0.7     | 0.690±0.015     | N/A         |          |           |
| VoxelNet | seq    | 60.0±1.4     | 42.2±1.6     | 0.549±0.031     | N/A         | 21.6±1.2 |           |
| VoxelNet | struct | 55.0±3.4     | 18.7±1.0     | 0.355±0.048     | N/A         |          |           |

#### With pretraining

| Model    | Split  | EC           | pfam     | LA              | BSD         | StructC  | StructSIM |
|:---------|:-------|:-------------|:---------|:----------------|:------------|:---------|:----------|
| GNN      | rand   | __84.1±0.3__ | 78.4±0.3 | __0.735±0.018__ | 0.740±0.003 | 57.5±0.8 |           |
| GNN      | seq    | __81.2±1.1__ | 70.8±2.4 | 0.555±0.011     | 0.450±0.010 | 62.4±1.1 |           |
| GNN      | struct | __71.0±1.7__ | 50.7±2.0 | 0.427±0.021     | 0.299±0.009 | 61.6±0.9 |           |
| PointNet | rand   | 57.2±1.6     |          | 0.670±0.030     |             |          |           |
| PointNet | seq    | 49.8±1.3     |          | 0.541±0.021     |             |          |           |
| PointNet | struct | 51.6±8.4     |          | __0.383±0.038__ |             |          |           |
| VoxelNet | rand   | 70.6±1.4     | 59.2±0.5 | 0.705±0.013     | N/A         | 28.4±0.9 |           |
| VoxelNet | seq    | 61.7±1.6     | 47.7±2.4 | __0.564±0.022__ | N/A         |          |           |
| VoxelNet | struct | 60.5±1.9     | 21.0±0.8 | 0.383±0.043     | N/A         |          |           |

## TODO

#### Tasks

- [x] EnzymeClass
- [x] LigandAffinity
- [x] BindingSiteDetection
- [x] ProteinFamily
- [x] StructureSimilarity
- [x] StructuralClass
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