# ProteinShake Evaluation

We build GNN, PointNet, and VoxelNet as baseline methods and perform evaluation on the proteinshake tasks.

## Results

The results are obtained by taking the average of 4 runs with different random seeds.

#### Tasks

| Task                    | Short Name | Metric    |
|:------------------------|:-----------|:----------|
| EnzymeClass             | EC         | accuracy  |
| ProteinFamily           | pfam       | accuracy  |
| LigandAffinity          | LA         | spearmanr |
| BindingSiteDetection    | BSD        | MCC       |
| StructuralClass         | StructC    | accuracy  |
| StructureSimilarity     | StructSIM  | spearmanr |
| ProteinProteinInterface | PPI        | auROC     |


#### Without pretraing (w/o PE)

| Model      | Split  | EC           | pfam         | LA              | BSD             | StructC      | StructSIM       | PPI             |
|:-----------|:-------|:-------------|:-------------|:----------------|:----------------|:-------------|:----------------|:----------------|
| GNN        | rand   | __79.0±0.7__ | __72.8±0.4__ | 0.670±0.019     | __0.721±0.010__ | __49.5±1.2__ | 0.598±0.018     | __0.592±0.009__ |
| GNN        | seq    | __73.7±1.6__ | __63.5±2.0__ | 0.513±0.031     | __0.399±0.020__ | __50.3±0.4__ | __0.663±0.016__ | 0.598±0.003     |
| GNN        | struct | __62.1±2.6__ | __41.1±0.9__ | __0.383±0.034__ | __0.219±0.022__ | __41.5±1.5__ | 0.518±0.010     | 0.500±0.018     |
| PointNet++ | rand   | 71.2±1.6     | 60.9±0.4     | 0.683±0.003     | 0.609±0.006     | 29.3±1.3     |                 | 0.573±0.011     |
| PointNet++ | seq    | 63.5±2.9     | 50.0±0.8     | __0.565±0.011__ | 0.268±0.018     | 27.5±1.2     |                 | __0.611±0.005__ |
| PointNet++ | struct | 60.0±2.1     | 26.9±1.2     | 0.374±0.024     | 0.152±0.019     | 18.8±1.1     |                 | __0.505±0.016__ |
| VoxelNet   | rand   | 65.6±1.2     | 54.3±0.7     | __0.690±0.015__ | N/A             | 22.1±1.4     | __0.620±0.010__ | N/A             |
| VoxelNet   | seq    | 60.0±1.4     | 42.2±1.6     | 0.549±0.031     | N/A             | 21.6±1.2     | 0.644±0.008     | N/A             |
| VoxelNet   | struct | 55.0±3.4     | 18.7±1.0     | 0.355±0.048     | N/A             | 15.8±1.2     | __0.573±0.007__ | N/A             |

#### With pretraining on AlphaFold.SwissProt (w/o PE)

| Model    | Split  | EC           | pfam         | LA              | BSD             | StructC      | StructSIM       | PPI             |
|:---------|:-------|:-------------|:-------------|:----------------|:----------------|:-------------|:----------------|:----------------|
| GNN      | rand   | __81.5±1.0__ | __73.9±0.3__ | __0.723±0.028__ | __0.744±0.003__ | __49.8±0.7__ | __0.640±0.010__ | __0.577±0.010__ |
| GNN      | seq    | __78.5±1.1__ | __65.9±2.0__ | 0.526±0.030     | __0.459±0.008__ | __51.4±1.1__ | __0.699±0.007__ | __0.587±0.010__ |
| GNN      | struct | __64.5±4.0__ | __41.8±1.2__ | __0.428±0.034__ | __0.299±0.023__ | __46.7±0.9__ | __0.587±0.003__ | __0.509±0.011__ |
| PointNet | rand   | 57.2±1.6     | 33.4±0.4     | 0.670±0.030     | 0.484±0.013     | 8.9±0.9      |                 | 0.517±0.013     |
| PointNet | seq    | 49.8±1.3     | 19.7±1.8     | 0.541±0.021     | 0.074±0.070     | 6.9±0.8      |                 | 0.522±0.007     |
| PointNet | struct | 51.6±8.4     | 7.1±0.4      | 0.383±0.038     | 0.141±0.013     | 5.2±0.6      |                 | 0.494±0.008     |
| VoxelNet | rand   | 70.6±1.4     | 59.2±0.5     | 0.705±0.013     | N/A             | 28.4±0.9     | 0.620±0.010     | N/A             |
| VoxelNet | seq    | 61.7±1.6     | 47.7±2.4     | __0.564±0.022__ | N/A             | 26.1±1.1     | 0.680±0.007     | N/A             |
| VoxelNet | struct | 60.5±1.9     | 21.0±0.8     | 0.383±0.043     | N/A             | 18.4±0.3     | 0.571±0.009     | N/A             |


#### Without pretraing

| Model    | Split  | EC           | pfam         | LA              | BSD             | StructC      | StructSIM   | PPI             |
|:---------|:-------|:-------------|:-------------|:----------------|:----------------|:-------------|:------------|:----------------|
| GNN      | rand   | __80.5±0.3__ | __77.4±0.4__ | 0.658±0.009     | __0.691±0.015__ | __55.3±0.6__ | 0.696±0.001 | __0.572±0.006__ |
| GNN      | seq    | __75.8±1.2__ | __72.7±1.1__ | 0.505±0.016     | __0.369±0.017__ | __59.5±1.2__ | 0.744±0.005 | __0.578±0.008__ |
| GNN      | struct | __61.6±1.5__ | __48.8±1.7__ | __0.377±0.016__ | __0.156±0.020__ | __55.9±2.4__ | 0.634±0.006 | __0.499±0.005__ |
| PointNet | rand   | 56.9±1.6     | 40.2±0.9     | __0.696±0.021__ | 0.500±0.007     | 10.0±1.0     |             | 0.512±0.030     |
| PointNet | seq    | 51.1±3.0     | 25.7±1.7     | __0.553±0.033__ | 0.045±0.053     | 8.6±0.9      |             | 0.530±0.016     |
| PointNet | struct | 53.7±2.6     | 9.9±0.8      | 0.364±0.031     | 0.080±0.056     | 3.4±1.4      |             | 0.483±0.029     |
| VoxelNet | rand   | 65.6±1.2     | 54.3±0.7     | 0.690±0.015     | N/A             | 22.1±1.4     | 0.620±0.010 | N/A             |
| VoxelNet | seq    | 60.0±1.4     | 42.2±1.6     | 0.549±0.031     | N/A             | 21.6±1.2     | 0.644±0.008 | N/A             |
| VoxelNet | struct | 55.0±3.4     | 18.7±1.0     | 0.355±0.048     | N/A             | 15.8±1.2     | 0.573±0.007 | N/A             |

#### With pretraining on AlphaFold.SwissProt

| Model    | Split  | EC           | pfam         | LA              | BSD             | StructC      | StructSIM   | PPI             |
|:---------|:-------|:-------------|:-------------|:----------------|:----------------|:-------------|:------------|:----------------|
| GNN      | rand   | __84.1±0.3__ | __78.4±0.3__ | __0.735±0.018__ | __0.740±0.003__ | __57.5±0.8__ | 0.720±0.002 | __0.573±0.009__ |
| GNN      | seq    | __81.2±1.1__ | __70.8±2.4__ | 0.555±0.011     | __0.450±0.010__ | __62.4±1.1__ | 0.767±0.001 | __0.599±0.006__ |
| GNN      | struct | __71.0±1.7__ | __50.7±2.0__ | 0.427±0.021     | __0.299±0.009__ | __61.6±0.9__ | 0.665±0.008 | __0.507±0.004__ |
| PointNet | rand   | 57.2±1.6     | 33.4±0.4     | 0.670±0.030     | 0.484±0.013     | 8.9±0.9      |             | 0.517±0.013     |
| PointNet | seq    | 49.8±1.3     | 19.7±1.8     | 0.541±0.021     | 0.074±0.070     | 6.9±0.8      |             | 0.522±0.007     |
| PointNet | struct | 51.6±8.4     | 7.1±0.4      | __0.383±0.038__ | 0.141±0.013     | 5.2±0.6      |             | 0.494±0.008     |
| VoxelNet | rand   | 70.6±1.4     | 59.2±0.5     | 0.705±0.013     | N/A             | 28.4±0.9     | 0.620±0.010 | N/A             |
| VoxelNet | seq    | 61.7±1.6     | 47.7±2.4     | __0.564±0.022__ | N/A             | 26.1±1.1     | 0.680±0.007 | N/A             |
| VoxelNet | struct | 60.5±1.9     | 21.0±0.8     | 0.383±0.043     | N/A             | 18.4±0.3     | 0.571±0.009 | N/A             |

## TODO

#### Tasks

- [x] EnzymeClass
- [x] LigandAffinity
- [x] BindingSiteDetection
- [x] ProteinFamily
- [x] StructureSimilarity
- [x] StructuralClass
- [x] ProteinProteinInterface
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

#### Supervised training/Finetuning

```bash
python experiments/train.py task=enzyme_class
```

#### Pretraining with masked residue prediction

```bash
python experiments/pretrain_mask_residues.py representation=graph
```
