# Scripts for evaluating pre-training on ProteinShake datasets

We import proteinshake as a dependency and build pre-training and evaluation scripts here.


## Setup

```bash
pip install -r requirements.txt
```

## Usage

To launch an example, call the main from the atom3d dataset class defined in `atom3d_data.py`

```
python datasets/atom3d_data.py
```

That module contains a class `Atom3DDataset` which extends `proteinshake.datasets.PDBDataset`.

The `Atom3DDataset` constructor accepts two main options (`atom_dataset (str)`, and `split_type (str)`). 

Select which dataset to use with the `atom_dataset` argument.

* `atom_dataset`: `['lba', 'smp', 'pip', 'res', 'msp', 'lep', 'psr']`

The dataset keys represent the following tasks:

| Key | Task | Type | Info |
| ----| -----| -----|------|
| `'lba'`| Ligand Binding Affinity | Protein-level Regression | Predict affinity for an input protein and small molecule |
| `'ppi'`| Protein Protein Interaction | Residue-level Binary Classification | Predict whether a pair of residues interacts or not |
| `'res'`| Residue Identity | Residue-level Multi-class Classification | Predict identity or residue given surrounding residues |
| `'msp'`| Mutation Stability Prediction | Residue-level Regression | Predict folding energy effect from residue mutation |
| `'lep'`| Ligand Efficacy Prediction | Protein-level Binary Classification | Predict whether a small molecule activates or inactivates a protein |
| `'psr'`| Protein Structure Ranking | Protein-level Regression | Predict the RMSD between given structure and ground state structure |

Select the logic for splitting with the `split_type` argument.


* `split_type`: `None` or (see dictionary below)

| Dataset | Split type |
| --------|------------|
| `'lba'` | `'sequence-identity-30', 'sequence-identity-60'`|
| `'ppi'` | `'DIPS'` |
| `'res'` | `'cath-topology'` |
| `'msp'` | `'sequence-identity-30'` |
| `'lep'` | `'protein'` |
| `'psr'` | `'year'` |

Example:

Create a pytorch geometric graph from an Atom3D dataset.
And 

```python
from datasets.atom3d_data import Atom3DDataset
proteins = Atom3DDataset(root="./data").to_graph(k=5).pyg()
```

The returned object is a list of pyg graphs that can be fed to a loader.

