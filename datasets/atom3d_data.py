""" Extend base ProteinShake dataset to load atom3d datasets.
"""

import os
import os.path as osp

from proteinshake.utils import save
from proteinshake.datasets import TorchPDBDataset
import atom3d.datasets.datasets as da

ATOM_DATASETS = {'lba', 'smp', 'pip', 'res', 'msp', 'lep', 'psr'}
FOLDERS = {'lba': 'pdbbind_2019-refined-set'}
SPLIT_TYPES = {'lba': ['sequence-identity-30', 'sequence-identity-30'],
               'ppi': ['DIPS'],
               'res': ['cath-topology'],
               'msp': ['sequence-identity-30'],
               'lep': ['protein'],
               'psr': ['year']
               }

class Atom3DDataset(TorchPDBDataset):
    """ Downloads any atom3d dataset into proteinshake.

    Attributes:
        atom_dataset (str): name of a dataset from atom3d ('lba', 'smp', 'pip', 'res', 'msp', 'lep', 'psr')
        split_type (str): the logic used for splitting examples. Default: None gives whole dataset.

    """
    def __init__(self, atom_dataset, root="data", split_type=None, **kwargs):
        self.atom_dataset = atom_dataset
        self.split_type = split_type

        self.raw_folder = FOLDERS[atom_dataset]

        assert atom_dataset in ATOM_DATASETS,\
            f"Invalid atom3d dataset, use one of {ATOM_DATASETS}"
        assert split_type is None or split_type in SPLIT_TYPES[atom_dataset],\
            f"Invalid split, choose from {SPLIT_TYPES[atom_dataset]}"

        super().__init__(**kwargs)
        pass
    def download(self):
        da.download_dataset(self.atom_dataset,
                            osp.join(self.root, "raw", "files"),
                            split=self.split_type
                            )
        pass

    def download_precomputed(self):
        pass

    def get_raw_files(self):
        fname = osp.join(FOLDERS[self.atom_dataset], "data")
        # fname = FOLDERS[self.atom_dataset]
                # else SPLIT_TYPES[self.split_type]
        return osp.join(self.root, "raw", "files", "raw", fname)

    def parse(self):
        if os.path.exists(f'{self.root}/{self.__class__.__name__}.json'):
            return
        protein_dfs = da.load_dataset(self.get_raw_files(), 'lmdb')
        proteins = []
        for p in protein_dfs:
            df = p['atoms_protein'].loc[p['atoms_protein']['name'] == 'CA']
            protein = {'ID': p['id'],
                       'sequence': ''.join(df['resname']),
                       'residue_index': df['residue'].tolist(),
                       'coords': df[['x','y','z']].values.tolist(),
                       'chain': df['chain'].tolist()
                       }
            proteins.append(protein)
            protein = self.add_protein_attributes(protein)
        save(proteins, f'{self.root}/{self.__class__.__name__}.json')
        pass

    def add_protein_attributes(self, protein):
        if self.atom_dataset == 'psr':
            pass
        if self.atom_dataset == 'lba':
            protein['smiles'] = protein['smiles']
            protein['affinity'] = protein['scores']['neglog_aff']
            pass
        return protein

if __name__ == "__main__":
    dataset = Atom3DDataset('psr', root='psr', use_precomputed=False)
    print(dataset)
