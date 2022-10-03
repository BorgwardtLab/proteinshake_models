""" Extend base ProteinShake dataset to load atom3d datasets.
"""
import os
import os.path as osp

from proteinshake.datasets import Dataset
from proteinshake.utils import save, load, write_avro
import atom3d.datasets.datasets as da

three2one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

ALPHABET = 'ARNDCEQGHILKMFPSTWYV'

ATOM_DATASETS = {'lba', 'smp', 'pip', 'res', 'msp', 'lep', 'psr'}

FOLDERS = {'lba': 'pdbbind_2019-refined-set',
           'psr': 'casp5_to_13',
           'ppi': 'DIPS',
           'msp': 'MSP'
           }

SPLIT_TYPES = {'lba': ['sequence-identity-30', 'sequence-identity-30'],
               'ppi': ['DIPS'],
               'res': ['cath-topology'],
               'msp': ['sequence-identity-30'],
               'lep': ['protein'],
               'psr': ['year']
               }

COORDS_KEY = {'lba': 'atoms_protein',
              'psr': 'atoms',
              'ppi': 'atoms_pairs'
              }

class Atom3DDataset(Dataset):
    """ Downloads any atom3d dataset into proteinshake.

    Attributes:
        atom_dataset (str): name of a dataset from atom3d ('lba', 'smp', 'pip', 'res', 'msp', 'lep', 'psr')
        split_type (str): the logic used for splitting examples. Default: None gives whole dataset.

    """
    def __init__(self, atom_dataset, root="data", resolution="residue", split_type=None, **kwargs):
        self.atom_dataset = atom_dataset
        self.split_type = split_type
        self.resolution = resolution

        self.raw_folder = FOLDERS[atom_dataset]

        assert atom_dataset in ATOM_DATASETS,\
            f"Invalid atom3d dataset, use one of {ATOM_DATASETS}"
        assert split_type is None or split_type in SPLIT_TYPES[atom_dataset],\
            f"Invalid split, choose from {SPLIT_TYPES[atom_dataset]}"

        super().__init__(**kwargs)
        pass

    def download(self):
        print("Downloading")
        da.download_dataset(self.atom_dataset,
                            osp.join(self.root, "raw", "files"),
                            split=self.split_type
                            )
        pass

    def download_precomputed(self, resolution="residue"):
        pass

    def get_raw_files(self):
        fname = osp.join(FOLDERS[self.atom_dataset], "data")
        # fname = FOLDERS[self.atom_dataset]
                # else SPLIT_TYPES[self.split_type]
        return osp.join(self.root, "raw", "files", "raw", fname)

    def parse_pdb(self, path):
        if os.path.exists(f'{self.root}/{self.__class__.__name__}.{self.resolution}.avro'):
            return
        protein_dfs = da.load_dataset(self.get_raw_files(), 'lmdb')
        proteins = []
        skipped = 0
        for protein_raw_info in protein_dfs:
            df_res = protein_raw_info[COORDS_KEY[self.atom_dataset]].loc[protein_raw_info[COORDS_KEY[self.atom_dataset]]['name'] == 'CA']
            df_res = df_res.loc[df_res['hetero'] == ' ']
            df_atom = protein_raw_info[COORDS_KEY[self.atom_dataset]]
            df_atom = df_atom.loc[df_atom['hetero'] == ' ']
            try:
                seq = ''.join([three2one[r] for r in df_res['resname']])
            except KeyError as e:
                print(f">> Skipped {skipped} proteins of {len(protein_dfs)} with non-standard amino-acid {e}.")
                skipped += 1
                continue

            protein = {
                'protein': {
                    'ID': protein_raw_info['id'],
                    'sequence': seq,
                },
                'residue': {
                    'residue_number': df_res['residue'].tolist(),
                    'residue_type': [three2one[r] for r in df_res['resname'].tolist()],
                    'chain_id': df_res['chain'].tolist(),
                    'x': df_res['x'].tolist(),
                    'y': df_res['y'].tolist(),
                    'z': df_res['z'].tolist(),
                },
                'atom': {
                    'atom_number': df_atom.index.tolist(),
                    'atom_type': df_atom['fullname'].tolist(),
                    'residue_number': df_atom['residue'].tolist(),
                    'residue_type': [three2one[r] for r in df_atom['resname'].tolist()],
                    'x': df_atom['x'].tolist(),
                    'y': df_atom['y'].tolist(),
                    'z': df_atom['z'].tolist(),
                },
            }
            protein = self.add_protein_attributes(protein, protein_raw_info)
            proteins.append(protein)

        return proteins

    def add_protein_attributes(self, protein, protein_raw_info):
        if self.atom_dataset == 'psr':
            protein['rmsd'] = protein_raw_info['scores']['rmsd']
            pass
        if self.atom_dataset == 'lba':
            protein['smiles'] = protein_raw_info['smiles']
            protein['affinity'] = protein_raw_info['scores']['neglog_aff']
            pass
        if self.atom_dataset == 'ppi':
            pass
        return protein

if __name__ == "__main__":
    dataset = Atom3DDataset('lba', root='lba', use_precomputed=False)
    print(dataset)
