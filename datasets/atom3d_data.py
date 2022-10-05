""" Extend base ProteinShake dataset to load atom3d datasets.
"""
import os
import os.path as osp

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

from proteinshake.datasets import Dataset
from proteinshake.utils import save, load, write_avro, unzip_file
import atom3d.datasets.datasets as da

three2one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

ALPHABET = 'ARNDCEQGHILKMFPSTWYV'

ATOM_DATASETS = {'lba', 'smp', 'ppi', 'res', 'msp', 'lep', 'psr'}

FOLDERS = {'lba': 'pdbbind_2019-refined-set',
           'psr': 'casp5_to_13',
           'ppi': 'DIPS',
           'msp': 'MSP'
           }

SPLIT_TYPES = {'lba': ['sequence-identity-30', 'sequence-identity-60'],
               'ppi': ['DIPS'],
               'res': ['cath-topology'],
               'msp': ['sequence-identity-30'],
               'lep': ['protein'],
               'psr': ['year']
               }

COORDS_KEY = {'lba': 'atoms_protein',
              'psr': 'atoms',
              'ppi': 'atoms_pairs',
              'msp': ('original_atoms', 'mutated_atoms')
              }

class Atom3DDataset(Dataset):
    """ Downloads any atom3d dataset into proteinshake.

    Attributes:
        atom_dataset (str): name of a dataset from atom3d ('lba', 'smp', 'ppi', 'res', 'msp', 'lep', 'psr')
        split_type (str): the logic used for splitting examples. Default: None gives whole dataset.

    """
    def __init__(self, atom_dataset,
                       **kwargs):
        self.atom_dataset = atom_dataset
        self.raw_folder = FOLDERS[atom_dataset]

        assert atom_dataset in ATOM_DATASETS,\
            f"Invalid atom3d dataset {atom_dataset}, use one of {ATOM_DATASETS}"

        super().__init__(**kwargs)
        pass

    def download(self):
        print("Downloading")
        da.download_dataset(self.atom_dataset,
                            osp.join(self.root, "raw", "files"),
                            )
        pass

    def download_precomputed(self, resolution='residue'):
        """ Downloads the precomputed dataset from the ProteinShake repository.
        """
        parsed_path = f'{self.root}/{self.__class__.__name__}.{resolution}.avro'
        if not os.path.exists(parsed_path):
            print(">>> Did not find precomputed data, downloading and parsing.")
            self.start_download()
            print(">>> Parsing")
            self.parse()
        pass

    def get_raw_files(self):
        fname = osp.join(FOLDERS[self.atom_dataset], "data")
        # fname = FOLDERS[self.atom_dataset]
                # else SPLIT_TYPES[self.split_type]
        return osp.join(self.root, "raw", "files", "raw", fname)

    def parse(self):
        protein_dfs = da.load_dataset(self.get_raw_files(), 'lmdb')
        proteins = []
        pairs = []
        skipped = 0
        for protein_raw_info in tqdm(protein_dfs):
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

        residue_proteins = [{'protein':p['protein'], 'residue':p['residue']} for p in proteins]
        atom_proteins = [{'protein':p['protein'], 'atom':p['atom']} for p in proteins]
        print(">>> Dumping")
        write_avro(residue_proteins, f'{self.root}/{self.__class__.__name__}.residue.avro')
        write_avro(atom_proteins, f'{self.root}/{self.__class__.__name__}.atom.avro')
        pass

    def add_protein_attributes(self, protein, protein_raw_info):
        
        if self.atom_dataset == 'psr':
            protein['protein']['rmsd'] = protein_raw_info['scores']['rmsd']
            protein['atom']['rmsd'] = protein_raw_info['scores']['rmsd']
            pass
        if self.atom_dataset == 'lba':
            mol = Chem.MolFromSmiles(protein_raw_info['smiles'])
            fp_morgan = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()))
            fp_maccs = list(map(int, MACCSkeys.GenMACCSKeys(mol).ToBitString()))

            protein['protein']['smiles'] = protein_raw_info['smiles']
            protein['protein']['affinity'] = protein_raw_info['scores']['neglog_aff']
            protein['protein']['fp_maccs'] = fp_maccs
            protein['protein']['fp_morgan_r2'] = fp_morgan

            protein['atom']['smiles'] = protein_raw_info['smiles']
            protein['atom']['affinity'] = protein_raw_info['scores']['neglog_aff']
            protein['atom']['fp_maccs'] = fp_maccs
            protein['atom']['fp_morgan_r2'] = fp_morgan
            pass
        if self.atom_dataset == 'ppi':
            pass
        if self.atom_dataset == 'msp':
            # see https://github.com/drorlab/atom3d/blob/master/examples/msp/gnn/data.py
            protein['protein']['label'] = int(protein_raw_info['label'])
            # id: '1A22_A_B_EA66A'
            mutation = protein_raw_info['id'].split('_')[-1]
            chain, res = mutation[1], int(mutation[2:-1])
            orig_idx = self._extract_mut_idx(orig_df, mutation)
            mut_idx = self._extract_mut_idx(mut_df, mutation)
            pass

        return protein

    def _extract_mut_idx(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        return idx

if __name__ == "__main__":
    dataset = Atom3DDataset('lba', root='lba', use_precomputed=False)
    print(dataset)
