""" Extend base ProteinShake dataset to load atom3d datasets.
"""
import os
import os.path as osp

import pandas as pd
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

COORDS_KEY = {'lba': ('atoms_protein'),
              'psr': ('atoms'),
              'ppi': ('atoms_pairs'),
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
        parsed_path = f'{self.root}/{self.__class__.__name__}_{self.atom_dataset}.{resolution}.avro'
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
        indices = []
        i = 0
        for protein_raw_info in tqdm(protein_dfs):
            if i > 10:
                break
            try:
                dfs_atom, dfs_res, lengths_atom, lengths_res  = [], [], [], []
                atom_keys = COORDS_KEY[self.atom_dataset]
                for k in atom_keys:
                    df_atom = protein_raw_info[k]
                    df_atom = df_atom.loc[df_atom['hetero'] == ' ']
                    df_res = df_atom.loc[df_atom['name'] == 'CA']
                    dfs_atom.append(df_atom)
                    dfs_res.append(df_res)
                    lengths_atom.append(len(df_atom))
                    lengths_res.append(len(df_res))

                df_atom = pd.concat(dfs_atom)
                df_res= pd.concat(dfs_res)
                # remove non-standards
                # df_atom = df_atom.loc[df_atom['resname'].isin(three2one)]

                protein = {
                    'protein': {
                        'ID': protein_raw_info['id'],
                        'sequence': ''.join([three2one[r] for r in df_res['resname']]),
                        'group_lengths_atom': lengths_atom,
                        'group_lengths_res': lengths_res,
                        'df_keys': ','.join(atom_keys)
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
            except Exception as e:
                print(f">> Failed on protein {i} with exception {e}.")
                continue
            else:
                proteins.append(protein)
                indices.append(i)
            finally:
                i += 1

        print(">>> Dumping")
        residue_proteins = [{'protein':p['protein'], 'residue':p['residue']} for p in proteins]
        write_avro(residue_proteins, f'{self.root}/{self.__class__.__name__}_{self.atom_dataset}.residue.avro')
        del residue_proteins

        atom_proteins = [{'protein':p['protein'], 'atom':p['atom']} for p in proteins]
        write_avro(atom_proteins, f'{self.root}/{self.__class__.__name__}_{self.atom_dataset}.atom.avro')
        del atom_proteins

        with open(f"{self.root}/indices.txt", "w") as ns:
            ns.write("\n".join(map(str, indices)))
        pass

    def add_protein_attributes(self, protein, protein_raw_info):

        if self.atom_dataset == 'psr':
            protein['protein']['rmsd'] = protein_raw_info['scores']['rmsd']
            protein['protein']['gdt_ts'] = protein_raw_info['scores']['gdt_ts']
            protein['protein']['gdt_ha'] = protein_raw_info['scores']['gdt_ha']
            protein['protein']['tm'] = protein_raw_info['scores']['tm']

            pass
        if self.atom_dataset == 'lba':
            mol = Chem.MolFromSmiles(protein_raw_info['smiles'])
            fp_morgan = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()))
            fp_maccs = list(map(int, MACCSkeys.GenMACCSKeys(mol).ToBitString()))

            protein['protein']['smiles'] = protein_raw_info['smiles']
            protein['protein']['affinity'] = protein_raw_info['scores']['neglog_aff']
            protein['protein']['fp_maccs'] = fp_maccs
            protein['protein']['fp_morgan_r2'] = fp_morgan

            pass
        if self.atom_dataset == 'ppi':
            pass
        if self.atom_dataset == 'msp':
            # see https://github.com/drorlab/atom3d/blob/master/examples/msp/gnn/data.py
            protein['protein']['label'] = int(protein_raw_info['label'])
            # id: '1A22_A_B_EA66A'
            mutation = protein_raw_info['id'].split('_')[-1]
            chain, res = mutation[1], int(mutation[2:-1])
            protein['protein']['mutation_chain'] = chain
            protein['protein']['mutation_res'] = res
            pass

        return protein


if __name__ == "__main__":
    dataset = Atom3DDataset('lba', root='lba', use_precomputed=False)
    print(dataset)
