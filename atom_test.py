import tempfile

from datasets.atom3d_data import Atom3DDataset
from torch_geometric.data import DataLoader

# with tempfile.TemporaryDirectory() as td:
# tasks = ('lba', 'psr', 'msp', 'ppi', 'res', 'lep')
tasks = ('res', 'lep')
SPLIT_TYPES = {'lba': ['sequence-identity-30', 'sequence-identity-60'],
               'ppi': ['DIPS'],
               'res': ['cath-topology'],
               'msp': ['sequence-identity-30'],
               'lep': ['protein'],
               'psr': ['year']
               }


for task in tasks:
    proteins = Atom3DDataset(root=task, split_type=SPLIT_TYPES[task][0], atom_dataset=task, use_precomputed=False)
    proteins = proteins.to_graph(k=5).pyg()
    dl = DataLoader(proteins, batch_size=2)

