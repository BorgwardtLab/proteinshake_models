import tempfile

from datasets.atom3d_data import Atom3DDataset
from torch_geometric.data import DataLoader

with tempfile.TemporaryDirectory() as td:
    print(td)
    proteins = Atom3DDataset(root=td, atom_dataset="lba", use_precomputed=False).to_graph(k=5).pyg()
    print(proteins)
    dl = DataLoader(proteins, batch_size=2)
    for b in dl:
        print(b)

