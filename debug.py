from proteinshake.datasets import EnzymeCommissionDataset
from proteinshake.tasks import *

task = RetrieveTask(root='data/tm')
ds = task.dataset.to_graph(eps=5).pyg()

for (graph1, protein_dict1), (graph2, protein_dict2)  in ds[task.train_ind[:5]]:
    print(task.target(protein_dict1, protein_dict2))
