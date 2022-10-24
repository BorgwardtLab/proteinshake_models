from proteinshake.datasets import EnzymeCommissionDataset
from scripts.get_training_info import *

# keep the original dataset and the pyg version separate
d = EnzymeCommissionDataset(root='data/test_carlos')
d_pyg = d.to_point().torch()

# load a dictionary with attributes for each task. e.g. num_classes, task_type
# these can be used for constructing the model
task_dict = get_task_info(d)

# for a pyg dataset, adds a .y attribute with the target
#set_y_pyg(d_pyg, d)
# returns a callable which takes model output as info and g.y
evaluator = get_evaluator(d)
# returns 3 lists of indices deterministically (train, val, test)
splits = get_splits(d)
print(splits)
