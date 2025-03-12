import os
import torch
from torch_geometric.data import InMemoryDataset
import pickle

'''
def read_sets(fpath, fold, split_type='warm'): 
    filename = split_type + '.kfold'
    print(f"Reading fold_{fold} from {fpath}")

    with open(os.path.join(fpath, filename), 'rb') as f:
        kfold = pickle.load(f)

    return kfold[f"fold_{fold}"]
'''

class GATADataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass



if __name__ == "__main__":
    GATADataset('data_raw/test')


