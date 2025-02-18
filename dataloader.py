import os
import utils
import torch

import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data


class BrainDataset(InMemoryDataset):
    """
    BrainDataset
        It is brain dataset that storaging brain graph data and its label

    Parameter:
    
    root: str
        the root of the data
    name: str
        the name of the data
    modality: str
        the modality of the data
    """
    def __init__(self, root, name, modality, transform=None, pre_transform=None):
        self.name = name
        self.modality = modality
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        return f'{self.name}.mat'
    
    @property
    def processed_file_names(self):
        return f'{self.name}.pt'

    def download(self):
        # Download to `self.raw_dir`.
        raise NotImplementedError

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        cor_adj, bi_adj, labels = utils.load_data(self.raw_dir, self.name, self.modality)

        for i in range(cor_adj.shape[0]):
            edge_index, edge_attr = utils.getEdgeIdxAttr(cor_adj[i])
            data = Data(num_nodes=cor_adj.shape[1],x=np.abs(cor_adj[i]), y=labels[i], edge_index=edge_index, edge_attr=edge_attr, id=i)
            data_list.append(data)
            

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BRDataset(Dataset):
    """
    BRDataset
        the dataset that storage the brain graph representation and its label
    
    Parameter
    
    x: torch.Tensor
        the brain graph representation
    y: int
        the label
    """
    def __init__(self, x, y, transform=None):
        super(BRDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)


class MyDataset(InMemoryDataset):
    def __init__(self, datalist, transform=None, pre_transform=None):
        super().__init__(transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self.collate(datalist)

