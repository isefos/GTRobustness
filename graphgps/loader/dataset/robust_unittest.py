import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import pickle

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, cumsum


# Copied from UPFD, not implemented yet, WIP 


class RobustnessUnitTest(InMemoryDataset):
    r"""The Robust Unittest dataset from <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust/>
    <https://github.com/LoadingByte/are-gnn-defenses-robust/blob/master/unit_test/sketch.py>

    Includes Cora ML and Citeseer with different splits. Also includes precomputed perturbations.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"cora_ml"`,
            :obj:`"citeseer"`).
        split (int): (0 to 4) Will load the dataset with the corresponding split.
            (default: :obj:`0`)
        pert_scenario (str, optional): None, "evasion", or "poisoning". Defines 
            which perturbation to load (if any). (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    url = 'https://github.com/LoadingByte/are-gnn-defenses-robust/raw/master/unit_test/unit_test.npz'

    def __init__(
        self,
        root: str,
        name: str,
        split: int = 0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        raise NotImplementedError
        assert name in ['cora_ml', 'citeseer']

        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

        self.split = int(split)
        assert self.split in [0, 1, 2, 3, 4]
        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed', self.feature)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
            'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz',
            'id_twitter_mapping.pkl',
        ]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = download_url(self.url.format(self.name), self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)
        download_url(self.id_twitter_mapping_urls[self.name], self.raw_dir, filename='id_twitter_mapping.pkl')

    def process(self):
        x = sp.load_npz(
            osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
        x = torch.from_numpy(x.todense()).to(torch.float)

        edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        with open(osp.join(self.raw_dir, 'id_twitter_mapping.pkl'), "rb") as f:
            twitter_id_mapping: dict[int, str] = pickle.load(f)
            node_idx = list(twitter_id_mapping.keys())
            assert node_idx == list(range(len(node_idx)))
        twitter_ids = np.zeros(len(twitter_id_mapping), dtype=np.int64)
        for i, twitter_id in enumerate(twitter_id_mapping.values()):
            if not twitter_id[0].isdigit():
                # is_root, make it negative to identify later
                twitter_id = '-' + ''.join(c for c in twitter_id if c.isdigit())
            twitter_ids[i] = int(twitter_id)
        twitter_ids = torch.from_numpy(twitter_ids)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice,
            'twitter_ids': node_slice, 
        }

        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(x=x, edge_index=edge_index, y=y, twitter_ids=twitter_ids)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            torch.save(self.collate(data_list), path)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')
