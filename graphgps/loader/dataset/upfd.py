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


class UPFD(InMemoryDataset):
    r"""The tree-structured fake news propagation graph classification dataset
    from the `"User Preference-aware Fake News Detection"
    <https://arxiv.org/abs/2104.12259>`_ paper.
    It includes two sets of tree-structured fake & real news propagation graphs
    extracted from Twitter.
    For a single graph, the root node represents the source news, and leaf
    nodes represent Twitter users who retweeted the same root news.
    A user node has an edge to the news node if and only if the user retweeted
    the root news directly.
    Two user nodes have an edge if and only if one user retweeted the root news
    from the other user.
    Four different node features are encoded using different encoders.
    Please refer to `GNN-FakeNews
    <https://github.com/safe-graph/GNN-FakeNews>`_ repo for more details.

    .. note::

        For an example of using UPFD, see `examples/upfd.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        upfd.py>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"politifact"`,
            :obj:`"gossipcop"`).
        feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`,
            :obj:`"bert"`, :obj:`"content"`).
            If set to :obj:`"profile"`, the 10-dimensional node feature
            is composed of ten Twitter user profile attributes.
            If set to :obj:`"spacy"`, the 300-dimensional node feature is
            composed of Twitter user historical tweets encoded by
            the `spaCy word2vec encoder
            <https://spacy.io/models/en#en_core_web_lg>`_.
            If set to :obj:`"bert"`, the 768-dimensional node feature is
            composed of Twitter user historical tweets encoded by the
            `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
            If set to :obj:`"content"`, the 310-dimensional node feature is
            composed of a 300-dimensional "spacy" vector plus a
            10-dimensional "profile" vector.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
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
    url = 'https://data.pyg.org/datasets/upfd_{}.zip'

    id_twitter_mapping_urls = {
        'politifact': 'https://github.com/safe-graph/GNN-FakeNews/raw/main/data/pol_id_twitter_mapping.pkl',
        'gossipcop': 'https://github.com/safe-graph/GNN-FakeNews/raw/main/data/gos_id_twitter_mapping.pkl',
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert name in ['politifact', 'gossipcop']
        assert split in ['train', 'val', 'test']

        self.root = root
        self.name = name
        self.feature = feature
        super().__init__(root, transform, pre_transform, pre_filter)

        assert split in ['train', 'val', 'test']
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


def to_add_somehow():
    # TODO: add this to the actual UPFD dataset (so it gets downloaded directly)
    dataset_name = "politifact"
    data_path = os.path.join(os.getcwd(), "datasets", "UPFD")

    # TODO: to run in current state, must manually download these files and put into correct dir
    # from: https://github.com/safe-graph/GNN-FakeNews/blob/main/data/gos_id_twitter_mapping.pkl
    # and https://github.com/safe-graph/GNN-FakeNews/blob/main/data/pol_id_twitter_mapping.pkl
    #     https://github.com/safe-graph/GNN-FakeNews/raw/main/data/gos_id_twitter_mapping.pkl
    id_mapping_files = {
        'politifact': os.path.join(data_path, "pol_id_twitter_mapping.pkl"),
        'gossipcop': os.path.join(data_path, "gos_id_twitter_mapping.pkl"),
    }
    id_mapping_path = id_mapping_files[dataset_name]
    raw_data_path = os.path.join(data_path, dataset_name, "raw")
    graph_indices_paths = {
        "train": os.path.join(raw_data_path, "train_idx.npy"),
        "val": os.path.join(raw_data_path, "val_idx.npy"),
        "test": os.path.join(raw_data_path, "test_idx.npy"),
    }

    # first we need to get the id mappings from the file:
    with open(id_mapping_path, "rb") as f:
        id_map: dict[int, str] = pickle.load(f)
    # the roots are strings (e.g. politifact1111), the users are integers
    # we can use this to separate the ids by graph
    id_mapping_per_graph: list[list[str]] = []
    for identifier in id_map.values():
        begin_new_graph: bool = False
        try:
            int(identifier)
        except ValueError:
            begin_new_graph = True
        if begin_new_graph:
            id_mapping_per_graph.append([identifier])
        else:
            id_mapping_per_graph[-1].append(identifier)
    # then we need to get the graph indexes to be able to look up the correct ids
    graph_indices: dict[str, list[int]] = {}
    for dataset_mode in ["train", "val", "test"]:
        with open(graph_indices_paths[dataset_mode], "rb") as f:
            graph_indices[dataset_mode] = [int(i) for i in np.load(f)]