import torch
import math


class WeightedIndexSampler:
    def __init__(
        self,
        weighted_idx: None | torch.Tensor,
        zero_idx: None | torch.Tensor,
        weight: int,
        max_index: int,
        output_device,
    ):
        """
        """
        assert isinstance(weight, int)
        assert weight > 1
        assert isinstance(max_index, int)
        if weighted_idx is not None and zero_idx is not None:
            n_indices = weighted_idx.size(0) + zero_idx.size(0)
            n_unique_indices = torch.unique(torch.cat((weighted_idx, zero_idx), dim=0)).size(0)
            if not n_indices == n_unique_indices:
                raise ValueError("Can't have duplicated indices or indices be both weighted and zero!")

        if weighted_idx is not None:
            weighted_idx, _ = torch.sort(weighted_idx, dim=0)
            # assert (weighted_idx[1:] - weighted_idx[:-1] > 0).all()
            assert weighted_idx[-1] <= max_index
            self.weighted_idx_padded = torch.cat(
                (weighted_idx, torch.tensor([-1], dtype=torch.int64)), dim=0,
            )
            self.weighted_idx = self.weighted_idx_padded[:-1]
        else:
            self.weighted_idx = None

        if zero_idx is not None:
            zero_idx, _ = torch.sort(zero_idx, dim=0)
            # assert (zero_idx[1:] - zero_idx[:-1] > 0).all()
            assert zero_idx[-1] <= max_index
            self.zero_idx_padded = torch.cat(
                (zero_idx, torch.tensor([-1], dtype=torch.int64)), dim=0,
            )
            self.zero_idx = self.zero_idx_padded[:-1]
        else:
            self.zero_idx = None

        self.output_device = output_device
        self.max_index: int = max_index
        self.n_w: int = weighted_idx.size(0) if weighted_idx is not None else 0
        self.n_z: int = zero_idx.size(0) if zero_idx is not None else 0
        self.w_minus_1: int = weight - 1
        self.n_total: int = self.max_index + self.w_minus_1 * self.n_w - self.n_z + 1
        assert self.n_total < 2 ** 63 - 1  # If we sample values of 64 bit integers

    def sample(self, num_samples: int):
        """
        maps a uniform sample to the weighted distribution using binary search(es)
        """
        values = torch.randint(
            low=0,
            high=self.n_total,
            size=(num_samples, ),
            dtype=torch.int64,
        )

        if self.weighted_idx is None and self.zero_idx is None:
            return values.to(device=self.output_device)

        weighted_idx_sample = torch.zeros_like(values, dtype=torch.int64) - 1

        l = torch.zeros_like(values, dtype=torch.int64)
        r = torch.zeros_like(values, dtype=torch.int64) + self.max_index
        step_size = torch.ones_like(values, dtype=torch.int64)

        max_steps = math.ceil(math.log2(self.n_total))
        for _ in range(max_steps):

            c = l + ((r - l) // 2)
            step_size[:] = 1

            if self.weighted_idx is not None:
                n_w_before = torch.searchsorted(
                    sorted_sequence=self.weighted_idx,
                    input=c,
                    right=False,
                )
                step_size[self.weighted_idx_padded[n_w_before] == c] += self.w_minus_1
            else:
                n_w_before = 0

            if self.zero_idx is not None:
                n_z_before = torch.searchsorted(
                    sorted_sequence=self.zero_idx,
                    input=c,
                    right=False,
                )
                step_size[self.zero_idx_padded[n_z_before] == c] = 0
            else:
                n_z_before = 0

            c_values_min = c + n_w_before * self.w_minus_1 - n_z_before
            c_values_max = c_values_min + step_size

            value_is_smaller_mask = values < c_values_min
            value_is_larger_mask = values >= c_values_max
            # assert torch.all(~torch.logical_and(value_is_smaller_mask, value_is_larger_mask))
            value_reached_mask = ~torch.logical_or(value_is_smaller_mask, value_is_larger_mask)

            r[value_is_smaller_mask] = c[value_is_smaller_mask] - 1
            l[value_is_larger_mask] = c[value_is_larger_mask] + 1

            unreached_mask = weighted_idx_sample < 0
            weighted_idx_sample[unreached_mask] = torch.where(value_reached_mask, c, -1)

            if value_reached_mask.all():
                assert torch.all(weighted_idx_sample >= 0)
                return weighted_idx_sample.to(device=self.output_device)

            remaining_mask = ~value_reached_mask
            l = l[remaining_mask]
            r = r[remaining_mask]
            # assert (r >= l).all()
            step_size = step_size[remaining_mask]
            values = values[remaining_mask]

        raise Exception("Sampling (binary search) should have concluded, something went wrong...")
    

def get_connected_sampling_fun(
    allow_existing_graph_pert: bool,
    is_undirected: bool,
    n_ex_edges: int,
    n_ex_nodes: int,
    n_new_nodes: int,
    device,
):
    """
    n_ex_edges is already precalculated (depends on is_undirected)
    """
    n_block_edges = n_ex_nodes * n_new_nodes
    if allow_existing_graph_pert:
        if is_undirected:
            # indices stay the same, but max number is capped to exclude the non-connected
            n_total = n_ex_edges + n_block_edges
            map_fun = lambda x: x

        else:
            # remap the indices that belong to 'second block'
            n_total = n_ex_edges + 2 * n_block_edges
            n_threshold = n_ex_edges + n_block_edges
            offset = n_ex_edges + n_block_edges

            def map_fun(x):
                mask = x >= n_threshold
                x[mask] += ((x[mask] - offset) // n_ex_nodes) * n_new_nodes
                return x

    else:
        if is_undirected:
            # increment each row by a specific number (cumsum)
            n_total = n_block_edges
            row_offset = torch.cumsum(torch.arange(n_ex_nodes-1, -1, -1, device=device), dim=0)
            
            def map_fun(x):
                x += row_offset[x // n_new_nodes]
                return x

        else:
            n_total = 2 * n_block_edges
            # TODO: increment first block by (row_num+1)*n_ex_nodes
            # second block row num with offset by n_block_edges + global offset -> 
            raise NotImplementedError

    def sampling_fun(n):
        return map_fun(torch.randint(n_total, (n, ), device=device))
    
    return sampling_fun
    

if __name__ == "__main__":
    max_index = 12
    weighted_idx = torch.tensor([7, 1, 5], dtype=torch.int64)
    zero_idx = torch.tensor([8, 9, 3], dtype=torch.int64)
    w = 3
    sampler = WeightedIndexSampler(
        weighted_idx=weighted_idx,
        zero_idx=zero_idx,
        weight=w,
        max_index=max_index,
        output_device=torch.device("cpu"),
    )
    n = 10000
    s = sampler.sample(n)
    sample_values, sample_frequencies = torch.unique(s, sorted=True, return_counts=True)
    d = {i: 0.0 for i in range(max_index + 1)}
    for v, f in zip(list(sample_values), list(sample_frequencies)):
        d[int(v.item())] = ((f / n) * sampler.n_total).item()
    for v, relative_f in d.items():
        rounded = round(relative_f)
        error = relative_f - rounded
        print(f"{v:>2}:  {rounded}  ({error:+.3f})")
