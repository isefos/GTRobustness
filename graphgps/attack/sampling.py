import torch
import math


class WeightedIndexSampler:
    def __init__(
        self,
        weighted_idx: None | torch.Tensor,
        zero_idx: None | torch.Tensor,
        weight: int,
        max_index: int,
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
            weighted_idx.sort(dim=0)
            self.weighted_idx_padded = torch.cat(
                (weighted_idx, torch.tensor([-1], dtype=torch.int64)), dim=0,
            )
            self.weighted_idx = self.weighted_idx_padded[:-1]
        else:
            self.weighted_idx = None

        if zero_idx is not None:
            zero_idx.sort(dim=0)
            self.zero_idx_padded = torch.cat(
                (zero_idx, torch.tensor([-1], dtype=torch.int64)), dim=0,
            )
            self.zero_idx = self.zero_idx_padded[:-1]
        else:
            self.zero_idx = None

        self.max_index: int = max_index
        self.n_w: int = weighted_idx.size(0) if weighted_idx is not None else 0
        self.n_z: int = zero_idx.size(0) if zero_idx is not None else 0
        self.w_minus_1: int = weight - 1
        self.n_total: int = self.max_index + 1 + self.w_minus_1 * self.n_w - self.n_z
        assert self.n_total < 2 ** 63 - 1  # If we sample values of 64 bit integers

    def sample(self, num_samples: int, device):
        """
        """
        values = torch.randint(
            low=0,
            high=self.n_total,
            size=(num_samples, ),
            dtype=torch.int64,
            device=device,
        )
        # if no weights to consider, can just return uniform sample
        if self.weighted_idx is None and self.zero_idx is None:
            return values
        # map the uniform sample to the weighted distribution using binary search
        weighted_idx_sample = torch.zeros_like(values, dtype=torch.int64, device=device) - 1
        l = torch.zeros_like(values, dtype=torch.int64, device=device)
        r = torch.zeros_like(values, dtype=torch.int64, device=device) + self.max_index
        step_size = torch.ones_like(values, dtype=torch.int64, device=device)
        num_steps = 0
        max_steps = math.ceil(math.log2(self.n_total))
        while not torch.all(weighted_idx_sample >= 0):
            num_steps += 1
            if num_steps > max_steps:
                raise Exception("Something must have gone wrong...")
            c = (r + l) // 2
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
            # set new boundaries
            value_is_larger_mask = values >= c_values_max
            l[value_is_larger_mask] = c[value_is_larger_mask] + 1
            value_is_smaller_mask = values < c_values_min
            r[value_is_smaller_mask] = c[value_is_smaller_mask] - 1
            # set the indexes that were reached/ found
            unreached_mask = weighted_idx_sample < 0
            value_on_next_mask = l >= r
            c[value_on_next_mask] = l[value_on_next_mask]
            value_on_current_mask = ~value_is_smaller_mask * ~value_is_larger_mask
            value_reached_mask = torch.logical_or(value_on_current_mask, value_on_next_mask)
            weighted_idx_sample[unreached_mask] = torch.where(value_reached_mask, c, -1)
            # limit the binary search to only the remaining
            remaining_mask = ~value_reached_mask
            l = l[remaining_mask]
            r = r[remaining_mask]
            step_size = step_size[remaining_mask]
            values = values[remaining_mask]
        return weighted_idx_sample
    

if __name__ == "__main__":
    max_index = 12
    weighted_idx = torch.tensor([1, 5, 7], dtype=torch.int64)
    zero_idx = torch.tensor([3, 8, 9], dtype=torch.int64)
    w = 3
    sampler = WeightedIndexSampler(
        weighted_idx=weighted_idx,
        zero_idx=zero_idx,
        weight=w,
        max_index=max_index,
    )
    n = 10000
    s = sampler.sample(n, torch.device("cpu"))
    sample_values, sample_frequencies = torch.unique(s, sorted=True, return_counts=True)
    d = {i: 0.0 for i in range(max_index + 1)}
    for v, f in zip(list(sample_values), list(sample_frequencies)):
        d[int(v.item())] = ((f / n) * sampler.n_total).item()
    for v, relative_f in d.items():
        rounded = round(relative_f)
        error = relative_f - rounded
        print(f"{v:>2}:  {rounded}  ({error:+.3f})")
