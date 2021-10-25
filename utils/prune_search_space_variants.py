from typing import List

import numpy as np
import torch
import torch.nn as nn
from enot.models import SearchSpaceModel


def prune_search_block(block, prune_indices: List[int]) -> None:

    n_was = len(block)
    n_new = n_was - len(prune_indices)
    keep_indices = np.array([i for i in range(n_was) if i not in prune_indices], dtype=np.int64)
    print(keep_indices)

    modules = [x for i, x in enumerate(block._operations) if i in keep_indices]
    block._operations = nn.ModuleList(modules)

    tmp_ = torch.zeros(n_new).to(block._block_parameters)
    tmp_[:] = block._block_parameters[torch.from_numpy(keep_indices).to(device=block._block_parameters.device)]
    tmp_.requres_grad = True
    block._block_parameters.data = tmp_.data

    # TODO: Make something like getter for _policy attribute of sampler
    block._sampler._policy._operations_count = n_new
    block._saved_block_parameters = block._block_parameters.detach()


def prune_search_space_variants(
        search_space: SearchSpaceModel,
        prune_indices: List[List[int]],
) -> None:

    search_blocks = search_space.search_blocks
    for block, indices in zip(search_blocks, prune_indices):
        prune_search_block(block, indices)

    print(search_space.architecture_probabilities)
    print([len(x) for x in search_blocks])
