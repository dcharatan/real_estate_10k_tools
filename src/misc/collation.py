from typing import Callable, Dict, List, Union

from torch import Tensor

Tree = Union[Dict[str, "Tree"], Tensor]


def collate(trees: List[Tree], merge_fn: Callable[[List[Tensor]], Tensor]) -> Tree:
    """Merge nested dictionaries of tensors."""
    if isinstance(trees[0], Tensor):
        return merge_fn(trees)
    else:
        return {
            key: collate([tree[key] for tree in trees], merge_fn) for key in trees[0]
        }
