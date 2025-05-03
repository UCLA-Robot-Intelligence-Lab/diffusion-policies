import collections
import torch
import torch.nn as nn
import copy

from typing import Dict, Callable, List
from contextlib import contextmanager


@contextmanager
def temporary_attribute(obj, attr, value):
    original_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


def inspect_dict_structure(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}: <dict>")
            inspect_dict_structure(value, indent + 1)
        elif isinstance(value, torch.Tensor):
            print("  " * indent + f"{key}: <Tensor, shape={tuple(value.shape)}>")
        else:
            print("  " * indent + f"{key}: {type(value).__name__}")


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[: len(x.shape)]
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer
