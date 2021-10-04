from abc import ABCMeta
from typing import Any
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# This piece of code fixes this PyTorch issue:
# https://github.com/pytorch/pytorch/issues/48439
def _process_args_for_checkpoint(
        args: Tuple[Union[torch.Tensor, Any], ...],
) -> Tuple[Union[torch.Tensor, Any], ...]:

    new_args = []
    for arg in args:
        if torch.is_tensor(arg) and arg.is_contiguous(memory_format=torch.channels_last):
            arg = arg.view_as(arg)
        new_args.append(arg)

    return tuple(new_args)


class CheckpointableModule(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self._use_grad_checkpoint: bool = False
        self._preserve_rng_state: bool = True

    @property
    def is_grad_checkpoint_enabled(self) -> bool:
        return self._use_grad_checkpoint

    def enable_grad_checkpoint(self, enable: bool = True) -> None:
        self._use_grad_checkpoint = enable

    def disable_grad_checkpoint(self) -> None:
        self._use_grad_checkpoint = False

    @property
    def preserve_rng_state(self) -> bool:
        return self._preserve_rng_state

    @preserve_rng_state.setter
    def preserve_rng_state(self, mode: bool) -> None:
        self._preserve_rng_state = mode

    def __call__(self, *args, **kwargs) -> Any:

        if self._use_grad_checkpoint and torch.is_grad_enabled():

            # TODO: try to fix this
            if kwargs:
                raise ValueError('Checkpoint mechanism does not support passing keyword arguments to module forward')

            parameters = tuple(self.parameters())
            n_parameters = len(parameters)

            def checkpoint_fn(*_args, **_kwargs):
                _args = _args[:-n_parameters]
                _args = _process_args_for_checkpoint(_args)
                return self._call_impl(*_args, **_kwargs)

            new_args = args + parameters

            return checkpoint(checkpoint_fn, *new_args, preserve_rng_state=self.preserve_rng_state)

        return self._call_impl(*args, **kwargs)


def is_checkpointable(module: torch.nn.Module) -> bool:
    return isinstance(module, CheckpointableModule)


def enable_checkpoint(model: torch.nn.Module, enable: bool = True) -> None:
    for module in model.modules():
        if is_checkpointable(module):
            module.enable_grad_checkpoint(enable)
