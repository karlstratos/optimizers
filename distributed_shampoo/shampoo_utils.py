"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import math
from abc import ABC
from copy import deepcopy
from typing import List, Union

import torch
import torch.distributed as dist

from matrix_functions import matrix_inverse_root
from torch import Tensor

logger = logging.getLogger(__name__)

###### ENUM CLASSES ######
class PreconditionerType(enum.Enum):
    FULL = 0
    DIAGONAL = 1


class GraftingType(enum.Enum):
    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4


class LargeDimMethod(enum.Enum):
    DIAGONAL = 0
    ADAGRAD = 1
    BLOCKING = 2


class RootInvStrategy(enum.Enum):
    NONE = 0
    PARAM = 1
    BLOCK = 2
    PRECOND = 3


###### MERGING AND BLOCKING HELPER FUNCTIONS ######
def merge_small_dims(tensor_shape: List[int], threshold: int) -> List[int]:
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (List[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """

    new_tensor_shape = [tensor_shape[0]]
    for next_tensor_shape in tensor_shape[1:]:
        new_dimension = new_tensor_shape[-1] * next_tensor_shape
        if (
            new_tensor_shape[-1] == 1
            or next_tensor_shape == 1
            or new_dimension <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)

    return new_tensor_shape


def multi_dim_split(tensor: Tensor, splits: List[int]) -> List[Tensor]:
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        splits (List[int]): List of sizes for each block or chunk along each dimension.

    Returns:
        split_grad (List[Tensor]): List of tensors.

    """
    split_tensors = [tensor]
    for dim, split in enumerate(splits):
        split_tensors = [
            s for t in split_tensors for s in torch.split(t, split, dim=dim)
        ]
    return split_tensors


def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
    """Concatenates multiple tensors to form single tensor across multiple dimensions.

    Args:
        split_tensor (List[Tensor]): List of tensor splits or blocks.
        num_splits (List[int]): Number of splits/blocks.

    Returns:
        merged_tensor (Tensor): Merged tensor.

    """
    merged_tensor = split_tensors
    for dim, split in reversed(list(enumerate(num_splits))):
        if split > 0:
            merged_tensor = [
                torch.cat(merged_tensor[i : i + split], dim=dim)
                for i in range(0, len(merged_tensor), split)
            ]
    assert len(merged_tensor) == 1
    return merged_tensor[0]


###### PRECONDITIONER CLASSES ######
class Preconditioner(ABC):
    """Preconditioner base class."""

    def __init__(self):
        self._parameter_count = 0
        pass

    def update_preconditioners(self, grad: Tensor) -> None:
        pass

    def precondition(self, grad: Tensor) -> Tensor:
        pass

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[float, Tensor],
    ) -> None:
        pass

    def compute_norm(self, grad: Tensor) -> Tensor:
        pass

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def broadcast(self):
        return

    def to(self, device: Union[None, torch.device] = None):
        pass


class AdagradPreconditioner(Preconditioner):
    """Adagrad/Adam/RMSProp preconditioner for a generic layer.

    Stores preconditioner using same format as parameter p. Operations are performed in-place.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)
        idx (Union[None, str, int]): Layer index (for logging purposes). (Default: None)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
        idx: Union[None, str, int] = None,
    ):
        super(AdagradPreconditioner, self).__init__()
        self.beta2 = beta2
        self.epsilon = epsilon
        self.preconditioner = torch.zeros_like(param)
        self.idx = idx
        self.num_updates = 0
        self.use_bias_correction = use_bias_correction
        self.bias_correction2 = torch.tensor(1.0)
        self._parameter_count += torch.prod(torch.tensor(self.preconditioner.shape))

        if self.idx is not None:
            self.preconditioner_idx = str(self.idx) + "." + str(0)
            logger.info(
                f"Diagonal Adagrad Preconditioner {self.preconditioner_idx} with Parameter {self.idx}"
            )

    def update_preconditioners(self, grad: Tensor) -> None:
        if self.beta2 == 1.0:
            self.preconditioner.addcmul_(grad, grad, value=1)
        else:
            self.preconditioner.mul_(self.beta2).addcmul_(
                grad, torch.conj(grad), value=1 - self.beta2
            )

        self.num_updates += 1
        if self.use_bias_correction and self.beta2 < 1.0:
            self.bias_correction2 = torch.tensor(1.0 - self.beta2**self.num_updates)

    def precondition(self, grad: Tensor) -> Tensor:
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        grad.div_(denom)
        return grad

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[float, Tensor],
    ) -> None:
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        param.addcdiv_(grad, denom, value=-lr)

    def compute_norm(self, grad: Tensor):
        denom = (self.preconditioner / self.bias_correction2).sqrt().add_(self.epsilon)
        adagrad_nrm = torch.linalg.norm(grad / denom)
        return adagrad_nrm

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self.preconditioner = self.preconditioner.to(device=device)
            self.bias_correction2 = self.bias_correction2.to(device=device)
            self._parameter_count = self._parameter_count.to(device=device)


class ShampooPreconditioner(Preconditioner):
    """Shampoo preconditioners for some generic layer.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        diagonal_threshold (int): Threshold for using diagonal preconditioners. If None, disabled. (Default: None)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        root_inv_strategy (RootInvStrategy): Strategy for assigning root inverse computations. (Default: RootInvStrategy.PRECOND)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        init_delay (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (default: 0)
        grafting_type (GraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
        diagonal_threshold: Union[None, int] = None,
        dtype: torch.dtype = torch.float,
        root_inv_strategy: RootInvStrategy = RootInvStrategy.PRECOND,
        idx: Union[None, int, str] = None,
        init_delay: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
    ):

        super(ShampooPreconditioner, self).__init__()

        # initialize parameters
        self.beta2 = beta2
        self.epsilon = epsilon
        self.diagonal_threshold = diagonal_threshold
        self.dtype = dtype
        self.num_updates = 0
        self.use_bias_correction = use_bias_correction
        self.bias_correction2 = torch.tensor(1.0)
        self.dims = torch.tensor(param.shape).numpy()
        self.order = param.dim()
        self.root_inv_strategy = root_inv_strategy
        self.idx = idx
        self.grafting_type = grafting_type
        self.init_delay = init_delay
        self.rank = (
            dist.get_rank() if self.root_inv_strategy != RootInvStrategy.NONE else 0
        )

        # initialize lists for each parameter
        self.preconditioners = []
        self.inv_preconditioners = []
        self.preconditioner_types = []
        self.preconditioner_ranks = []
        if self.idx is not None:
            self.preconditioner_idx = [
                str(self.idx) + "." + str(k) for k in range(self.order)
            ]

        for k, dim in enumerate(self.dims):

            if self.diagonal_threshold is not None and dim > self.diagonal_threshold:
                preconditioner = torch.zeros(
                    dim, dtype=param.dtype, device=param.device
                )
                inv_preconditioner = None
                preconditioner_type = PreconditionerType.DIAGONAL
                num_params = dim
                preconditioner_rank = None
                if self.idx is not None:
                    logger.info(
                        f"Diagonal Preconditioner {self.preconditioner_idx[k]} with Parameter {self.idx}, Order {k}, Dim {dim}, Number of Params {num_params}, DType {self.dtype}"
                    )

            else:
                preconditioner = torch.zeros(
                    (dim, dim), dtype=self.dtype, device=param.device
                )
                inv_preconditioner = torch.zeros(
                    (dim, dim), dtype=param.dtype, device=param.device
                )
                preconditioner_type = PreconditionerType.FULL
                num_params = dim**2
                preconditioner_rank = -1
                if self.idx is not None:
                    logger.info(
                        f"Full Matrix Preconditioner {self.preconditioner_idx[k]} with Parameter {self.idx}, Order {k}, Dim {dim}, Number of Params {num_params}, DType {self.dtype}"
                    )

            self._parameter_count += num_params
            self.preconditioners.append(preconditioner)
            self.inv_preconditioners.append(inv_preconditioner)
            self.preconditioner_types.append(preconditioner_type)
            self.preconditioner_ranks.append(preconditioner_rank)

        # initialize grafting
        if self.grafting_type == GraftingType.NONE:
            self.grafting = None
        elif self.grafting_type == GraftingType.SGD:
            self.grafting = SGDGrafting(param)
        elif self.grafting_type == GraftingType.ADAGRAD:
            self.grafting = AdagradGrafting(param, epsilon=grafting_epsilon)
        elif self.grafting_type == GraftingType.RMSPROP:
            self.grafting = RMSPropGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif self.grafting_type == GraftingType.ADAM:
            self.grafting = AdamGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        else:
            raise ValueError(f"Invalid Grafting Type {self.grafting_type}!")

        # add number of parameters for grafted method
        if self.grafting_type != GraftingType.NONE:
            self._parameter_count += self.grafting.parameter_count

    def update_preconditioners(self, grad: Tensor) -> None:

        assert (
            len(self.dims)
            == len(self.preconditioners)
            == len(self.preconditioner_types)
        ), f"Length of dimensions {len(self.dims)}, preconditioners {len(self.preconditioners)}, and preconditioner types {len(self.preconditioner_types)} are not equal!"

        for k, (dim, preconditioner, preconditioner_type) in enumerate(
            zip(self.dims, self.preconditioners, self.preconditioner_types)
        ):
            if self.beta2 != 1.0:
                preconditioner.mul_(self.beta2)

            # update preconditioners (diagonal case)
            if preconditioner_type == PreconditionerType.DIAGONAL:
                diagonal_or_outer_product = (
                    torch.linalg.norm(
                        grad.transpose(0, k).contiguous().view(dim, -1),
                        dim=1,
                    ).pow(2),
                )

            # update preconditioners (full-matrix case)
            else:
                contract_idx = [*range(k)] + [*range(k + 1, self.order)]
                diagonal_or_outer_product = torch.tensordot(
                    grad,
                    grad,
                    dims=(contract_idx, contract_idx),
                )
                if diagonal_or_outer_product.dtype != self.dtype:
                    diagonal_or_outer_product = diagonal_or_outer_product.to(
                        dtype=self.dtype
                    )

            preconditioner.add_(
                diagonal_or_outer_product,
                alpha=1 - self.beta2 if self.beta2 != 1.0 else 1.0,
            )

        # update grafting method
        if self.grafting_type != GraftingType.NONE:
            self.grafting.update_preconditioners(grad)

        self.num_updates += 1
        if self.use_bias_correction and self.beta2 < 1.0:
            self.bias_correction2 = 1.0 - self.beta2**self.num_updates

    def _shampoo_precondition(self, grad: Tensor) -> Tensor:

        preconditioned_grad = grad.clone()

        assert (
            len(self.preconditioners)
            == len(self.inv_preconditioners)
            == len(self.preconditioner_types)
        ), f"Length of preconditioners {len(self.preconditioners)}, inverse preconditioners {len(self.inv_preconditioners)}, and preconditioner types {len(self.preconditioner_types)} are not equal!"

        for k, (preconditioner, inv_preconditioner, preconditioner_type) in enumerate(
            zip(
                self.preconditioners,
                self.inv_preconditioners,
                self.preconditioner_types,
            )
        ):

            # handle diagonal case while retaining dims
            if self.diagonal_threshold is not None:

                # precondition in diagonal case
                if preconditioner_type == PreconditionerType.DIAGONAL:
                    denom = (preconditioner / self.bias_correction2).add_(self.epsilon)
                    preconditioned_grad.div_(
                        denom.pow(-1 / (2 * self.order))[
                            (None,) * k + (...,) + (None,) * (self.order - k - 1)
                        ]
                    )

                # precondition in full-matrix case
                else:
                    gradient_idx = [*range(1, self.order + 1)]
                    matrix_product_idx = deepcopy(gradient_idx)
                    matrix_product_idx[k] = 0
                    preconditioned_grad = torch.einsum(
                        inv_preconditioner,
                        [0, k + 1],
                        preconditioned_grad,
                        gradient_idx,
                        matrix_product_idx,
                    )

            # more efficient if no diagonal preconditioners; transposes grad continually
            else:
                preconditioned_grad = torch.tensordot(
                    preconditioned_grad, inv_preconditioner, [[0], [0]]
                )

        # apply grafting
        if self.grafting_type != GraftingType.NONE:
            grafting_norm = self.grafting.direction_norm(grad)
            preconditioned_grad = (
                preconditioned_grad
                * grafting_norm
                / (torch.linalg.norm(preconditioned_grad) + 1e-16)
            )

        return preconditioned_grad

    def _graft_precondition(self, grad: Tensor) -> Tensor:
        return (
            self.grafting.precondition(grad)
            if self.grafting_type != GraftingType.NONE
            else grad
        )

    def precondition(self, grad: Tensor) -> Tensor:
        return (
            self._graft_precondition(grad)
            if self.num_updates <= self.init_delay
            else self._shampoo_precondition(grad)
        )

    def compute_root_inverse(self) -> None:
        assert (
            len(self.preconditioners)
            == len(self.preconditioner_types)
            == len(self.preconditioner_ranks)
        ), f"Length of preconditioners {len(self.preconditioners)}, preconditioner types {len(self.preconditioner_types)}, and preconditioner ranks {len(self.preconditioner_ranks)} are not equal!"

        for k, (preconditioner, preconditioner_type, preconditioner_rank) in enumerate(
            zip(
                self.preconditioners,
                self.preconditioner_types,
                self.preconditioner_ranks,
            )
        ):

            # check that this is a full matrix preconditioner
            if preconditioner_type == PreconditionerType.FULL and (
                preconditioner_rank == -1 or preconditioner_rank == self.rank
            ):
                # add epsilon term and incorporate bias correction
                bias_corrected_preconditioner = preconditioner / self.bias_correction2

                # check if nan or inf values
                if torch.any(torch.isnan(bias_corrected_preconditioner)):
                    logger.warning(
                        f"Encountered nan values in preconditioner {self.idx}.{k}!"
                    )
                elif torch.any(torch.isinf(bias_corrected_preconditioner)):
                    logger.warning(
                        f"Encountered inf values in preconditioner {self.idx}.{k}!"
                    )

                # compute inverse preconditioner and store
                root = 2 * self.order
                inv_preconditioner = matrix_inverse_root(
                    A=bias_corrected_preconditioner,
                    root=root,
                    epsilon=self.epsilon,
                )

                if inv_preconditioner.dtype != self.inv_preconditioners[k].dtype:
                    inv_preconditioner = inv_preconditioner.to(
                        dtype=self.inv_preconditioners[k].dtype
                    )
                self.inv_preconditioners[k] = inv_preconditioner

    def precondition_and_update(
        self, param, grad: Tensor, lr: Union[float, Tensor]
    ) -> None:
        if self.num_updates <= self.init_delay:
            self.grafting.precondition_and_update(param, grad, lr)
        else:
            preconditioned_grad = self.precondition(grad)
            param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self):
        for k in range(self.order):
            if self.preconditioner_types[k] == PreconditionerType.FULL:
                dist.broadcast(
                    self.inv_preconditioners[k],
                    src=self.preconditioner_ranks[k],
                )

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self.bias_correction2 = self.bias_correction2.to(device=device)
            self._parameter_count = self._parameter_count.to(device=device)
            self.preconditioners = [
                preconditioner.to(device) for preconditioner in self.preconditioners
            ]
            self.inv_preconditioners = [
                inv_preconditioner.to(device)
                for inv_preconditioner in self.inv_preconditioners
            ]

    def _assign_preconditioners_rank(
        self,
        rank: int,
        world_size: int,
        preconditioner_rank_increment: int = 0,
    ) -> int:
        for k in range(self.order):
            if self.preconditioner_types[k] == PreconditionerType.FULL:
                self.preconditioner_ranks[k] = rank % world_size
                rank += preconditioner_rank_increment
                if self.idx is not None:
                    logger.info(
                        f"Assigned Preconditioner {self.preconditioner_idx[k]} to rank {self.preconditioner_ranks[k]}"
                    )
        return rank

    def assign_preconditioners_rank(self, rank: int, world_size: int) -> int:
        if self.root_inv_strategy == RootInvStrategy.NONE:
            return -1
        elif self.root_inv_strategy in (RootInvStrategy.PARAM, RootInvStrategy.BLOCK):
            return self._assign_preconditioners_rank(
                rank, world_size, preconditioner_rank_increment=0
            )
        elif self.root_inv_strategy == RootInvStrategy.PRECOND:
            return self._assign_preconditioners_rank(
                rank, world_size, preconditioner_rank_increment=1
            )
        else:
            raise NotImplementedError(
                "Root inverse strategy is not implemented! Specified root inverse strategy is "
                + str(self.root_inv_strategy)
                + "."
            )


class BlockShampooPreconditioner(Preconditioner):
    """Shampoo with blocking applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        block_size (int): Block size for blocking large tensors. (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        root_inv_strategy (RootInvStrategy): Strategy for assigning root inverse computations. (Default: RootInvStrategy.PRECOND)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged. (Default: True)
        init_delay (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
        block_size: int = 1024,
        dtype: torch.dtype = torch.float,
        root_inv_strategy: RootInvStrategy = RootInvStrategy.NONE,
        idx: Union[None, int, str] = None,
        use_merge_dims: bool = True,
        init_delay: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
    ):
        super(BlockShampooPreconditioner, self).__init__()

        # Set hyperparameters
        self.beta2 = beta2
        self.epsilon = epsilon
        self.use_bias_correction = use_bias_correction
        self.block_size = block_size
        self.dtype = dtype
        self.num_updates = 0
        self.idx = idx
        self.root_inv_strategy = root_inv_strategy
        self.init_delay = init_delay
        self.use_merge_dims = use_merge_dims
        self.original_dims = [*torch.tensor(param.shape).numpy()]
        self.merged_dims = (
            merge_small_dims(self.original_dims, self.block_size)
            if self.block_size is not None and use_merge_dims
            else self.original_dims
        )
        self.original_order = param.dim()
        self.merged_order = (
            len(self.merged_dims) if use_merge_dims else self.original_order
        )

        # Construct splits for blocking
        self.splits = [block_size] * len(self.merged_dims)
        self.num_splits = [math.ceil(dim / block_size) for dim in self.merged_dims]

        # Construct multiple preconditioners for each block
        self.split_preconditioners = []
        self.split_sizes = []

        if self.use_merge_dims:
            param = param.view(self.merged_dims)

        split_param = multi_dim_split(param, self.splits)
        for i, p in enumerate(split_param):
            self.split_sizes.append(torch.tensor(p.shape))
            split_idx = str(idx) + "." + str(i)
            preconditioner = ShampooPreconditioner(
                p,
                beta2=beta2,
                epsilon=epsilon,
                use_bias_correction=use_bias_correction,
                dtype=dtype,
                root_inv_strategy=root_inv_strategy,
                idx=split_idx,
                init_delay=init_delay,
                grafting_type=grafting_type,
                grafting_beta2=grafting_beta2,
                grafting_epsilon=grafting_epsilon,
            )
            self.split_preconditioners.append(preconditioner)
            self._parameter_count += preconditioner.parameter_count

    def update_preconditioners(self, grad: Tensor):
        if self.use_merge_dims:
            grad = grad.view(self.merged_dims)
        split_grad = multi_dim_split(grad, self.splits)
        for i, g in enumerate(split_grad):
            self.split_preconditioners[i].update_preconditioners(g)
        self.num_updates += 1

    def precondition(self, grad: Tensor) -> Tensor:
        if self.use_merge_dims:
            grad = grad.view(self.merged_dims)
        split_grad = multi_dim_split(grad, self.splits)
        split_preconditioned_grad = []
        for i, g in enumerate(split_grad):
            preconditioned_g = self.split_preconditioners[i].precondition(g)
            split_preconditioned_grad.append(preconditioned_g)
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self.num_splits)
        if self.use_merge_dims:
            preconditioned_grad = preconditioned_grad.view(self.original_dims)
        return preconditioned_grad

    def compute_root_inverse(self) -> None:
        for preconditioner in self.split_preconditioners:
            preconditioner.compute_root_inverse()

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[Tensor, float],
    ) -> None:
        preconditioned_grad = self.precondition(grad)
        param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self):
        for preconditioner in self.split_preconditioners:
            preconditioner.broadcast()

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self._parameter_count = self._parameter_count.to(device=device)
            for preconditioner in self.split_preconditioners:
                preconditioner.to(device=device)

    def _assign_preconditioners_rank(
        self,
        rank: int,
        world_size: int,
        block_rank_increment: int = 0,
    ) -> int:
        for preconditioner in self.split_preconditioners:
            rank = preconditioner.assign_preconditioners_rank(rank, world_size)
            rank += block_rank_increment
        return rank

    def assign_preconditioners_rank(self, rank: int, world_size: int) -> int:
        if self.root_inv_strategy == RootInvStrategy.NONE:
            return -1
        elif self.root_inv_strategy in (RootInvStrategy.PARAM, RootInvStrategy.PRECOND):
            return self._assign_preconditioners_rank(
                rank, world_size, block_rank_increment=0
            )
        elif self.root_inv_strategy == RootInvStrategy.BLOCK:
            return self._assign_preconditioners_rank(
                rank, world_size, block_rank_increment=1
            )
        else:
            raise NotImplementedError(
                "Root inverse strategy is not implemented! Specified root inverse strategy is "
                + str(self.root_inv_strategy)
                + "."
            )


###### GRAFTING CLASSES ######
class Grafting(ABC):
    """Grafting base class.

    We graft the method by storing and maintaining the preconditioner for the grafted method.
    Therefore, any additional modifications including gradient EMA/filtering and momentum are
    not included in grafting.

    """

    def __init__(self, param: Tensor):
        self._parameter_count = 0
        pass

    def update_preconditioners(self, grad: Tensor):
        pass

    def precondition(self, grad: Tensor) -> Tensor:
        pass

    def direction_norm(self, grad: Tensor) -> Tensor:
        pass

    def precondition_and_update(
        self,
        param: Tensor,
        grad: Tensor,
        lr: Union[float, Tensor],
    ):
        pass

    @property
    def parameter_count(self):
        return self._parameter_count

    def to(self, device: Union[None, torch.device] = None):
        return


class SGDGrafting(Grafting):
    """SGD grafting.

    Grafts the stochastic gradient method by returning the norm of the gradient.

    Args:
        param (Tensor): Parameter of interest.

    """

    def __init__(self, param: Tensor):
        super(SGDGrafting, self).__init__(param)

    def precondition(self, grad: Tensor) -> Tensor:
        return grad

    def direction_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(grad)

    def precondition_and_update(self, param, grad: Tensor, lr: Union[float, Tensor]):
        param.add_(grad, alpha=-lr)


class AdagradGrafting(Grafting):
    """Adagrad grafting.

    Supports RMSProp and Adam by determining beta2 and use_bias_correction.

    Note: beta1 is not included since that is shared between both Shampoo and the grafted optimizer.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        param: Tensor,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ):
        super(AdagradGrafting, self).__init__(param)
        self.preconditioner = AdagradPreconditioner(
            param, beta2=beta2, epsilon=epsilon, use_bias_correction=use_bias_correction
        )
        self._parameter_count += self.preconditioner.parameter_count

    def update_preconditioners(self, grad: Tensor):
        self.preconditioner.update_preconditioners(grad)

    def precondition(self, grad: Tensor) -> Tensor:
        return self.preconditioner.precondition(grad)

    def direction_norm(self, grad: Tensor) -> Tensor:
        return self.preconditioner.compute_norm(grad)

    def precondition_and_update(self, param, grad: Tensor, lr: Union[float, Tensor]):
        self.preconditioner.precondition_and_update(param, grad, lr)

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self.preconditioner.to(device=device)
        return


class RMSPropGrafting(AdagradGrafting):
    """RMSProp grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.99, epsilon: float = 1e-8):
        super(RMSPropGrafting, self).__init__(
            param=param, beta2=beta2, epsilon=epsilon, use_bias_correction=False
        )


class AdamGrafting(AdagradGrafting):
    """Adam grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.999, epsilon: float = 1e-8):
        super(AdamGrafting, self).__init__(
            param=param, beta2=beta2, epsilon=epsilon, use_bias_correction=True
        )
