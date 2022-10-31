from typing import List

import torch

from spmd import DeviceMesh
from spmd.tensor.placement_types import Placement

from .utils import split_tensor_along_last_dim


def _scatter_local_tensor(input_: torch.Tensor, mesh: DeviceMesh, mesh_dim: int):
    world_size_dim = mesh.size(mesh_dim)
    rank_dim = mesh.get_coordinate_on_dim(mesh_dim)
    if world_size_dim == 1:
        return input_
    split_tensor = split_tensor_along_last_dim(input_, world_size_dim)
    output = split_tensor[rank_dim].contiguous()
    return output


def gather_tensor(input_: torch.Tensor, mesh: DeviceMesh, mesh_dim: int):
    world_size_dim = mesh.size(mesh_dim)
    if world_size_dim == 1:
        return input_
    input_size = list(input_.size())
    input_size[-1] *= world_size_dim
    gather_tensor = torch.empty(*input_size, dtype=input_.dtype, device=input_.device)
    # (deprecated) mesh.all_gather_base
    # mesh.all_gather_base(gather_tensor, input_, mesh_dim=mesh_dim, tensor_dim=-1)
    torch.distributed.all_gather_into_tensor(gather_tensor, input_, mesh.get_dim_groups()[mesh_dim])
    return gather_tensor


class _ShardToTPRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, mesh: DeviceMesh, shard_strategy: List[Placement]):
        ctx.mesh = mesh
        ctx.shard_strategy = shard_strategy

        mesh_dim_ = None
        for mesh_dim, shard_this_mesh_dim in enumerate(shard_strategy):
            if shard_this_mesh_dim.is_shard():
                if shard_this_mesh_dim.dim == 0:
                    mesh_dim_ = mesh_dim

        ctx.mesh_dim = mesh_dim_
        return _scatter_local_tensor(input_, mesh, mesh_dim_)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_tensor(grad_output, ctx.mesh, ctx.mesh_dim), None, None


shard_to_tp_region = _ShardToTPRegion.apply
