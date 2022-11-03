from typing import List

import torch
import torch.distributed as dist

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


def gather_tensor(input_: torch.Tensor, tensor_dim: int, group: dist.ProcessGroup, async_op=False):
    world_size_dim = group.size()
    if world_size_dim == 1:
        return input_

    if tensor_dim != 0:
        input_ = input_.transpose(tensor_dim, 0).contiguous()
    input_size = list(input_.size())
    input_size[0] *= world_size_dim
    gather_tensor = torch.empty(*input_size, dtype=input_.dtype, device=input_.device)
    handle = dist.all_gather_into_tensor(gather_tensor, input_, group=group, async_op=async_op)

    post_func = lambda x: x

    if tensor_dim != 0:
        post_func = lambda x: x.transpose(tensor_dim, 0).contiguous()

    if async_op:
        return gather_tensor, handle, post_func

    return post_func(gather_tensor)


def reduce_scatter_tensor(input_: torch.Tensor,
                          tensor_dim: int,
                          group: dist.ProcessGroup,
                          async_op=False):
    world_size_dim = group.size()
    if world_size_dim == 1:
        return input_

    if tensor_dim != 0:
        input_ = input_.transpose(tensor_dim, 0).contiguous()
    input_size = list(input_.size())
    input_size[0] //= world_size_dim
    scatter_tensor = torch.empty(*input_size, dtype=input_.dtype, device=input_.device)
    handle = dist.reduce_scatter_tensor(scatter_tensor, input_, group=group, async_op=async_op)

    post_func = lambda x: x

    if tensor_dim != 0:
        post_func = lambda x: x.transpose(tensor_dim, 0).contiguous()

    if async_op:
        return scatter_tensor, handle, post_func

    return post_func(scatter_tensor)


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
        return gather_tensor(grad_output, -1, ctx.mesh.get_dim_groups()[ctx.mesh_dim]), None, None


shard_to_tp_region = _ShardToTPRegion.apply
