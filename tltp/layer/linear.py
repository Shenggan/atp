from typing import List

import torch
from torch.nn.parameter import Parameter
import torch.distributed as dist

from spmd import DeviceMesh, Shard
from spmd.tensor.placement_types import Placement

from .mapping import shard_to_tp_region, gather_tensor, reduce_scatter_tensor
from .utils import delay_kernel


class TLLinearFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                mesh: DeviceMesh, input_seq_shard: bool, output_seq_shard: bool,
                forward_allreduce_mesh_dim: int, backward_allreduce_mesh_dim: int):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.input_shape = input.shape
        ctx.mesh = mesh
        ctx.backward_allreduce_mesh_dim = backward_allreduce_mesh_dim
        ctx.input_seq_shard = input_seq_shard
        ctx.output_seq_shard = output_seq_shard
        ctx.seq_dim = 1

        # all_gather input in sequence dim (replicate dim on mesh)
        if ctx.input_seq_shard:
            gather_mesh_dim = 1 - forward_allreduce_mesh_dim
            gather_group = mesh.get_dim_groups()[gather_mesh_dim]
            input = gather_tensor(input, tensor_dim=ctx.seq_dim, group=gather_group)

        # convert the tensor shapes to 2D for aten ops
        input_2d = input.view(-1, input.size()[-1])
        output = torch.ops.aten.mm(input_2d, weight.t())

        output = output.view(*(input.size()[:-1]), -1)

        # if ctx.output_seq_shard, use reduce_scatter instead of allreduce
        reduce_group = mesh.get_dim_groups()[forward_allreduce_mesh_dim]
        if ctx.output_seq_shard:
            output = reduce_scatter_tensor(output, tensor_dim=ctx.seq_dim, group=reduce_group)
        else:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=reduce_group)

        if ctx.use_bias:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        mesh = ctx.mesh
        backward_allreduce_mesh_dim = ctx.backward_allreduce_mesh_dim

        grad_output = grad_output.contiguous()

        # all_gather input in sequence dim (replicate dim on mesh)
        if ctx.input_seq_shard:
            gather_mesh_dim = backward_allreduce_mesh_dim
            gather_group = mesh.get_dim_groups()[gather_mesh_dim]
            input = gather_tensor(input, tensor_dim=ctx.seq_dim, group=gather_group)

        # if ctx.output_seq_shard, allgather here
        if ctx.output_seq_shard:
            gather_mesh_dim = 1 - backward_allreduce_mesh_dim
            gather_group = mesh.get_dim_groups()[gather_mesh_dim]
            grad_output = gather_tensor(grad_output, tensor_dim=ctx.seq_dim, group=gather_group)

        grad_output = grad_output.view(-1, grad_output.size()[-1])
        input_2d = input.view(-1, input.size()[-1])

        # async allreduce for grad_input
        grad_input = torch.ops.aten.mm(grad_output, weight)
        grad_input = grad_input.view(*(input.size()))

        # local scatter grad_input
        handle = None
        reduce_group = mesh.get_dim_groups()[backward_allreduce_mesh_dim]
        if ctx.input_seq_shard:
            grad_input, handle = reduce_scatter_tensor(grad_input, tensor_dim=ctx.seq_dim, group=reduce_group, async_op=True)
        else:
            handle = dist.all_reduce(grad_input,
                                     op=dist.ReduceOp.SUM,
                                     group=reduce_group,
                                     async_op=True)
        delay_kernel(grad_output.device)

        grad_weight = torch.ops.aten.mm(grad_output.t(), input_2d)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # sync grad_input here
        if handle is not None:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class TLLinear(torch.nn.Module):
    """Linear Layer with Two Level Tensor Parallelism

    # X (shard(1), replicate), W (shard(0), shard(1)) -> Z (replicate, shard(1))
    # X (replicate, shard(1)), W (shard(1), shard(0)) -> Z (shard(1), replicate)

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        module_mesh: mesh for the module.
        shard_strategy: shard strategy for weight
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 module_mesh: DeviceMesh,
                 shard_strategy: List[Placement],
                 input_is_shard: bool = False,
                 input_seq_shard: bool = False,
                 output_seq_shard: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype=torch.float32) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.mesh = module_mesh
        self.shard_strategy = shard_strategy
        self.params_dtype = params_dtype

        print(f"[{dist.get_rank()}]: {self.mesh}, {self.shard_strategy}")

        self.shard_input_size = self.input_size
        self.shard_output_size = self.output_size

        self.input_seq_shard = input_seq_shard
        self.output_seq_shard = output_seq_shard

        self.input_is_shard = input_is_shard
        if self.input_seq_shard:
            print(
                "[TLLinear] warning set self.input_is_shard=True because self.input_seq_shard=True")
            self.input_is_shard = True

        self.skip_bias_add = skip_bias_add

        assert (len(self.shard_strategy) == self.mesh.ndim == 2)
        self.forward_allreduce_mesh_dim = None
        self.backward_allreduce_mesh_dim = None
        for mesh_dim, shard_this_mesh_dim in enumerate(self.shard_strategy):
            if shard_this_mesh_dim.is_shard():
                if shard_this_mesh_dim.dim == 0:
                    self.shard_input_size = self.input_size // self.mesh.size(mesh_dim)
                    self.forward_allreduce_mesh_dim = mesh_dim
                if shard_this_mesh_dim.dim == 1:
                    self.shard_output_size = self.output_size // self.mesh.size(mesh_dim)
                    self.backward_allreduce_mesh_dim = mesh_dim

        if self.mesh.get_rank() == 0:
            print(f"shard weight shape: ({self.shard_input_size}, {self.shard_output_size})")
            print(f"self.forward_allreduce_mesh_dim: {self.forward_allreduce_mesh_dim}")
            print(f"self.backward_allreduce_mesh_dim: {self.backward_allreduce_mesh_dim}")

        self.weight = Parameter(
            torch.empty(self.shard_output_size,
                        self.shard_input_size,
                        device=torch.cuda.current_device(),
                        dtype=self.params_dtype))
        self.bias = Parameter(
            torch.empty(self.shard_output_size,
                        device=torch.cuda.current_device(),
                        dtype=self.params_dtype))

        self.linear_func = TLLinearFunc.apply

    def forward(self, input_):
        if self.input_seq_shard:
            assert len(input_.size()) > 2, "input_ must has sequence dimension"
        bias = self.bias if not self.skip_bias_add else None

        if not self.input_is_shard:
            input_ = shard_to_tp_region(input_, self.mesh, self.shard_strategy)
        output = self.linear_func(input_, self.weight, bias, self.mesh, self.input_seq_shard,
                                  self.output_seq_shard, self.forward_allreduce_mesh_dim,
                                  self.backward_allreduce_mesh_dim)
        return output


class ColumnLinear(TLLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 module_mesh: DeviceMesh,
                 seq_shard: bool = False,
                 params_dtype=torch.float32) -> None:

        assert module_mesh.ndim == 2, "still need 2d device mesh for megatron-style TP"
        shard_strategy = None
        if module_mesh.size(0) == 1:
            shard_strategy = [Shard(0), Shard(1)]
        elif module_mesh.size(1) == 1:
            shard_strategy = [Shard(1), Shard(0)]
        else:
            print("module_mesh not for ColumnLinear!")
            exit(-1)
        super().__init__(input_size,
                         output_size,
                         module_mesh,
                         shard_strategy,
                         input_is_shard=False,
                         input_seq_shard=seq_shard,
                         output_seq_shard=False,
                         params_dtype=params_dtype)


class RowLinear(TLLinear):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 module_mesh: DeviceMesh,
                 seq_shard: bool = False,
                 input_is_shard: bool = False,
                 params_dtype=torch.float32) -> None:

        assert module_mesh.ndim == 2, "still need 2d device mesh for megatron-style TP"
        shard_strategy = None
        if module_mesh.size(0) == 1:
            shard_strategy = [Shard(1), Shard(0)]
        elif module_mesh.size(1) == 1:
            shard_strategy = [Shard(0), Shard(1)]
        else:
            print("module_mesh not for RowLinear!")
            exit(-1)
        super().__init__(input_size,
                         output_size,
                         module_mesh,
                         shard_strategy,
                         input_is_shard=input_is_shard,
                         input_seq_shard=False,
                         output_seq_shard=seq_shard,
                         params_dtype=params_dtype)
