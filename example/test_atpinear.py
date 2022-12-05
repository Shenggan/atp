import torch
import torch.distributed as dist
from spmd import Shard

from atp.layer import ColumnLinear, RowLinear, ATPLinear
import atp.distributed as atp_dist
from atp.layer.mapping import gather_tensor


def test_column_linear():
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    atp_dist.init_mesh((world_size, 1))
    mesh = atp_dist.get_default_mesh()

    seq_shard = True

    atp_linear_1 = ColumnLinear(64, 64 * 4, mesh, seq_shard=seq_shard).cuda()
    global_linear = torch.nn.Linear(64, 64 * 4).cuda()

    print(f"ColumnLinear Parameter: {atp_linear_1.weight.shape}, {atp_linear_1.bias.shape}")
    print(f"Global Linear Parameter: {global_linear.weight.shape}, {global_linear.bias.shape}")

    tp_group = mesh.get_dim_groups()[0]

    input_ = torch.randn(4, 256, 64).cuda().requires_grad_(True)

    with torch.no_grad():
        dist.broadcast(input_, src=0, group=tp_group)
        if seq_shard:
            input_local = input_.chunk(world_size,
                                       dim=1)[dist.get_rank(tp_group)].clone().requires_grad_(True)
        else:
            input_local = input_.clone().requires_grad_(True)
        global_linear.weight.data = gather_tensor(atp_linear_1.weight.data,
                                                  tensor_dim=0,
                                                  group=tp_group)

        global_linear.bias.data = gather_tensor(atp_linear_1.bias.data, tensor_dim=0, group=tp_group)

    # forward
    print("input_shape: ", input_local.shape, input_.shape)
    dist_out = atp_linear_1(input_local)
    global_out = global_linear(input_)
    print("output_shape: ", dist_out.shape, global_out.shape)

    with torch.no_grad():
        atp_out = gather_tensor(dist_out, tensor_dim=-1, group=tp_group)

        if torch.allclose(global_out, atp_out):
            print("FORWARD PASS!!!")
        else:
            print("FORWARD FAILED!!!")

        # bacward
        global_grad = torch.randn_like(global_out)
        dist.broadcast(global_grad, src=0, group=tp_group)

        dist_grad = global_grad.chunk(world_size, dim=-1)[dist.get_rank(tp_group)].clone()

    dist_out.backward(dist_grad)
    global_out.backward(global_grad)

    if seq_shard:
        atp_input_grad = gather_tensor(input_local.grad, tensor_dim=1, group=tp_group)
    else:
        atp_input_grad = input_local.grad

    check_pass = False
    if torch.allclose(atp_input_grad, input_.grad, atol=1e-4):
        gather_weight_grad = gather_tensor(atp_linear_1.weight.grad, tensor_dim=0, group=tp_group)
        gather_bias_grad = gather_tensor(atp_linear_1.bias.grad, tensor_dim=0, group=tp_group)

        if torch.allclose(gather_weight_grad, global_linear.weight.grad,
                          atol=1e-4) and torch.allclose(
                              gather_bias_grad, global_linear.bias.grad, atol=1e-4):
            check_pass = True

    if check_pass:
        print("BACKWARD PASS!!!")
    else:
        print("BACKWARD FAILED!!!")


def test_row_linear():
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    atp_dist.init_mesh((world_size, 1))
    mesh = atp_dist.get_default_mesh()

    seq_shard = True

    atp_linear_1 = RowLinear(64, 64 * 4, mesh, input_is_shard=False, seq_shard=seq_shard).cuda()
    global_linear = torch.nn.Linear(64, 64 * 4).cuda()

    print(f"RowLinear Parameter: {atp_linear_1.weight.shape}, {atp_linear_1.bias.shape}")
    print(f"Global Linear Parameter: {global_linear.weight.shape}, {global_linear.bias.shape}")

    tp_group = mesh.get_dim_groups()[0]
    with torch.no_grad():
        input_ = torch.randn(4, 256, 64).cuda().requires_grad_(True)
        dist.broadcast(input_, src=0, group=tp_group)

        # input_local = input_.chunk(4, dim=-1)[dist.get_rank(tp_group)].clone().requires_grad_(True)
        input_local = input_.clone().requires_grad_(True)

        dist.broadcast(atp_linear_1.bias.data, src=0, group=tp_group)
        global_linear.weight.data = gather_tensor(atp_linear_1.weight.data,
                                                  tensor_dim=-1,
                                                  group=tp_group)

        global_linear.bias.data = atp_linear_1.bias.data

    # forward
    print("input_shape: ", input_local.shape, input_.shape)
    dist_out = atp_linear_1(input_local)
    global_out = global_linear(input_)
    print("output_shape: ", dist_out.shape, global_out.shape)

    atp_out = gather_tensor(dist_out, tensor_dim=1, group=tp_group)

    if torch.allclose(global_out, atp_out):
        print("FORWARD PASS!!!")
    else:
        print("FORWARD FAILED!!!")

    with torch.no_grad():
        # bacward
        global_grad = torch.randn_like(global_out)
        dist.broadcast(global_grad, src=0, group=tp_group)

        if seq_shard:
            dist_grad = global_grad.chunk(world_size, dim=1)[dist.get_rank(tp_group)].clone()
        else:
            dist_grad = global_grad.clone()

    dist_out.backward(dist_grad)
    global_out.backward(global_grad)

    check_pass = False

    if torch.allclose(input_local.grad, input_.grad, atol=1e-4):

        gather_weight_grad = gather_tensor(atp_linear_1.weight.grad, tensor_dim=1, group=tp_group)

        if torch.allclose(gather_weight_grad, global_linear.weight.grad,
                          atol=1e-4) and torch.allclose(
                              atp_linear_1.bias.grad, global_linear.bias.grad, atol=1e-4):
            check_pass = True

    if check_pass:
        print("BACKWARD PASS!!!")
    else:
        print("BACKWARD FAILED!!!")


def test_atp_linear():
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    atp_dist.init_mesh((world_size // 2, 2))
    mesh = atp_dist.get_default_mesh()

    # shard_strategy = [Shard(1), Shard(0)]
    shard_strategy = [Shard(0), Shard(1)]

    atp_linear_1 = ATPLinear(64, 64 * 4, mesh, shard_strategy=shard_strategy).cuda()
    global_linear = torch.nn.Linear(64, 64 * 4).cuda()

    print(f"ATPLinear Parameter: {atp_linear_1.weight.shape}, {atp_linear_1.bias.shape}")
    print(f"Global Linear Parameter: {global_linear.weight.shape}, {global_linear.bias.shape}")

    with torch.no_grad():
        # align input
        input_ = torch.randn(4, 256, 64).cuda().requires_grad_(True)
        dist.broadcast(input_, src=0)

        input_local = input_.clone().requires_grad_(True)

        # align weight and bias
        tmp_gather = atp_linear_1.weight.data
        for shard_idx, dim_shard in enumerate(shard_strategy):
            shard_group = mesh.get_dim_groups()[shard_idx]
            # weight is transposed in linear, so tensor_dim=1 - dim_shard.dim
            tmp_gather = gather_tensor(tmp_gather, tensor_dim=1 - dim_shard.dim, group=shard_group)
        global_linear.weight.data = tmp_gather

        bias_shard_dim = atp_linear_1.backward_allreduce_mesh_dim
        bais_shard_group = mesh.get_dim_groups()[bias_shard_dim]

        global_linear.bias.data = gather_tensor(atp_linear_1.bias.data,
                                                tensor_dim=0,
                                                group=bais_shard_group)

    # forward
    print("input_shape: ", input_local.shape, input_.shape)
    dist_out = atp_linear_1(input_local)
    global_out = global_linear(input_)
    print("output_shape: ", dist_out.shape, global_out.shape)

    with torch.no_grad():
        atp_out = gather_tensor(dist_out,
                               tensor_dim=-1,
                               group=mesh.get_dim_groups()[atp_linear_1.backward_allreduce_mesh_dim])

        if torch.allclose(global_out, atp_out):
            print("FORWARD PASS!!!")
        else:
            print("FORWARD FAILED!!!")

        # bacward
        global_grad = torch.randn_like(global_out)
        dist.broadcast(global_grad, src=0)

        output_shard_group = mesh.get_dim_groups()[atp_linear_1.backward_allreduce_mesh_dim]
        dist_grad = global_grad.chunk(output_shard_group.size(),
                                      dim=-1)[dist.get_rank(output_shard_group)].clone()

    dist_out.backward(dist_grad)
    global_out.backward(global_grad)

    check_pass = False

    if torch.allclose(input_local.grad, input_.grad, atol=1e-4):

        # align weight and bias
        gather_weight_grad = atp_linear_1.weight.grad
        for shard_idx, dim_shard in enumerate(shard_strategy):
            shard_group = mesh.get_dim_groups()[shard_idx]
            # weight is transposed in linear, so tensor_dim=1 - dim_shard.dim
            gather_weight_grad = gather_tensor(gather_weight_grad,
                                               tensor_dim=1 - dim_shard.dim,
                                               group=shard_group)

        bias_shard_dim = atp_linear_1.backward_allreduce_mesh_dim
        bais_shard_group = mesh.get_dim_groups()[bias_shard_dim]
        gather_bias_grad = gather_tensor(atp_linear_1.bias.grad,
                                         tensor_dim=0,
                                         group=bais_shard_group)

        if torch.allclose(gather_weight_grad, global_linear.weight.grad,
                          atol=1e-4) and torch.allclose(
                              gather_bias_grad, global_linear.bias.grad, atol=1e-4):
            check_pass = True

    if check_pass:
        print("BACKWARD PASS!!!")
    else:
        print("BACKWARD FAILED!!!")


if __name__ == '__main__':
    test_row_linear()
