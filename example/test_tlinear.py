import torch
import torch.distributed as dist

from tltp.layer import ColumnLinear, RowLinear, TLLinear
import tltp.distributed as tltp_dist
from tltp.layer.mapping import gather_tensor


def test_column_linear():
    dist.init_process_group("nccl")
    tltp_dist.init_mesh((4, 1))
    mesh = tltp_dist.get_default_mesh()

    tl_linear_1 = ColumnLinear(64, 64 * 4, mesh).cuda()
    global_linear = torch.nn.Linear(64, 64 * 4).cuda()

    print(f"ColumnLinear Parameter: {tl_linear_1.weight.shape}, {tl_linear_1.bias.shape}")
    print(f"Global Linear Parameter: {global_linear.weight.shape}, {global_linear.bias.shape}")

    tp_group = mesh.get_dim_groups()[0]

    input_ = torch.randn(4, 256, 64).cuda().requires_grad_(True)

    with torch.no_grad():
        dist.broadcast(input_, src=0, group=tp_group)
        input_local = input_.clone().requires_grad_(True)
        global_linear.weight.data = gather_tensor(tl_linear_1.weight.data,
                                                  tensor_dim=0,
                                                  group=tp_group)

        global_linear.bias.data = gather_tensor(tl_linear_1.bias.data, tensor_dim=0, group=tp_group)

    # forward
    dist_out = tl_linear_1(input_local)
    global_out = global_linear(input_)
    print(dist_out.shape, global_out.shape)

    with torch.no_grad():
        tl_out = gather_tensor(dist_out, tensor_dim=-1, group=tp_group)

        if torch.allclose(global_out, tl_out):
            print("FORWARD PASS!!!")
        else:
            print("FORWARD FAILED!!!")

        # bacward
        global_grad = torch.randn_like(global_out)
        dist.broadcast(global_grad, src=0, group=tp_group)

        dist_grad = global_grad.chunk(4, -1)[dist.get_rank(tp_group)].clone()

    dist_out.backward(dist_grad)
    global_out.backward(global_grad)

    check_pass = False
    if torch.allclose(input_local.grad, input_.grad, atol=1e-4):
        gather_weight_grad = gather_tensor(tl_linear_1.weight.grad, tensor_dim=0, group=tp_group)
        gather_bias_grad = gather_tensor(tl_linear_1.bias.grad, tensor_dim=0, group=tp_group)

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
    tltp_dist.init_mesh((4, 1))
    mesh = tltp_dist.get_default_mesh()

    tl_linear_1 = RowLinear(64, 64 * 4, mesh, input_is_shard=False).cuda()
    global_linear = torch.nn.Linear(64, 64 * 4).cuda()

    print(f"RowLinear Parameter: {tl_linear_1.weight.shape}, {tl_linear_1.bias.shape}")
    print(f"Global Linear Parameter: {global_linear.weight.shape}, {global_linear.bias.shape}")

    tp_group = mesh.get_dim_groups()[0]
    with torch.no_grad():
        input_ = torch.randn(4, 256, 64).cuda().requires_grad_(True)
        dist.broadcast(input_, src=0, group=tp_group)

        # input_local = input_.chunk(4, dim=-1)[dist.get_rank(tp_group)].clone().requires_grad_(True)
        input_local = input_.clone().requires_grad_(True)

        dist.broadcast(tl_linear_1.bias.data, src=0, group=tp_group)
        global_linear.weight.data = gather_tensor(tl_linear_1.weight.data,
                                                  tensor_dim=-1,
                                                  group=tp_group)

        global_linear.bias.data = tl_linear_1.bias.data

    dist_out = tl_linear_1(input_local)
    global_out = global_linear(input_)

    if torch.allclose(global_out, dist_out):
        print("FORWARD PASS!!!")
    else:
        print("FORWARD FAILED!!!")

    with torch.no_grad():
        # bacward
        global_grad = torch.randn_like(global_out)
        dist.broadcast(global_grad, src=0, group=tp_group)

        dist_grad = global_grad.clone()

    dist_out.backward(dist_grad)
    global_out.backward(global_grad)

    check_pass = False

    if torch.allclose(input_local.grad, input_.grad, atol=1e-4):

        gather_weight_grad = gather_tensor(tl_linear_1.weight.grad, tensor_dim=1, group=tp_group)

        if torch.allclose(gather_weight_grad, global_linear.weight.grad,
                          atol=1e-4) and torch.allclose(
                              tl_linear_1.bias.grad, global_linear.bias.grad, atol=1e-4):
            check_pass = True

    if check_pass:
        print("BACKWARD PASS!!!")
    else:
        print("BACKWARD FAILED!!!")


if __name__ == '__main__':
    test_row_linear()
