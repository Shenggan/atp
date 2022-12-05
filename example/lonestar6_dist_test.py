import torch
import torch.distributed as dist
import oneccl_bindings_for_pytorch

from atp.utils import lonestar6_launch

prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/cpu-async'),
    record_shapes=False,
    with_stack=False)


def feedforward_aten_func(weight_1_, bias_1_, weight_2_, bias_2_, input_):
    addmm_o_1 = torch.ops.aten.addmm(bias_1_, input_, weight_1_)
    gelu_o = torch.ops.aten.gelu(addmm_o_1)
    out = torch.ops.aten.addmm(bias_2_, gelu_o, weight_2_)
    return out


def get_feedforward_para(dim, ratio=4, device="cpu"):
    weight_1_ = torch.rand(dim, ratio * dim, device=device)
    bias_1_ = torch.rand(ratio * dim, device=device)
    weight_2_ = torch.rand(ratio * dim, dim, device=device)
    bias_2_ = torch.rand(dim, device=device)
    return weight_1_, bias_1_, weight_2_, bias_2_


def test_dist(dim=1024):

    lonestar6_launch()

    para = get_feedforward_para(dim=dim, ratio=4, device="cpu")
    input_ = torch.rand(4 * 512, dim, device="cpu")
    tensor = torch.ones(10000, 256 * 16)

    prof.start()
    for _ in range(10):
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

        output_ = feedforward_aten_func(*para, input_)
        print(output_.shape)

        work.wait()
        print(tensor)
        prof.step()

    # del os.environ['OMP_NUM_THREADS']
    dist.barrier()
    prof.stop()


if __name__ == '__main__':
    test_dist()
