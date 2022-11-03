import torch
import torch.distributed as dist
from spmd import Shard

from tltp.layer import TLLinear
import tltp.distributed as tltp_dist


def main():
    dist.init_process_group("nccl")
    tltp_dist.init_mesh((2, 2))
    mesh = tltp_dist.get_default_mesh()

    shard_strategy_1 = [Shard(1), Shard(0)]

    shard_strategy_2 = [Shard(0), Shard(1)]

    tl_linear_1 = TLLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=False,
                           input_seq_shard=False).cuda()
    tl_linear_2 = TLLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    tl_linear_3 = TLLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=True,
                           input_seq_shard=True).cuda()
    tl_linear_4 = TLLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    tl_linear_5 = TLLinear(2048,
                           2048 * 4,
                           mesh,
                           shard_strategy_1,
                           input_is_shard=True,
                           input_seq_shard=True).cuda()
    tl_linear_6 = TLLinear(2048 * 4,
                           2048,
                           mesh,
                           shard_strategy_2,
                           input_is_shard=True,
                           output_seq_shard=True).cuda()

    input_ = torch.randn(4, 1024, 2048).cuda().requires_grad_(True)

    def test_loop():
        output_ = tl_linear_2(tl_linear_1(input_))
        output_ = tl_linear_4(tl_linear_3(output_))
        output_ = tl_linear_6(tl_linear_5(output_))

        grad_output = torch.randn_like(output_).cuda()

        print(f"output_.shape: {output_.shape}")

        output_.backward(grad_output)

        print(f"input_.grad.shape: {input_.grad.shape}")

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/tllinear'),
        record_shapes=False,
        with_stack=False)

    prof.start()
    torch.cuda.synchronize()
    for _ in range(10):
        test_loop()
        torch.cuda.synchronize()
        prof.step()
    prof.stop()


if __name__ == '__main__':
    main()
