import torch
import torch.distributed as dist
from spmd import Shard

from atp.layer.feedforward import FeedForward, ATPFeedForward
import atp.distributed as atp_dist


def main():
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    atp_dist.init_mesh((world_size, 1))

    mesh = atp_dist.get_default_mesh()

    hidden_dim = 4096

    seq_shard = True

    ff_module = ATPFeedForward(hidden_dim, seq_shard=seq_shard).cuda().to(dtype=torch.float16)

    input_ = torch.randn(4, 1024 // mesh.get_dim_groups()[0].size(),
                         hidden_dim // mesh.get_dim_groups()[1].size()).to(
                             device="cuda", dtype=torch.float16).requires_grad_(True)

    def test_loop():
        output_ = ff_module(input_)

        grad_output = torch.randn_like(output_).cuda()

        print(f"output_.shape: {output_.shape}")

        output_.backward(grad_output)

        print(f"input_.grad.shape: {input_.grad.shape}")

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/megatron-ff-fp16'),
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