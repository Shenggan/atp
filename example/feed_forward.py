import os

import torch
import torch.distributed as dist

from spmd.tensor import DeviceMesh

TEST = 'megatron'
# TEST = 'hybrid'

prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{TEST}'),
    record_shapes=False,
    with_stack=False)


def feedforward_aten_func(weight_1_, bias_1_, weight_2_, bias_2_, input_):
    addmm_o_1 = torch.ops.aten.addmm(bias_1_, input_, weight_1_)
    gelu_o = torch.ops.aten.gelu(addmm_o_1)
    out = torch.ops.aten.addmm(bias_2_, gelu_o, weight_2_)
    return out


def feedforward_aten_1d_func(weight_1_, bias_1_, weight_2_, bias_2_, input_, mesh):

    output_size = (input_.size(0) * mesh.size(), input_.size(1))
    gathered_tensor = torch.empty(*output_size, dtype=input_.dtype, device=input_.device).chunk(8)

    addmm_o_1 = torch.empty(input_.size(0) * mesh.size(),
                            weight_1_.size(1),
                            dtype=input_.dtype,
                            device=input_.device)

    addmm_o_1_list = addmm_o_1.chunk(8)

    handles = []
    for chunk_id, chunk_data in enumerate(input_.chunk(8)):
        handles.append(
            torch.distributed.all_gather_into_tensor(gathered_tensor[chunk_id],
                                                     chunk_data,
                                                     group=mesh.get_dim_groups()[0],
                                                     async_op=True))

    for handle in handles:
        handle.wait()
        torch.ops.aten.addmm.out(bias_1_,
                                 gathered_tensor[chunk_id],
                                 weight_1_,
                                 out=addmm_o_1_list[chunk_id])

    gelu_o = torch.ops.aten.gelu(addmm_o_1)

    reduce_scatter_size = (gelu_o.size(0) // 2, weight_2_.size(1))
    output = torch.empty(*reduce_scatter_size, device=input_.device, dtype=input_.dtype)
    output_chunk = output.chunk(8)
    handles = []
    for chunk_id, chunk_data in enumerate(gelu_o.chunk(8)):
        out = torch.ops.aten.addmm(bias_2_, chunk_data, weight_2_)
        handles.append(
            torch.distributed.reduce_scatter_tensor(output_chunk[chunk_id],
                                                    out,
                                                    group=mesh.get_dim_groups()[0],
                                                    async_op=True))

    for handle in handles:
        handle.wait()
    return output


def feedforward_aten_hybrid_func(weight_1_, bias_1_, weight_2_, bias_2_, input_, mesh):
    output_size = (input_.size(0) * mesh.size(0), input_.size(1))
    gathered_tensor = torch.empty(*output_size, dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(gathered_tensor,
                                             input_,
                                             group=mesh.get_dim_groups()[0])
    addmm_o_1 = torch.ops.aten.addmm(bias_1_, gathered_tensor, weight_1_)
    mesh.all_reduce(addmm_o_1, mesh_dim=1)

    gelu_o = torch.ops.aten.gelu(addmm_o_1)
    out = torch.ops.aten.addmm(bias_2_, gelu_o, weight_2_)
    reduce_scatter_size = (out.size(0) // 2, out.size(1))
    output = torch.empty(*reduce_scatter_size, device=out.device, dtype=out.dtype)
    torch.distributed.reduce_scatter_tensor(output, out, group=mesh.get_dim_groups()[0])
    return output


def get_feedforward_para(dim, ratio=4, device="cuda"):
    weight_1_ = torch.rand(dim, ratio * dim, device=device)
    bias_1_ = torch.rand(ratio * dim, device=device)
    weight_2_ = torch.rand(ratio * dim, dim, device=device)
    bias_2_ = torch.rand(dim, device=device)
    return weight_1_, bias_1_, weight_2_, bias_2_


def get_feedforward_para_1d(dim, mesh, ratio=4, device="cuda"):
    weight_1_ = torch.rand(dim, int(ratio * dim / mesh.size()), device=device)
    bias_1_ = torch.rand(int(ratio * dim / mesh.size()), device=device)
    weight_2_ = torch.rand(int(ratio * dim / mesh.size()), dim, device=device)
    bias_2_ = torch.rand(dim, device=device)

    return weight_1_, bias_1_, weight_2_, bias_2_


def get_feedforward_para_hybird(dim, mesh, ratio=4, device="cuda"):
    weight_1_ = torch.rand(int(dim / mesh.size(1)), int(ratio * dim / mesh.size(0)), device=device)
    bias_1_ = torch.rand(int(ratio * dim / mesh.size(0)), device=device)
    weight_2_ = torch.rand(int(ratio * dim / mesh.size(0)), int(dim / mesh.size(1)), device=device)
    bias_2_ = torch.rand(int(dim / mesh.size(1)), device=device)

    return weight_1_, bias_1_, weight_2_, bias_2_


def megatron_style(dim=4096):
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = "cuda"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    print(f"DIST INFO: {rank} {world_size}")

    mesh = DeviceMesh(device, torch.arange(world_size))

    para = get_feedforward_para_1d(dim=dim, ratio=4, device=device, mesh=mesh)

    input_ = torch.rand(int(4096 / mesh.size()), dim, device=device)

    prof.start()
    torch.cuda.synchronize()
    for _ in range(10):
        output_ = feedforward_aten_1d_func(*para, input_, mesh=mesh)
        print(output_.shape)
        torch.cuda.synchronize()
        prof.step()
    prof.stop()


def hybrid_style(dim=4096):
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = "cuda"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    print(f"DIST INFO: {rank} {world_size}")

    mesh_size_last_dim = 2
    mesh = DeviceMesh(device, torch.arange(world_size).reshape(-1, mesh_size_last_dim))

    print(f"Mesh Info: {mesh}")

    para = get_feedforward_para_hybird(dim=dim, ratio=4, device=device, mesh=mesh)

    input_ = torch.rand(int(4096 / mesh.size(0)), int(dim / mesh.size(1)), device=device)

    prof.start()
    torch.cuda.synchronize()
    for _ in range(10):
        output_ = feedforward_aten_hybrid_func(*para, input_, mesh=mesh)
        print(output_.shape)
        torch.cuda.synchronize()
        prof.step()
    prof.stop()


def test(dim=1024):
    para = get_feedforward_para(dim=dim, ratio=4, device="cuda")
    input_ = torch.rand(64 * 512, dim, device="cuda")
    output_ = feedforward_aten_func(*para, input_)
    print(output_.shape)


if __name__ == '__main__':
    if TEST == 'megatron':
        megatron_style()
    else:
        hybrid_style()
