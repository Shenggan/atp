import os

import torch.distributed as dist


def lonestar6_launch():

    try:
        import oneccl_bindings_for_pytorch
    except:
        print("please install oneccl_bindings_for_pytorch for cpu experiments on lonestar6")

    os.environ['RANK'] = os.environ.get('MV2_COMM_WORLD_RANK', str(0))
    os.environ['WORLD_SIZE'] = os.environ.get('MV2_COMM_WORLD_SIZE', str(1))
    os.environ['MASTER_ADDR'] = os.environ.get('CCL_HOST', '127.0.0.1')
    os.environ['MASTER_PORT'] = '29500'  # your master port

    dist.init_process_group(backend='ccl')

    rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"LONESTAR6 DIST INFO: {rank} {world_size}")
