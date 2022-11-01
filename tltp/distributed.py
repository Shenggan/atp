import os
import operator
from functools import reduce
from typing import Optional, List

import numpy as np

import torch
import torch.distributed as dist

from spmd import DeviceMesh


class GroupMember(object):
    ALL_TLTP_MESH: Optional[List[DeviceMesh]] = None
    TLTP_MESH: Optional[DeviceMesh] = None


def init_mesh(mesh_info, device_type="cuda"):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"DIST INFO: {rank} {world_size}")
    print(f"MESH INFO: {mesh_info}")

    mesh_size = reduce(operator.mul, mesh_info, 1)

    assert world_size % mesh_size == 0, 'world_size({}) is not divisible by mesh_size({})'.format(
        world_size, mesh_size)

    dp_size = world_size // mesh_size
    dp_rank = rank // mesh_size

    # Sub Mesh also need to create mesh on every rank.
    for dp_rank_ in range(dp_size):
        mesh_device_id = (np.array(range(mesh_size)) + (dp_rank_ * mesh_size)).reshape(mesh_info)
        if GroupMember.ALL_TLTP_MESH is None:
            GroupMember.ALL_TLTP_MESH = []
        GroupMember.ALL_TLTP_MESH.append(DeviceMesh(device_type, mesh_device_id))

    GroupMember.TLTP_MESH = GroupMember.ALL_TLTP_MESH[dp_rank]


def get_default_mesh():
    return GroupMember.TLTP_MESH
