import math
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from functorch._src.named_members_polyfill import (_named_buffers, _named_parameters)

import atp.distributed as atp_dist
from atp.layer.mapping import gather_tensor

CHUNK = 1

SEQ_PARA = True

BWD_ASYNC = True

# torch.backends.cuda.matmul.allow_tf32 = True


def _scatter_tensor(tensor: torch.Tensor, dim, pg):
    local_tensor = tensor.chunk(dist.get_world_size(pg), dim)[dist.get_rank(pg)]
    return local_tensor.contiguous()


def aten_linear_forward(weight, bias, input_, output_buf=None):
    t = torch.ops.aten.t.default(weight)
    input_2d = input_.view(-1, input_.size()[-1])
    if bias is not None:
        if output_buf is not None:
            torch.ops.aten.addmm.out(bias, input_2d, t, out=output_buf)
        else:
            output_buf = torch.ops.aten.addmm.default(bias, input_2d, t)
    else:
        if output_buf is not None:
            torch.ops.aten.mm.out(input_2d, t, out=output_buf)
        else:
            output_buf = torch.ops.aten.mm.default(input_2d, t)
    output_buf = output_buf.view(*(input_.size()[:-1]), -1)
    return output_buf, input_2d, t


def aten_linear_backward(output_grad, weight_t, input_2d, reduce_group=None):
    output_grad_2d = output_grad.view(-1, output_grad.size()[-1])
    t_4 = torch.ops.aten.t.default(weight_t)
    mm = torch.ops.aten.mm.default(output_grad_2d, t_4)
    grad_input = mm.view(*(output_grad.size()[:-1]), -1)
    work = None
    if reduce_group:
        work = dist.all_reduce(grad_input,
                               op=dist.ReduceOp.SUM,
                               group=reduce_group,
                               async_op=BWD_ASYNC)
    t_5 = torch.ops.aten.t.default(output_grad_2d)
    mm_1 = torch.ops.aten.mm.default(t_5, input_2d)
    t_6 = torch.ops.aten.t.default(mm_1)
    sum_1 = torch.ops.aten.sum.dim_IntList(output_grad_2d, [0], True)
    grad_bias = torch.ops.aten.view.default(sum_1, [output_grad.size(-1)])
    grad_weight = torch.ops.aten.t.default(t_6)

    if work:
        work.wait()
    return grad_input, grad_weight, grad_bias


@torch.no_grad()
def aten_transformer_layer_forward(primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
                                   primals_7, primals_8, primals_9, primals_10, primals_11,
                                   primals_12, primals_13, dim_per_head, causal_mask):

    batch_, seq_, dim_ = primals_13.shape
    device = primals_13.device
    dtype = primals_13.dtype
    scale_ratio = math.sqrt(dim_per_head)

    mesh = atp_dist.get_default_mesh()

    native_layer_norm = torch.ops.aten.native_layer_norm.default(primals_13, [dim_], primals_1,
                                                                 primals_2, 1e-06)
    getitem = native_layer_norm[0]
    getitem_1 = native_layer_norm[1]
    getitem_2 = native_layer_norm[2]
    native_layer_norm = None

    #(TODO) gather getitem
    if SEQ_PARA:
        getitem = gather_tensor(getitem, tensor_dim=1, group=_get_default_group())
    getitem = _scatter_tensor(getitem, dim=-1, pg=mesh.get_dim_groups()[1])

    view_1 = torch.empty(getitem.size(0),
                         getitem.size(1),
                         primals_3.size()[0],
                         device=getitem.device,
                         dtype=getitem.dtype)

    chunk_tensors = [torch.chunk(t, CHUNK) for t in (getitem, view_1)]
    works = []
    for getitem_chunk, view_1_chunk in zip(*chunk_tensors):
        # ===== qkv linear ======
        view_1_chunk_2d = view_1_chunk.view(
            view_1_chunk.size(0) * view_1_chunk.size(1), view_1_chunk.size(2))
        aten_linear_forward(primals_3, None, getitem_chunk, output_buf=view_1_chunk_2d)
        works.append(
            dist.all_reduce(view_1_chunk,
                            op=dist.ReduceOp.SUM,
                            group=mesh.get_dim_groups()[1],
                            async_op=True))
    for work in works:
        work.wait()

    t = torch.ops.aten.t.default(primals_3)
    view = getitem.view(-1, getitem.size()[-1])

    view_1 += primals_4
    primals_3 = getitem = None

    view_1 = _scatter_tensor(view_1, dim=-1, pg=mesh.get_dim_groups()[1])

    num_heads = view_1.shape[-1] // (3 * dim_per_head)
    if SEQ_PARA:
        seq_ *= mesh.size(0) * mesh.size(1)

    # get q,k,v
    view_2 = torch.ops.aten.view.default(view_1, [batch_, seq_, num_heads, 3 * dim_per_head])
    view_1 = None
    permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3])
    view_2 = None
    split = torch.ops.aten.split.Tensor(permute, dim_per_head, -1)
    permute = None
    getitem_3 = split[0]
    getitem_4 = split[1]
    getitem_5 = split[2]
    split = None

    # attention core
    transpose = torch.ops.aten.transpose.int(getitem_4, -1, -2)
    getitem_4 = None
    expand = torch.ops.aten.expand.default(getitem_3, [batch_, num_heads, seq_, dim_per_head])
    getitem_3 = None
    clone = torch.ops.aten.clone.default(expand, memory_format=torch.contiguous_format)
    expand = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone,
                                                       [batch_ * num_heads, seq_, dim_per_head])
    clone = None
    expand_1 = torch.ops.aten.expand.default(transpose, [batch_, num_heads, dim_per_head, seq_])
    transpose = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format=torch.contiguous_format)
    expand_1 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1,
                                                         [batch_ * num_heads, dim_per_head, seq_])
    clone_1 = None
    bmm = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(bmm, [batch_, num_heads, seq_, seq_])
    bmm = None
    div = torch.ops.aten.div.Tensor(_unsafe_view_2, scale_ratio)
    _unsafe_view_2 = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-10000.0,
                                                         dtype=dtype,
                                                         layout=torch.strided,
                                                         device=device)
    _tensor_constant0 = causal_mask
    where = torch.ops.aten.where.self(_tensor_constant0, div, scalar_tensor)
    _tensor_constant0 = div = scalar_tensor = None
    _softmax = torch.ops.aten._softmax.default(where, -1, False)
    where = None
    expand_2 = torch.ops.aten.expand.default(_softmax, [batch_, num_heads, seq_, seq_])
    _reshape_alias = torch.ops.aten._reshape_alias.default(expand_2,
                                                           [batch_ * num_heads, seq_, seq_],
                                                           [seq_ * seq_, seq_, 1])
    expand_2 = None
    expand_3 = torch.ops.aten.expand.default(getitem_5, [batch_, num_heads, seq_, dim_per_head])
    getitem_5 = None
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format=torch.contiguous_format)
    expand_3 = None
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_2,
                                                         [batch_ * num_heads, seq_, dim_per_head])
    clone_2 = None

    bmm_1 = torch.ops.aten.bmm.default(_reshape_alias, _unsafe_view_3)
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(bmm_1,
                                                         [batch_, num_heads, seq_, dim_per_head])
    transpose_1 = torch.ops.aten.transpose.int(_unsafe_view_4, 1, 2)
    _unsafe_view_5 = torch.ops.aten.clone.default(transpose_1,
                                                  memory_format=torch.contiguous_format)
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(_unsafe_view_5,
                                                         [batch_, seq_, num_heads * dim_per_head])

    # ===== attn linear ======
    _unsafe_view_5 = gather_tensor(_unsafe_view_5, tensor_dim=-1, group=mesh.get_dim_groups()[1])

    view_4 = torch.empty(_unsafe_view_5.size(0),
                         _unsafe_view_5.size(1),
                         primals_5.size(0),
                         device=_unsafe_view_5.device,
                         dtype=_unsafe_view_5.dtype)

    chunk_tensors = [torch.chunk(t, CHUNK) for t in (view_4, _unsafe_view_5)]
    works = []
    for view_4_chunk, _unsafe_view_5_chunk in zip(*chunk_tensors):
        view_4_chunk_2d = view_4_chunk.view(
            view_4_chunk.size(0) * view_4_chunk.size(1), view_4_chunk.size(2))
        aten_linear_forward(weight=primals_5,
                            bias=None,
                            input_=_unsafe_view_5_chunk,
                            output_buf=view_4_chunk_2d)
        works.append(
            dist.all_reduce(view_4_chunk,
                            op=dist.ReduceOp.SUM,
                            group=mesh.get_dim_groups()[0],
                            async_op=True))

    for work in works:
        work.wait()

    t_1 = torch.ops.aten.t.default(primals_5)
    view_3 = _unsafe_view_5.view(-1, _unsafe_view_5.size()[-1])

    view_4 += primals_6
    view_4 = gather_tensor(view_4, tensor_dim=-1, group=mesh.get_dim_groups()[1])
    if SEQ_PARA:
        view_4 = _scatter_tensor(view_4, dim=1, pg=_get_default_group())

    primals_5 = primals_6 = _unsafe_view_5 = None

    add = torch.ops.aten.add.Tensor(primals_13, view_4)
    view_4 = None
    native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(add, [dim_], primals_7,
                                                                   primals_8, 1e-06)
    getitem_6 = native_layer_norm_1[0]
    getitem_7 = native_layer_norm_1[1]
    getitem_8 = native_layer_norm_1[2]
    native_layer_norm_1 = None

    if SEQ_PARA:
        getitem_6 = gather_tensor(getitem_6, tensor_dim=1, group=_get_default_group())
    getitem_6 = _scatter_tensor(getitem_6, dim=-1, pg=mesh.get_dim_groups()[1])

    view_6 = torch.empty(getitem_6.size(0),
                         getitem_6.size(1),
                         primals_9.size(0),
                         device=getitem_6.device,
                         dtype=getitem_6.dtype)

    chunk_tensors = [torch.chunk(t, CHUNK) for t in (view_6, getitem_6)]
    works = []
    for view_6_chunk, getitem_6_chunk in zip(*chunk_tensors):
        view_6_chunk_2d = view_6_chunk.view(
            view_6_chunk.size(0) * view_6_chunk.size(1), view_6_chunk.size(2))
        aten_linear_forward(weight=primals_9,
                            bias=None,
                            input_=getitem_6_chunk,
                            output_buf=view_6_chunk_2d)
        works.append(
            dist.all_reduce(view_6_chunk,
                            op=dist.ReduceOp.SUM,
                            group=mesh.get_dim_groups()[1],
                            async_op=True))

    for work in works:
        work.wait()

    t_2 = torch.ops.aten.t.default(primals_9)
    view_5 = getitem_6.view(-1, getitem_6.size()[-1])

    view_6 += primals_10

    gelu = torch.ops.aten.gelu.default(view_6)

    view_8 = torch.empty(gelu.size(0),
                         gelu.size(1),
                         primals_11.size(0),
                         device=gelu.device,
                         dtype=gelu.dtype)

    chunk_tensors = [torch.chunk(t, CHUNK) for t in (view_8, gelu)]
    works = []
    for view_8_chunk, gelu_chunk in zip(*chunk_tensors):
        view_8_chunk_2d = view_8_chunk.view(
            view_8_chunk.size(0) * view_8_chunk.size(1), view_8_chunk.size(2))
        aten_linear_forward(weight=primals_11,
                            bias=None,
                            input_=gelu_chunk,
                            output_buf=view_8_chunk_2d)
        works.append(
            dist.all_reduce(view_8_chunk,
                            op=dist.ReduceOp.SUM,
                            group=mesh.get_dim_groups()[0],
                            async_op=True))

    for work in works:
        work.wait()

    t_3 = torch.ops.aten.t.default(primals_11)
    view_7 = gelu.view(-1, gelu.size()[-1])

    primals_9 = primals_10 = getitem_6 = None
    view_8 += primals_12

    view_8 = gather_tensor(view_8, tensor_dim=-1, group=mesh.get_dim_groups()[1])
    if SEQ_PARA:
        view_8 = _scatter_tensor(view_8, dim=1, pg=_get_default_group())

    primals_11 = primals_12 = None

    add_1 = torch.ops.aten.add.Tensor(add, view_8)
    view_8 = None

    return [
        add_1, _reshape_alias, view, primals_2, view_7, primals_8, _unsafe_view, t_1, getitem_8,
        add, primals_7, t_2, t_3, _unsafe_view_3, view_6, primals_1, _softmax, getitem_2, getitem_7,
        view_5, _unsafe_view_1, t, getitem_1, view_3, primals_13
    ]


@torch.no_grad()
def aten_transformer_layer_backward(_reshape_alias, view, primals_2, view_7, primals_8,
                                    _unsafe_view, t_1, getitem_8, add, primals_7, t_2, t_3,
                                    _unsafe_view_3, view_6, primals_1, _softmax, getitem_2,
                                    getitem_7, view_5, _unsafe_view_1, t, getitem_1, view_3,
                                    primals_13, tangents_1, dim_per_head, causal_mask):

    device = primals_13.device
    dtype = primals_13.dtype
    batch_, seq_, dim_ = primals_13.shape
    device = primals_13.device
    scale_ratio = math.sqrt(dim_per_head)

    mesh = atp_dist.get_default_mesh()

    detach_1 = torch.ops.aten.detach.default(_softmax)
    _softmax = None
    detach_2 = torch.ops.aten.detach.default(detach_1)
    detach_1 = None

    if SEQ_PARA:
        seq_ *= mesh.size(0) * mesh.size(1)

    tangents_1_gather = tangents_1
    if SEQ_PARA:
        tangents_1_gather = gather_tensor(tangents_1, tensor_dim=1, group=_get_default_group())
    tangents_1_gather = _scatter_tensor(tangents_1_gather, dim=-1, pg=mesh.get_dim_groups()[1])

    # ===== ff linear 2 ======
    _reshape_alias_2, t_7, view_9 = aten_linear_backward(tangents_1_gather,
                                                         t_3,
                                                         view_7,
                                                         reduce_group=mesh.get_dim_groups()[1])
    t_3 = view_7 = None

    gelu_backward = torch.ops.aten.gelu_backward.default(_reshape_alias_2, view_6)
    _reshape_alias_2 = view_6 = None

    # ===== ff linear 1 ======
    _reshape_alias_4, t_11, view_10 = aten_linear_backward(gelu_backward,
                                                           t_2,
                                                           view_5,
                                                           reduce_group=mesh.get_dim_groups()[0])
    gelu_backward = t_2 = view_5 = None

    _reshape_alias_4 = gather_tensor(_reshape_alias_4,
                                     tensor_dim=-1,
                                     group=mesh.get_dim_groups()[1])
    if SEQ_PARA:
        _reshape_alias_4 = _scatter_tensor(_reshape_alias_4, dim=1, pg=_get_default_group())

    native_layer_norm_backward = torch.ops.aten.native_layer_norm_backward.default(
        _reshape_alias_4, add, [dim_], getitem_7, getitem_8, primals_7, primals_8,
        [True, True, True])
    _reshape_alias_4 = add = getitem_7 = getitem_8 = primals_7 = primals_8 = None
    getitem_9 = native_layer_norm_backward[0]
    getitem_10 = native_layer_norm_backward[1]
    getitem_11 = native_layer_norm_backward[2]
    native_layer_norm_backward = None
    add_2 = torch.ops.aten.add.Tensor(tangents_1, getitem_9)
    tangents_1 = getitem_9 = None

    if SEQ_PARA:
        add_2 = gather_tensor(add_2, tensor_dim=1, group=_get_default_group())
    add_2 = _scatter_tensor(add_2, dim=-1, pg=mesh.get_dim_groups()[1])
    # ===== attn linear ======
    _reshape_alias_6, t_15, view_11 = aten_linear_backward(add_2,
                                                           t_1,
                                                           view_3,
                                                           reduce_group=mesh.get_dim_groups()[1])
    add_2 = t_1 = view_3 = None
    _reshape_alias_6 = _scatter_tensor(_reshape_alias_6, dim=-1, pg=mesh.get_dim_groups()[1])

    num_heads = _reshape_alias_6.shape[-1] // dim_per_head

    _reshape_alias_7 = torch.ops.aten._reshape_alias.default(
        _reshape_alias_6, [batch_, seq_, num_heads, dim_per_head],
        [seq_ * num_heads * dim_per_head, num_heads * dim_per_head, dim_per_head, 1])
    _reshape_alias_6 = None
    transpose_2 = torch.ops.aten.transpose.int(_reshape_alias_7, 1, 2)
    _reshape_alias_7 = None
    clone_4 = torch.ops.aten.clone.default(transpose_2, memory_format=torch.contiguous_format)
    transpose_2 = None
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_4,
                                                         [batch_ * num_heads, seq_, dim_per_head])
    clone_4 = None
    transpose_3 = torch.ops.aten.transpose.int(_reshape_alias, 1, 2)
    _reshape_alias = None
    bmm_2 = torch.ops.aten.bmm.default(transpose_3, _unsafe_view_6)
    transpose_3 = None
    transpose_4 = torch.ops.aten.transpose.int(_unsafe_view_3, 1, 2)
    _unsafe_view_3 = None
    bmm_3 = torch.ops.aten.bmm.default(_unsafe_view_6, transpose_4)
    _unsafe_view_6 = transpose_4 = None
    _reshape_alias_8 = torch.ops.aten._reshape_alias.default(
        bmm_2, [batch_, num_heads, seq_, dim_per_head],
        [num_heads * seq_ * dim_per_head, seq_ * dim_per_head, dim_per_head, 1])
    bmm_2 = None
    _reshape_alias_9 = torch.ops.aten._reshape_alias.default(
        bmm_3, [batch_, num_heads, seq_, seq_], [num_heads * seq_ * seq_, seq_ * seq_, seq_, 1])
    bmm_3 = None
    detach_3 = torch.ops.aten.detach.default(detach_2)
    detach_2 = None
    detach_4 = torch.ops.aten.detach.default(detach_3)
    detach_3 = None
    _softmax_backward_data = torch.ops.aten._softmax_backward_data.default(
        _reshape_alias_9, detach_4, -1, dtype)
    _reshape_alias_9 = detach_4 = None
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(0,
                                                           dtype=dtype,
                                                           layout=torch.strided,
                                                           device=device)
    _tensor_constant0_1 = causal_mask
    where_1 = torch.ops.aten.where.self(_tensor_constant0_1, _softmax_backward_data,
                                        scalar_tensor_1)
    _tensor_constant0_1 = _softmax_backward_data = scalar_tensor_1 = None
    div_1 = torch.ops.aten.div.Tensor(where_1, scale_ratio)
    where_1 = None
    _reshape_alias_10 = torch.ops.aten._reshape_alias.default(div_1,
                                                              [batch_ * num_heads, seq_, seq_],
                                                              [seq_ * seq_, seq_, 1])
    div_1 = None
    transpose_5 = torch.ops.aten.transpose.int(_unsafe_view, 1, 2)
    _unsafe_view = None
    bmm_4 = torch.ops.aten.bmm.default(transpose_5, _reshape_alias_10)
    transpose_5 = None
    transpose_6 = torch.ops.aten.transpose.int(_unsafe_view_1, 1, 2)
    _unsafe_view_1 = None
    bmm_5 = torch.ops.aten.bmm.default(_reshape_alias_10, transpose_6)
    _reshape_alias_10 = transpose_6 = None
    _reshape_alias_11 = torch.ops.aten._reshape_alias.default(
        bmm_4, [batch_, num_heads, dim_per_head, seq_],
        [num_heads * dim_per_head * seq_, dim_per_head * seq_, seq_, 1])
    bmm_4 = None
    _reshape_alias_12 = torch.ops.aten._reshape_alias.default(
        bmm_5, [batch_, num_heads, seq_, dim_per_head],
        [num_heads * seq_ * dim_per_head, seq_ * dim_per_head, dim_per_head, 1])
    bmm_5 = None
    transpose_7 = torch.ops.aten.transpose.int(_reshape_alias_11, -1, -2)
    _reshape_alias_11 = None
    cat = torch.ops.aten.cat.default([_reshape_alias_12, transpose_7, _reshape_alias_8], 3)
    _reshape_alias_12 = transpose_7 = _reshape_alias_8 = None
    permute_1 = torch.ops.aten.permute.default(cat, [0, 2, 1, 3])
    cat = None
    clone_5 = torch.ops.aten.clone.default(permute_1, memory_format=torch.contiguous_format)
    permute_1 = None
    _unsafe_view_7 = torch.ops.aten._unsafe_view.default(
        clone_5, [batch_, seq_, 3 * num_heads * dim_per_head])
    clone_5 = None

    # ===== qkv linear ======
    _unsafe_view_7 = gather_tensor(_unsafe_view_7, tensor_dim=-1, group=mesh.get_dim_groups()[1])
    _reshape_alias_14, t_19, view_12 = aten_linear_backward(_unsafe_view_7,
                                                            t,
                                                            view,
                                                            reduce_group=mesh.get_dim_groups()[0])
    _unsafe_view_7 = t = view = None

    _reshape_alias_14 = gather_tensor(_reshape_alias_14,
                                      tensor_dim=-1,
                                      group=mesh.get_dim_groups()[1])
    if SEQ_PARA:
        _reshape_alias_14 = _scatter_tensor(_reshape_alias_14, dim=1, pg=_get_default_group())

    native_layer_norm_backward_1 = torch.ops.aten.native_layer_norm_backward.default(
        _reshape_alias_14, primals_13, [dim_], getitem_1, getitem_2, primals_1, primals_2,
        [False, True, True])
    _reshape_alias_14 = primals_13 = getitem_1 = getitem_2 = primals_1 = primals_2 = None
    getitem_13 = native_layer_norm_backward_1[1]
    getitem_14 = native_layer_norm_backward_1[2]
    native_layer_norm_backward_1 = None

    return [
        getitem_13, getitem_14, t_19, view_12, t_15, view_11, getitem_10, getitem_11, t_11, view_10,
        t_7, view_9, None
    ]


class ATPTransformerLayerFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        # args: *para_, input_tensor_, self.num_heads, *buf_

        out = aten_transformer_layer_forward(*args)
        ctx.save_for_backward(*(out[1:]))

        ctx.causal_mask = args[-1]
        ctx.num_heads = args[-2]

        return out[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        para_grad = aten_transformer_layer_backward(*ctx.saved_tensors, grad_output, ctx.num_heads,
                                                    ctx.causal_mask)
        return *para_grad, None, None, None


class AtenTransformerLayer(nn.Module):

    def __init__(self, dim, dim_per_head, seq_length, mlp_ratio=4) -> None:
        super().__init__()

        self.dim = dim
        self.dim_per_head = dim_per_head
        self.seq_length = seq_length
        self.ffn_hidden_size = mlp_ratio * dim

        self.mesh = atp_dist.get_default_mesh()

        # init parameter
        factory_kwargs = {'device': None, 'dtype': None}
        self.norm1_w = nn.Parameter(torch.empty(self.dim, **factory_kwargs))
        self.norm1_b = nn.Parameter(torch.empty(self.dim, **factory_kwargs))

        self.qkv_linear_w = nn.Parameter(
            torch.empty((3 * self.dim // self.mesh.size(0), self.dim // self.mesh.size(1)),
                        **factory_kwargs))
        self.qkv_linear_b = nn.Parameter(
            torch.empty(3 * self.dim // self.mesh.size(0), **factory_kwargs))

        self.att_linear_w = nn.Parameter(
            torch.empty((self.dim // self.mesh.size(1), self.dim // self.mesh.size(0)),
                        **factory_kwargs))
        self.att_linear_b = nn.Parameter(
            torch.empty(self.dim // self.mesh.size(1), **factory_kwargs))

        self.norm2_w = nn.Parameter(torch.empty(self.dim, **factory_kwargs))
        self.norm2_b = nn.Parameter(torch.empty(self.dim, **factory_kwargs))

        self.ff1_linear_w = nn.Parameter(
            torch.empty((self.ffn_hidden_size // self.mesh.size(0), self.dim // self.mesh.size(1)),
                        **factory_kwargs))
        self.ff1_linear_b = nn.Parameter(
            torch.empty((self.ffn_hidden_size // self.mesh.size(0)), **factory_kwargs))

        self.ff2_linear_w = nn.Parameter(
            torch.empty((self.dim // self.mesh.size(1), self.ffn_hidden_size // self.mesh.size(0)),
                        **factory_kwargs))
        self.ff2_linear_b = nn.Parameter(
            torch.empty((self.dim // self.mesh.size(1)), **factory_kwargs))

        causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.uint8))
        causal_mask = causal_mask.view(1, 1, seq_length, seq_length).bool()

        self.register_buffer('causal_mask', causal_mask)

    def forward(self, input_tensor_):
        para_ = list(dict(_named_parameters(self, remove_duplicate=False)).values())
        buf_ = list(dict(_named_buffers(self, remove_duplicate=False)).values())
        return ATPTransformerLayerFunc.apply(*para_, input_tensor_, self.dim_per_head, *buf_)


class AtenTransformer(nn.Module):

    def __init__(self, dim, dim_per_head, seq_length, layer, mlp_ratio=4):
        super(AtenTransformer, self).__init__()
        self.layers = nn.ModuleList([
            AtenTransformerLayer(dim=dim,
                                 dim_per_head=dim_per_head,
                                 seq_length=seq_length,
                                 mlp_ratio=mlp_ratio) for _ in range(layer)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():

    parser = argparse.ArgumentParser(description='ATen Transformer Implenmenattion for Adaptive TP')

    parser.add_argument("--batch-size", default=4, type=int, help='batch size')
    parser.add_argument("--chunk-size", default=1, type=int, help='chunk size')
    parser.add_argument("--seq-size", default=2048, type=int, help='seq length')
    parser.add_argument("--dim", default=12288, type=int, help='hidden dim size')
    parser.add_argument("--heads", default=48, type=int, help='num of heads')
    parser.add_argument("--layer", default=2, type=int, help='num of heads')

    parser.add_argument("--sm-size", default=2, type=int, help='sub mesh size')

    parser.add_argument('--seq-para', action='store_true', help='use sequence parallelism.')
    parser.add_argument('--fp16', action='store_true', help='use half precision.')
    parser.add_argument('--prof', action='store_true', help='use half precision.')
    parser.add_argument('--cpu', action='store_true', help='run cpu version.')
    parser.add_argument('--no-bwd-async', action='store_true', help='run cpu version.')

    args = parser.parse_args()

    global CHUNK
    global SEQ_PARA
    global BWD_ASYNC

    CHUNK = args.chunk_size
    SEQ_PARA = args.seq_para
    BWD_ASYNC = not args.no_bwd_async

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    atp_dist.init_mesh((world_size // args.sm_size, args.sm_size))

    device = torch.device("cuda")
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16

    mesh = atp_dist.get_default_mesh()

    if dist.get_rank() == 0:
        print(args)

    batch_, seq_, dim_ = args.batch_size, args.seq_size, args.dim
    dim_per_head = args.dim // args.heads
    test_module = AtenTransformer(dim=dim_,
                                  dim_per_head=dim_per_head,
                                  layer=args.layer,
                                  seq_length=seq_).to(device=device, dtype=dtype)

    input_ = torch.ones(batch_, seq_, dim_).to(device=device, dtype=dtype)
    if SEQ_PARA:
        input_ = torch.ones(batch_, seq_ // (mesh.size(0) * mesh.size(1)), dim_).to(device=device,
                                                                                    dtype=dtype)

    output_grad = torch.rand_like(input_)

    if args.prof:
        prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1,
                                                                       warmup=2,
                                                                       active=3,
                                                                       repeat=1),
                                      on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                          f'./log/atp-{args.sm_size}-{args.chunk_size}/'),
                                      profile_memory=False,
                                      record_shapes=False,
                                      with_stack=False)

        prof.start()

    for _ in range(6):
        fwd_start = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output_ = test_module(input_)

        bwd_start.record()
        output_.backward(output_grad)
        end.record()

        dist.barrier()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print(
                f"Step Time: {fwd_start.elapsed_time(end)}; Fwd Time: {fwd_start.elapsed_time(bwd_start)}; Bwd Time: {bwd_start.elapsed_time(end)}"
            )

        if args.prof:
            prof.step()

    if args.prof:
        prof.stop()

    dist.barrier()
    dist.destroy_process_group(dist.group.WORLD)


if __name__ == '__main__':
    main()
