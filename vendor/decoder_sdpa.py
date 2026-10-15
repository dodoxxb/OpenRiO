"""Memory-efficient exact softmax attention for new decoder training only.

The caller discards attention weights.  Requesting need_weights=False dispatches
PyTorch MHA to SDPA, preserving its mask semantics, projections and parameters.
CUDA training explicitly requires EFFICIENT_ATTENTION: silent quadratic math
fallback is forbidden.  Inference retains the unmodified public MHA path.
"""
import functools
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend


class TrainingSDPAMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(query,key,value,key_padding_mask=key_padding_mask,
                need_weights=need_weights,attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,is_causal=is_causal)
        if query.dtype != torch.float32:
            raise RuntimeError('Paper-aligned SDPA contract requires FP32 training')
        backend=SDPBackend.EFFICIENT_ATTENTION if query.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backend):
            return super().forward(query,key,value,key_padding_mask=key_padding_mask,
                need_weights=False,attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,is_causal=is_causal)


def convert_decoder(model):
    """Change implementation class only; state dict and parameter objects persist."""
    from engine.rtv4.rtdetrv2_obb_decoder import TransformerDecoderLayer
    converted=0
    for layer in model.modules():
        if isinstance(layer,TransformerDecoderLayer):
            if not isinstance(layer.self_attn,TrainingSDPAMultiheadAttention):
                layer.self_attn.__class__=TrainingSDPAMultiheadAttention
            converted+=1
    return converted


def install():
    """Apply only to decoders subsequently constructed in this opt-in process."""
    from engine.rtv4.rtdetrv2_obb_decoder import TransformerDecoderLayer
    if getattr(TransformerDecoderLayer.__init__,'_paper_sdpa',False):
        return
    original=TransformerDecoderLayer.__init__
    @functools.wraps(original)
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.self_attn.__class__=TrainingSDPAMultiheadAttention
    initialize._paper_sdpa=True
    TransformerDecoderLayer.__init__=initialize
