"""Recompute dense decoder layers instead of storing their activations.

Only decoder layers (LayerNorm, no BatchNorm) are checkpointed.  Whole batches,
all target/query tokens, masks, FP32 and parameter objects are preserved.
"""
import functools
import torch
from torch.utils.checkpoint import checkpoint


def install(default_threshold=1024):
    from engine.rtv4.rtdetrv2_obb_decoder import TransformerDecoderLayer
    if getattr(TransformerDecoderLayer.forward,'_paper_dense_checkpoint',False):return
    original=TransformerDecoderLayer.forward
    @functools.wraps(original)
    def forward(self,target,reference_points,memory,memory_spatial_shapes,
                attn_mask=None,memory_mask=None,query_pos_embed=None):
        threshold=getattr(self,'_paper_checkpoint_threshold',default_threshold)
        args=(target,reference_points,memory,memory_spatial_shapes,
              attn_mask,memory_mask,query_pos_embed)
        if self.training and torch.is_grad_enabled() and threshold is not None and target.shape[1]>threshold:
            return checkpoint(functools.partial(original,self),*args,
                              use_reentrant=False,preserve_rng_state=True)
        return original(self,*args)
    forward._paper_dense_checkpoint=True
    TransformerDecoderLayer.forward=forward
