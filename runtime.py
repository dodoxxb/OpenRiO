"""Portable registrations. Model lineage is explicit; no workspace imports."""
import sys,random
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import paper_modules,training_support
import openrio_criterion,openrio_evaluator,openrio_exact_matcher_overlay,openrio_sampling_overlay
from engine.core import register
from engine.data.transforms.container import Compose
from engine.data.transforms.mosaic_obb import MosaicOBB
from engine.data.transforms import _transforms
for cls in (_transforms.ConvertPILImage,_transforms.ConvertBoxes):
    if '_transform' in cls.__dict__ and 'transform' not in cls.__dict__:
        cls.transform=lambda self,inpt,params:self._transform(inpt,params)
@register()
class PaperAlignedCompose(Compose):
    """Preserve the released epoch policy for all MosaicOBB subclasses.

    The public Compose identifies mosaics by two literal class names, which
    otherwise lets PaperOrientedDenseO2O bypass probability and epochs 78:148.
    """
    def stop_epoch_forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        epoch = sample[-1].epoch
        boundaries = self.policy['epoch']
        if not isinstance(boundaries, list) or len(boundaries) != 3:
            return super().stop_epoch_forward(*inputs)
        policy_ops = self.policy['ops']
        active = boundaries[0] <= epoch < boundaries[1]
        with_mosaic = active and random.random() <= self.mosaic_prob
        for transform in self.transforms:
            name = type(transform).__name__
            if name in policy_ops and (epoch < boundaries[0] or epoch >= boundaries[-1]):
                continue
            if isinstance(transform, MosaicOBB) and not with_mosaic:
                continue
            if name in ('RandomZoomOut', 'RandomIoUCrop', 'RandomZoomOutOBB', 'RandomIoUCropOBB') and with_mosaic:
                continue
            sample = transform(sample)
        return sample

def install_dn_zero_input():
    """Zero DN reference angles while preserving original criterion targets."""
    import engine.rtv4.denoising_obb as dn
    import engine.rtv4.rtdetrv2_obb_decoder as decoder
    if getattr(decoder.get_contrastive_denoising_training_group_obb, '_paper_aligned_zero_input', False):
        return
    original = decoder.get_contrastive_denoising_training_group_obb

    def zero_input(targets, *args, **kwargs):
        references = []
        for target in targets:
            copied = dict(target)
            if torch.is_tensor(target.get('boxes')):
                copied['boxes'] = target['boxes'].clone()
                copied['boxes'][..., 4] = 0
            references.append(copied)
        return original(references, *args, **kwargs)

    zero_input._paper_aligned_zero_input = True
    decoder.get_contrastive_denoising_training_group_obb = zero_input
    dn.get_contrastive_denoising_training_group_obb = zero_input

def register_lineage(lineage):
    if lineage=='compact4d':
        install_dn_zero_input()
        class CompactSineGeometryMLP(paper_modules.SineGeometryMLP):
            def __init__(self,hidden_dim,num_pos_feats=None,temperature=10000.,coordinate_order='yxwh',act='relu'):
                torch.nn.Module.__init__(self)
                self.num_pos_feats=hidden_dim//2 if num_pos_feats is None else num_pos_feats
                self.temperature=temperature;self.coordinate_order=coordinate_order
                self.mlp=paper_modules.MLP(4*self.num_pos_feats,hidden_dim,hidden_dim,2,act=act)
                for layer in self.mlp.layers:torch.nn.init.xavier_uniform_(layer.weight)
        paper_modules.SineGeometryMLP=CompactSineGeometryMLP
    elif lineage!='public5d':raise ValueError(lineage)
    from decoder_sdpa import install
    install()
    from decoder_checkpoint import install
    install(default_threshold=1024)
    from matcher_memory import install
    install(default_max_pairs=1_000_000)
