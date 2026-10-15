"""Avoid allocating cross-image matching costs that Hungarian never uses.

For large matrices, invoke the unchanged public matcher once per image.  This
keeps its exact focal/L1/KLD/Hausdorff expressions, nan handling and SciPy LSAP.
The outer criterion still computes all losses on the full batch and unchanged
union assignments; this is not gradient microbatching or a positive budget.
"""
import functools
import torch


def install(default_max_pairs=1_000_000):
    from engine.rtv4.obb_matcher import HungarianMatcherOBB
    if getattr(HungarianMatcherOBB.forward,'_paper_per_image_memory',False):return
    original=HungarianMatcherOBB.forward
    @functools.wraps(original)
    @torch.no_grad()
    def forward(self,outputs,targets,return_topk=False):
        threshold=getattr(self,'_paper_max_cost_pairs',default_max_pairs)
        bs,queries=outputs['pred_logits'].shape[:2]
        pairs=bs*queries*sum(len(t['labels']) for t in targets)
        # Preserve the dormant public top-k contract without reinterpreting it.
        if return_topk or bs<=1 or threshold is None or pairs<=threshold:
            return original(self,outputs,targets,return_topk=return_topk)
        indices=[]
        for i,target in enumerate(targets):
            item={'pred_logits':outputs['pred_logits'][i:i+1],
                  'pred_boxes':outputs['pred_boxes'][i:i+1]}
            indices.append(original(self,item,[target],return_topk=False)['indices'][0])
        return {'indices':indices}
    forward._paper_per_image_memory=True
    HungarianMatcherOBB.forward=forward
