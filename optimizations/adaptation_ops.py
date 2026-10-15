"""Image-only shifts and DESA-inspired, bounded parameter-state controls."""
import hashlib
import cv2
import numpy as np
import torch


def image_seed(name, salt):
    return int.from_bytes(hashlib.sha256(f'{name}:{salt}'.encode()).digest()[:8], 'little')


def shift(image, kind, name):
    x=image.astype(np.float32)/255
    if kind=='clean':return x
    if kind=='noise':
        rng=np.random.default_rng(image_seed(name,'gaussian_sigma_0.08'))
        return np.clip(x+rng.normal(0,.08,x.shape).astype(np.float32),0,1)
    if kind=='blur':return cv2.GaussianBlur(x,(13,13),2.,borderType=cv2.BORDER_REFLECT_101)
    if kind=='dark':return .6*np.power(x,2.)
    raise ValueError(kind)


def phase(stream, position, total):
    if stream!='recovery':return stream,stream
    third=min(position*3//total,2)
    return ('clean','noise','clean')[third],('clean_before','noise','clean_after')[third]


def strong_view(image, boxes, name, seed):
    rng=np.random.default_rng(image_seed(name,f'strong_{seed}'))
    out=image.copy();b=boxes.copy()
    flip=bool(rng.random()<.5)
    if flip:
        out=np.ascontiguousarray(out[:,::-1]);b[:,0]=800-b[:,0];b[:,4]=(-b[:,4])%np.pi
    # No erasure/crop: retained pseudo targets remain visible and geometrically valid.
    contrast=float(rng.uniform(.9,1.1));brightness=float(rng.uniform(-.05,.05))
    out=np.clip((out-.5)*contrast+.5+brightness,0,1)
    out=np.clip(out+rng.normal(0,.01,out.shape).astype(np.float32),0,1)
    return out,b,dict(flip=flip,contrast=contrast,brightness=brightness)


def drift(parameters, source):
    ratios=[(p.detach()-s).abs().sum()/s.abs().sum().clamp_min(1e-6) for p,s in zip(parameters,source)]
    return float(torch.stack(ratios).mean())


def controls(teacher_drift, student_drift, count, confidence):
    # Equations 4--9 and coefficient settings from arXiv:2609.01795v1.
    r=min((count/300)*(-np.log(max(confidence,1e-8))),1.)
    d=max(teacher_drift,0.)
    low=.9990-.0060*(1-d*d/(d*d+.036**2))
    high=.9996-.0016*(1-d*d/(d*d+.021**2))
    momentum=high-(high-low)*r
    ds=max(student_drift,0.)
    anchor=.035*ds*ds/(ds*ds+.07**2)
    assert .993-1e-10<=momentum<=.9996+1e-10 and 0<=anchor<=.035
    return dict(uncertainty=r,momentum=momentum,anchor=anchor,lower=low,upper=high)


@torch.no_grad()
def interpolate_(parameters, reference, retain):
    torch._foreach_mul_(parameters,retain)
    torch._foreach_add_(parameters,reference,alpha=1-retain)


def normalized_targets(boxes):
    x=boxes.astype(np.float32).copy();x[:,:4]/=800;x[:,4]=(x[:,4]%np.pi)/np.pi
    return x


@torch.no_grad()
def source_normalized_distance(parameters, reference, source):
    """Weighted norm of the actual AdamW displacement, with fixed source scales."""
    ratios = [(p.detach() - r).abs().sum() / s.abs().sum().clamp_min(1e-6)
              for p, r, s in zip(parameters, reference, source)]
    return float(torch.stack(ratios).mean())


def mechanism_contract(previous_drift, adam_displacement, pre_anchor_drift,
                       anchor, anchored_drift, teacher_after,
                       teacher_upper, rollback):
    """Check identities/inequalities only; these are not AP guarantees."""
    update_upper = previous_drift + adam_displacement
    anchor_expected = (1 - anchor) * pre_anchor_drift
    tolerance = 2e-6 + 2e-5 * max(update_upper, anchor_expected, teacher_upper)
    update_violation = max(0., pre_anchor_drift - update_upper)
    anchor_residual = abs(anchored_drift - anchor_expected)
    teacher_violation = max(0., teacher_after - teacher_upper)
    if max(update_violation, anchor_residual, teacher_violation) > tolerance:
        raise ArithmeticError('Adaptation-state mechanism contract violated')
    return dict(student_drift_before_update=previous_drift,
                actual_adam_displacement=adam_displacement,
                student_update_drift_upper=update_upper,
                student_update_bound_violation=update_violation,
                anchor_contraction_expected=anchor_expected,
                anchor_contraction_absolute_residual=anchor_residual,
                anchor_active=anchor > 0.,
                teacher_drift_after_update=teacher_after,
                teacher_convexity_upper=teacher_upper,
                teacher_convexity_violation=teacher_violation,
                teacher_bound_case='source_reset' if rollback else 'ema_convexity',
                mechanism_tolerance=tolerance)
