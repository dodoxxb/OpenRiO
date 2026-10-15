"""Causal online OBB adaptation: emit prediction, then update for future images.

This process cannot open annotation files. Ground-truth evaluation is separate.
"""
import argparse,copy,fcntl,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import cv2,numpy as np,torch,yaml
from torch import nn
H=Path(__file__).resolve().parent;C=H.parent
from . import adaptation_ops as ops


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deny_annotations(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace('\\','/')
        if '/Annotations/' in name or name.endswith('independent_gt.pt'):
            raise PermissionError('Adaptation process is forbidden to read ground truth: '+name)


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).unsqueeze(0).cuda()


def select_norms(model):
    selected={}
    for module_name,module in model.named_modules():
        if isinstance(module,(nn.BatchNorm2d,nn.LayerNorm,nn.GroupNorm)):
            for name,p in module.named_parameters(recurse=False):
                p.requires_grad_(True);selected[module_name+'.'+name]=p
    assert selected
    return selected


def pseudo_targets(pred):
    from engine.data.dataset.mmeval import rbox_iou
    ids=np.flatnonzero(pred['scores']>=.4);keep=[]
    for c in np.unique(pred['labels'][ids]):
        ix=ids[pred['labels'][ids]==c];ix=ix[np.argsort(-pred['scores'][ix],kind='stable')]
        overlap=rbox_iou(pred['boxes'][ix],pred['boxes'][ix]);active=np.ones(len(ix),bool)
        for j in range(len(ix)):
            if active[j]:keep.append(int(ix[j]));active[j+1:] &= overlap[j,j+1:]<=.5
    return np.asarray(sorted(keep,key=lambda j:-pred['scores'][j]),int)


LOSS_KEYS = {
    'full': frozenset(['loss_focal', 'loss_l1', 'loss_kld']),
    'classification': frozenset(['loss_focal']),
    'geometry': frozenset(['loss_l1', 'loss_kld']),
}


def loss_final(criterion, output, target, mode):
    """Select native weighted loss terms; matching and weight values never change."""
    if mode not in LOSS_KEYS:
        raise ValueError('Unknown loss_mode: ' + str(mode))
    expected = LOSS_KEYS['full']
    if set(criterion.weight_dict) != expected:
        raise ValueError('Unexpected native weight keys: ' + repr(sorted(criterion.weight_dict)))
    indices = criterion.matcher(output, [target])['indices']
    weighted = {}
    for kind in criterion.losses:
        values = criterion.get_loss(kind, output, [target], indices, max(len(target['labels']), 1))
        unknown = set(values) - expected
        if unknown:
            raise ValueError('Unexpected native loss keys: ' + repr(sorted(unknown)))
        for key, value in values.items():
            if key in weighted:
                raise ValueError('Duplicate native loss key: ' + key)
            weighted[key] = value * criterion.weight_dict[key]
    if set(weighted) != expected:
        raise ValueError('Missing native loss keys: ' + repr(sorted(expected - set(weighted))))
    selected = {key: value for key, value in weighted.items() if key in LOSS_KEYS[mode]}
    return sum(selected.values()), selected


def run(spec,limit=0):
    probe = bool(limit)
    if spec.get('loss_mode') not in LOSS_KEYS: raise ValueError('Explicit supported loss_mode required')
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'status.json').exists(),'Fresh output required'
    for path,digest in spec.get('code_hashes',{}).items():assert sha(path)==digest,path
    save(out/'spec.json',spec)
    sys.addaudithook(deny_annotations)
    random.seed(spec['order_seed']);np.random.seed(spec['order_seed']);torch.manual_seed(spec['order_seed'])
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(torch.device(spec.get('device','cuda:0')))
    torch.cuda.set_per_process_memory_fraction(spec['allocator_fraction'])
    assert torch.cuda.mem_get_info()[0]>(spec['admission_free_mib']-256)*2**20
    root=Path(spec['data_root'])
    files=[str(root/'ImageSets/Main/test.txt')]
    names=Path(files[0]).read_text().split()
    assert len(names)==len(set(names))==11738
    ds={'img_folder':str(root/'JPEGImages-test')}
    import openrio
    assert Path(openrio.__file__).resolve()==C/'openrio.py'
    cfg,student,post,identity=openrio.load_model(spec['model'],torch.device(spec.get('device','cuda:0')),root)
    assert identity['sha256']==spec['checkpoint_sha256']
    teacher=copy.deepcopy(student).requires_grad_(False).eval()
    initial={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    selected=select_norms(student);params=list(selected.values());source=[p.detach().clone() for p in params]
    tp=dict(teacher.named_parameters());teacher_params=[tp[k] for k in selected]
    optimizer=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=0)
    criterion=cfg.criterion.cuda().eval()
    order=np.random.default_rng(spec['order_seed']).permutation(len(names)).tolist()
    full_order=list(order)
    if limit:order=order[:limit]
    save(out/'manifest.json',dict(identity=identity,images=names,order=order,stream=spec['stream'],
        test_list_sha256=sha(files[0]),full_order=full_order,diagnostic_limit=limit,
        selected_parameters={k:list(v.shape) for k,v in selected.items()},trainable=sum(p.numel() for p in params),
        ground_truth_access='blocked by process audit hook',batch_size=1,emit_before_update=True,
        bn_buffers='frozen; eval mode for both teacher and student',precision='FP32, matmul/cuDNN TF32 disabled',
        losses='native final-layer selected terms; no DN/auxiliary losses',loss_mode=spec['loss_mode'],
        selected_loss_keys=sorted(LOSS_KEYS[spec['loss_mode']]),
        mechanism='matched loss-term intervention; native weights/matcher unchanged',
        stream_manifest_sha256={f:sha(f) for f in files},model_ancestry='frozen release candidate; historical DIOR test exposure remains disclosed'))
    chunks={};history=[];updates=0;rollbacks=0;source_only=False;empty=0;started=time.monotonic();norms_with_grad=set();last={}
    for position,index in enumerate(order):
        name=names[index];kind,phase=ops.phase(spec['stream'],position,len(order))
        raw=cv2.imread(str(Path(ds['img_folder'])/(name+'.jpg')));assert raw is not None and raw.shape[:2]==(800,800)
        image=ops.shift(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),kind,name);x=to_tensor(image)
        with torch.no_grad():
            result=post(teacher(x),torch.tensor([[800,800]],device='cuda'))[0]
            pred={k:result[k].detach().cpu().numpy() for k in ['boxes','scores','labels']}
        assert all(np.isfinite(v).all() for v in pred.values()) and len(pred['scores'])<=300
        # Commit the prediction before any update from this image.
        chunks[index]=dict(bboxes=pred['boxes'],scores=pred['scores'],labels=pred['labels'],phase=phase,position=position)
        info=dict(position=position,index=index,phase=phase,updated=False,rollback=False)
        if spec['arm']!='frozen' and not source_only:
            keep=pseudo_targets(pred);info['pseudo_count']=len(keep)
            if len(keep):
                confidence=float(pred['scores'][keep].mean());td=ops.drift(teacher_params,source)
                momentum=ops.controls(td,0,len(keep),confidence)['momentum'] if spec['arm'].startswith('dynamic') else .999
                view,boxes,augmentation=ops.strong_view(image,pred['boxes'][keep],name,spec['order_seed'])
                target=dict(boxes=torch.from_numpy(ops.normalized_targets(boxes)).cuda(),labels=torch.from_numpy(pred['labels'][keep].astype(np.int64)).cuda())
                optimizer.zero_grad(set_to_none=True);output=student(to_tensor(view));loss,parts=loss_final(criterion,output,target,spec['loss_mode'])
                assert torch.isfinite(loss),'Nonfinite pseudo loss: stop, no automatic retry'
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,.1);assert torch.isfinite(norm),'Nonfinite gradient'
                if probe:
                    norms_with_grad.update(k for k,p in selected.items() if p.grad is not None and torch.count_nonzero(p.grad))
                before_step=[p.detach().clone() for p in params]
                previous_sd=ops.drift(before_step,source)
                optimizer.step();sd=ops.drift(params,source)
                adam_displacement=ops.source_normalized_distance(params,before_step,source)
                del before_step
                anchor=ops.controls(td,sd,len(keep),confidence)['anchor'] if 'anchor' in spec['arm'] else 0.
                if anchor:ops.interpolate_(params,source,1-anchor)
                anchored=ops.drift(params,source);rollback=spec['arm'].endswith('rollback') and anchored>spec['rollback_drift']
                if rollback:
                    # Reset only adapted parameters; all other model state was frozen.
                    ops.interpolate_(params,source,0.);ops.interpolate_(teacher_params,source,0.);optimizer.state.clear();rollbacks+=1
                    # A repeatedly triggered guard falls back to source predictions
                    # for the remainder; all images are still evaluated.
                    if rollbacks>=spec['maximum_rollbacks']:source_only=True
                else:ops.interpolate_(teacher_params,params,momentum)
                post_teacher=ops.drift(teacher_params,source)
                teacher_upper=0. if rollback else momentum*td+(1-momentum)*anchored
                mechanism=ops.mechanism_contract(previous_sd,adam_displacement,sd,anchor,anchored,post_teacher,teacher_upper,rollback)
                info.update(mechanism)
                updates+=1;info.update(updated=True,loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm),teacher_drift=td,student_drift_before_anchor=sd,student_drift=anchored,ema=momentum,anchor=anchor,pseudo_confidence=confidence,rollback=rollback,source_only=source_only)
                del output,loss,parts,target
            else:empty+=1
        history.append(info);last=info
        if len(chunks)>=256 or position+1==len(order):
            torch.save(chunks,out/f'predictions_{position//256:04d}.pt');chunks={}
        if position%32==0 or position+1==len(order):
            save(out/'status.json',dict(status='RUNNING',images=position+1,total=len(order),updates=updates,rollbacks=rollbacks,empty=empty,last=last,elapsed_seconds=time.monotonic()-started,peak_mib=torch.cuda.max_memory_allocated()/2**20,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        if (position+1)%1024==0 or position+1==len(order):
            ck=dict(student={k:p.detach().cpu() for k,p in selected.items()},teacher={k:p.detach().cpu() for k,p in zip(selected,teacher_params)},optimizer=optimizer.state_dict(),position=position,source_sha256=identity['sha256'],spec=spec)
            torch.save(ck,out/'adaptation_last.tmp');(out/'adaptation_last.tmp').replace(out/'adaptation_last.pth')
            save(out/'trace_partial.json',history)
    stress=None
    if probe and spec['arm']!='frozen':
        # Exercise the worst native pseudo-target count, independent of confidence.
        # This extra backward pass does not step the optimizer or read any labels.
        assert len(pred['scores'])==300
        view,boxes,_=ops.strong_view(image,pred['boxes'],name,spec['order_seed'])
        target=dict(boxes=torch.from_numpy(ops.normalized_targets(boxes)).cuda(),labels=torch.from_numpy(pred['labels'].astype(np.int64)).cuda())
        optimizer.zero_grad(set_to_none=True);output=student(to_tensor(view));loss,parts=loss_final(criterion,output,target,spec['loss_mode'])
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,.1);assert torch.isfinite(norm)
        stress=dict(pseudo_count=300,loss=float(loss.detach()),gradient_norm=float(norm),optimizer_step=False)
        optimizer.zero_grad(set_to_none=True);del output,loss,parts,target
    save(out/'trace.json',history)
    for k,v in student.state_dict().items():
        if k not in selected:assert torch.equal(v.detach().cpu(),initial[k]),'Frozen state changed: '+k
    for k,v in teacher.state_dict().items():
        if k not in selected:assert torch.equal(v.detach().cpu(),initial[k]),'Teacher frozen state changed: '+k
    if probe and spec['arm']!='frozen':assert updates>0 and norms_with_grad,'Probe must demonstrate a real gradient update'
    assert sha(C/identity['weight'])==identity['sha256'],'Source file mutated'
    save(out/'status.json',dict(status='PROBE_COMPLETE' if probe else 'PREDICTIONS_COMPLETE',images=len(order),updates=updates,rollbacks=rollbacks,empty=empty,source_only=source_only,frozen_state_unchanged=True,parameters_with_nonzero_gradient=sorted(norms_with_grad) if probe else None,peak_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_seconds=time.monotonic()-started,ground_truth_read=False,full_stream=not probe,loss_mode=spec['loss_mode'],adaptation_is_not_training_free=spec['arm']!='frozen',dense_probe=stress,final_student_drift=ops.drift(params,source),final_teacher_drift=ops.drift(teacher_params,source)))
