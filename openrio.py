"""Standalone OpenRiO inference/evaluation entry point; see README.md."""
import argparse,json,hashlib,time,sys
from pathlib import Path
import torch,yaml,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_model(name,device,data_root=None):
    manifest=json.loads((ROOT/'models.json').read_text());entry=manifest[name]
    config=ROOT/entry['config'];weights=ROOT/entry['weight']
    assert digest(config)==entry['config_sha256'],'Configuration checksum mismatch'
    assert digest(weights)==entry['sha256'],'Weight checksum mismatch'
    import runtime
    runtime.register_lineage(entry['lineage'])
    from openrio_config import OpenRiOYAMLConfig
    cfg=OpenRiOYAMLConfig(str(config))
    def subst(v):
        if isinstance(v,dict):return {k:subst(x) for k,x in v.items()}
        if isinstance(v,list):return [subst(x) for x in v]
        if isinstance(v,str):return v.replace('${DATA_ROOT}',str(data_root or '${DATA_ROOT}'))
        return v
    cfg.yaml_cfg=subst(cfg.yaml_cfg)
    model=cfg.model.to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['model'],strict=True)
    return cfg,model,cfg.postprocessor.to(device).eval(),entry

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['predict','evaluate'])
    p.add_argument('--model',choices=sorted(json.loads((ROOT/'models.json').read_text())),default='n');p.add_argument('--device',default='cuda:0')
    p.add_argument('--data-root',type=Path);p.add_argument('--image',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.15);p.add_argument('--limit',type=int,default=0,help='Diagnostic only; nonzero limits suppress AP reporting')
    a=p.parse_args();assert 0<a.memory_fraction<=1
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    cfg,model,post,entry=load_model(a.model,device,a.data_root)
    origins={n:str(Path(sys.modules[n].__file__).resolve()) for n in ['engine','paper_modules','openrio_config','rio_cleanroom_overlay']}
    assert all(Path(p).is_relative_to(ROOT) for p in origins.values()), 'Runtime imported outside candidate'
    metadata=dict(runtime_origins=origins,model=a.model,checkpoint_sha256=entry['sha256'],config_sha256=entry['config_sha256'],torch=torch.__version__,device=str(device),protocol=entry['protocol'],start=time.time())
    if a.command=='predict':
        if a.image is None:p.error('--image is required')
        im=Image.open(a.image).convert('RGB')
        if im.width!=im.height:raise ValueError('Candidate single-image API currently requires square DIOR-style tiles; arbitrary-image letterbox inversion is not validated.')
        x=torch.from_numpy(np.array(im.resize((800,800),Image.Resampling.BILINEAR))).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.inference_mode():pred=post(model(x),torch.tensor([[im.width,im.height]],device=device))
        result={k:v.cpu().tolist() for k,v in pred[0].items()};(a.output/'prediction.json').write_text(json.dumps(result))
        metadata.update(images=1,box_format='cxcywh_angle_radians',image_sha256=digest(a.image))
    else:
        if a.data_root is None:p.error('--data-root is required')
        cfg.yaml_cfg['val_dataloader'].update(total_batch_size=a.batch_size,num_workers=a.workers)
        dl=cfg.val_dataloader;ev=cfg.evaluator;ev.cleanup();seen=set();total=len(dl.dataset)
        assert total==11738,'Published protocol expects exactly 11738 images'
        with torch.inference_mode():
            for images,targets in dl:
                ix=[int(t['image_id']) for t in targets];assert not seen.intersection(ix)
                preds=post(model(images.to(device)),torch.stack([t['orig_size'] for t in targets]).to(device))
                assert all(torch.isfinite(v[k]).all() for v in preds for k in ['boxes','scores'])
                ev.update(dict(zip(ix,preds)));seen.update(ix)
                if len(seen)%128==0:(a.output/'progress.json').write_text(json.dumps(dict(images=len(seen),total=total)))
                if a.limit and len(seen)>=a.limit:break
        torch.save(ev.predictions,a.output/'predictions.pt');metadata.update(images=len(seen),complete=len(seen)==total)
        if not a.limit:
            assert seen==set(range(total));ev.img_ids=sorted(ev.predictions);ev.iou_thrs=[.5,.75];ev.summarize()
            (a.output/'metrics.json').write_text(json.dumps({k:float(ev.stats[k])*100 for k in ['AP50','AP75']},indent=2))
    metadata['elapsed_seconds']=time.time()-metadata['start'];(a.output/'run.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
