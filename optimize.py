"""Five-view prediction and complete DIOR-R evaluation."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import openrio
from optimizations.multiview import variants, views_for

METHODS = ['single_view', 'five_view_nms', 'fixed_box_rank', 'agreement', 'uniform', 'weighted']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['predict', 'evaluate'])
    p.add_argument('--model', choices=sorted(json.loads((openrio.ROOT/'models.json').read_text())), default='n_rotation_multiview')
    p.add_argument('--method', choices=METHODS+['all'], default='uniform')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--memory-fraction', type=float, default=.12)
    p.add_argument('--limit', type=int, default=0, help='Diagnostic only; suppress metrics')
    p.add_argument('--save-views', action='store_true')
    a = p.parse_args()
    if not 0 < a.memory_fraction <= 1 or a.limit < 0:
        p.error('Invalid memory fraction or image limit')
    if a.command == 'predict' and a.image is None:
        p.error('--image is required')
    if a.command == 'evaluate' and a.data_root is None:
        p.error('--data-root is required')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    cfg, model, post, identity = openrio.load_model(a.model, device, a.data_root)
    names = METHODS if a.method == 'all' else [a.method]
    started = time.monotonic()
    metadata = dict(model=a.model, identity=identity, methods=names, view_count=5,
                    geometry='continuous 800-pixel edges; radian angles modulo pi',
                    no_parameter_updates=True, no_ground_truth_in_inference=True)
    if a.command == 'predict':
        im = Image.open(a.image).convert('RGB')
        if im.size != (800, 800):
            raise ValueError('Validated optimization input must be exactly 800x800')
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        views = views_for(model, post, x)
        parts = variants(views)
        for name in names:
            (a.output/(name+'.json')).write_text(json.dumps({k: v.tolist() for k, v in parts[name].items()}))
        metadata['images'] = 1
    else:
        cfg.yaml_cfg['val_dataloader'].update(total_batch_size=1, num_workers=a.workers)
        dl = cfg.val_dataloader
        if len(dl.dataset) != 11738:
            raise ValueError('Published evaluation expects the complete 11738-image split')
        evaluators = {}
        for name in names:
            ev = copy.copy(cfg.evaluator)
            ev.cleanup()
            evaluators[name] = ev
        seen, cache = set(), []
        for images, targets in dl:
            if targets[0]['orig_size'].tolist() != [800, 800]:
                raise ValueError('Expected native 800x800 DIOR-R tiles')
            index = int(targets[0]['image_id'])
            assert index not in seen
            views = views_for(model, post, images.to(device))
            parts = variants(views)
            for name, ev in evaluators.items():
                ev.update({index: {k: torch.as_tensor(v) for k, v in parts[name].items()}})
            seen.add(index)
            if a.save_views:
                cache.append(dict(index=index, views=views))
                if len(cache) == 256:
                    torch.save(cache, a.output/f'views_{(len(seen)-1)//256:04d}.pt')
                    cache = []
            if len(seen) % 32 == 0:
                (a.output/'progress.json').write_text(json.dumps(dict(images=len(seen), total=11738)))
            if a.limit and len(seen) >= a.limit:
                break
        if cache:
            torch.save(cache, a.output/f'views_{(len(seen)-1)//256:04d}.pt')
        metadata.update(images=len(seen), complete=not a.limit and len(seen) == 11738)
        metrics = {}
        for name, ev in evaluators.items():
            target = a.output/name
            target.mkdir()
            torch.save(ev.predictions, target/'predictions.pt')
            if not a.limit:
                assert seen == set(range(11738))
                ev.img_ids = sorted(ev.predictions)
                ev.iou_thrs = [.5, .75]
                ev.summarize()
                metrics[name] = {k: 100*float(ev.stats[k]) for k in ['AP50', 'AP75']}
                (target/'metrics.json').write_text(json.dumps(metrics[name], indent=2))
        if metrics:
            (a.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    assert openrio.digest(openrio.ROOT/identity['weight']) == identity['sha256']
    metadata['elapsed_seconds'] = time.monotonic()-started
    (a.output/'run.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
