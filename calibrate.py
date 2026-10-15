"""Reproduce the released Nano two-buffer calibration using training images only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import openrio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.12)
    p.add_argument('--verify-reference', action='store_true')
    a = p.parse_args()
    if not 0 < a.memory_fraction <= 1:
        p.error('Invalid memory fraction')
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    _, model, _, identity = openrio.load_model('n_calibration_parent', device, a.data_root)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    names = (openrio.ROOT/'recipes/n_calibration_images.txt').read_text().split()
    train = set()
    for split in ['train', 'val']:
        train.update((a.data_root/f'ImageSets/Main/{split}.txt').read_text().split())
    test = set((a.data_root/'ImageSets/Main/test.txt').read_text().split())
    assert len(names) == len(set(names)) == 256 and set(names) <= train and not set(names) & test
    modules = dict(model.named_modules())
    layers = [f'backbone.stages.{j}.blocks.0.aggregation.1.bn' for j in (2, 3)]
    for name in layers:
        bn = modules[name]
        acc = {'n': 0}

        def observe(module, inputs):
            x = inputs[0].detach().double()
            acc['sum'] = acc.get('sum', 0)+x.sum((0, 2, 3))
            acc['sq'] = acc.get('sq', 0)+x.square().sum((0, 2, 3))
            acc['n'] += x.numel()//x.shape[1]

        hook = bn.register_forward_pre_hook(observe)
        try:
            with torch.no_grad():
                for start in range(0, len(names), 8):
                    images = []
                    for stem in names[start:start+8]:
                        with Image.open(a.data_root/'JPEGImages-trainval'/(stem+'.jpg')) as im:
                            im = im.convert('RGB')
                            assert im.size == (800, 800)
                            images.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255))
                    model(torch.stack(images).to(device))
        finally:
            hook.remove()
        mean = acc['sum']/acc['n']
        var = (acc['sq']/acc['n']-mean.square()).clamp_min(0)
        bn.running_mean.copy_(mean)
        bn.running_var.copy_(var)
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    allowed = {name+'.'+key for name in layers for key in ['running_mean', 'running_var']}
    assert set(changed) <= allowed and changed
    receipt = dict(parent=identity, training_images=names, annotations_read=False,
                   changed_state_keys=changed, learned_parameters_unchanged=True,
                   reference_check='not requested')
    if a.verify_reference:
        entry = json.loads((openrio.ROOT/'models.json').read_text())['n_calibrated']
        reference = torch.load(openrio.ROOT/entry['weight'], map_location='cpu', weights_only=True)['model']
        errors = {k: float((after[k].double()-reference[k].double()).abs().max()) for k in after}
        receipt.update(reference_max_abs=max(errors.values()), reference_check='bitwise' if all(torch.equal(after[k], reference[k]) for k in after) else 'tolerance')
        assert all(torch.allclose(after[k], reference[k], atol=2e-6, rtol=1e-5) for k in after)
    torch.save({'model': after}, a.output/'model.pth')
    (a.output/'calibration.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['parent', 'training_images']}))


if __name__ == '__main__':
    main()
