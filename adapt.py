"""Causal test-time parameter adaptation under an explicit image-only shift.

Reported gains concern the complete online prediction stream. The final
parameter state does not inherit the stream's AP as a static model score.
"""
import argparse
import json
from pathlib import Path
from optimizations.adaptation import run

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', choices=['n_noise', 's_noise'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--frozen', action='store_true', help='Required same-stream source control')
    p.add_argument('--limit', type=int, default=0, help='Diagnostic only; cannot be scored as a full stream')
    p.add_argument('--memory-fraction', type=float)
    p.add_argument('--loss-mode', choices=['full', 'classification'])
    a = p.parse_args()
    if a.limit < 0 or not a.device.startswith('cuda'):
        p.error('This validated adaptation implementation requires CUDA and a nonnegative limit')
    spec = json.loads((ROOT/'recipes'/(a.recipe+'.json')).read_text())
    spec.update(data_root=str(a.data_root.resolve()), output=str(a.output.resolve()), device=a.device)
    if a.frozen:
        spec['arm'] = 'frozen'
    if a.memory_fraction is not None:
        if not 0 < a.memory_fraction <= 1:
            p.error('Memory fraction must be in (0,1]')
        spec['allocator_fraction'] = a.memory_fraction
    if a.loss_mode:
        spec['loss_mode'] = a.loss_mode
    run(spec, a.limit)
    (a.output/'spec.json').write_text(json.dumps(spec, indent=2)+'\n')


if __name__ == '__main__':
    main()
