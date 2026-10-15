"""Paired byte-to-owned-CPU detection latency, with eager/graph output checks."""
import argparse
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import openrio
from optimizations.cuda_graph import StaticForward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='n')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--memory-fraction', type=float, default=.12)
    p.add_argument('--images', type=int, default=32)
    p.add_argument('--repetitions', type=int, default=3)
    a = p.parse_args()
    if not 0 < a.memory_fraction <= 1 or min(a.images, a.repetitions) < 1:
        p.error('Invalid memory or timing counts')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    if device.type != 'cuda':
        p.error('CUDA Graph requires CUDA')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _, model, post, identity = openrio.load_model(a.model, device, a.data_root)
    for module in model.modules():
        for key, value in list(vars(module).items()):
            if torch.is_tensor(value) and value.device.type == 'cpu' and (key.startswith('pos_embed') or key == 'num_points_scale'):
                setattr(module, key, value.to(device))
    names = (a.data_root/'ImageSets/Main/test.txt').read_text().split()
    ids = np.random.default_rng(13).choice(len(names), a.images, replace=False)
    payloads = [(a.data_root/'JPEGImages-test'/(names[i]+'.jpg')).read_bytes() for i in ids]

    def tensor(payload):
        with Image.open(io.BytesIO(payload)) as im:
            im = im.convert('RGB')
            assert im.size == (800, 800)
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    def pipeline(forward, payload):
        result = post(forward(tensor(payload)), torch.tensor([[800, 800]], device=device))
        return [{k: v.detach().cpu().clone() for k, v in row.items()} for row in result]

    parity = dict(images=0, bitwise_images=0, max_box_abs=0., max_logit_abs=0.)
    samples = {'eager': [], 'graph': []}
    with torch.inference_mode():
        start = time.monotonic()
        graph = StaticForward(model, device)
        capture = time.monotonic()-start
        for payload in payloads:
            x = tensor(payload)
            expected, actual = model(x), graph(x)
            exact = True
            for key, field in [('pred_boxes', 'max_box_abs'), ('pred_logits', 'max_logit_abs')]:
                parity[field] = max(parity[field], float((expected[key]-actual[key]).abs().max()))
                assert torch.allclose(expected[key], actual[key], atol=1e-6, rtol=1e-5)
                exact &= torch.equal(expected[key], actual[key])
            parity['images'] += 1
            parity['bitwise_images'] += int(exact)
        for fn in [model, graph]:
            for _ in range(20):
                pipeline(fn, payloads[0])
        rng = np.random.default_rng(13)
        for _ in range(a.repetitions):
            for payload in payloads:
                for name in rng.permutation(['eager', 'graph']):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    pipeline(model if name == 'eager' else graph, payload)
                    torch.cuda.synchronize(device)
                    samples[name].append(1000*(time.perf_counter()-start))
    result = dict(identity=identity, parity=parity, capture_seconds=capture,
                  scope='compressed bytes resident in host memory, PIL decode, RGB conversion, tensor preprocessing, H2D, model, native top300 postprocess, owned CPU detections; excludes disk/network/serialization',
                  samples_ms=samples, summary={k: {'p50_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95))} for k, v in samples.items()},
                  device=torch.cuda.get_device_name(device), torch=torch.__version__,
                  shared_device='not assumed isolated; inspect local workload when interpreting speed')
    (a.output/'timing.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
