"""Offline scoring of a completed adaptation stream, in a separate CPU process."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
import runtime
sys.path.insert(0, str(Path(__file__).resolve().parent/'tools'))
from tools.independent_eval import annotations
from engine.data.dataset.mmeval import eval_rbbox_map


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    if (a.run/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists')
    status = json.loads((a.run/'status.json').read_text())
    manifest = json.loads((a.run/'manifest.json').read_text())
    assert status['status'] == 'PREDICTIONS_COMPLETE' and status['images'] == 11738
    assert status['frozen_state_unchanged'] and not status['ground_truth_read']
    names = (a.data_root/'ImageSets/Main/test.txt').read_text().split()
    assert manifest['images'] == names
    assert hashlib.sha256((a.data_root/'ImageSets/Main/test.txt').read_bytes()).hexdigest() == manifest['test_list_sha256']
    preds = {}
    for path in sorted(a.run.glob('predictions_*.pt')):
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        assert not set(preds).intersection(chunk)
        preds.update(chunk)
    assert set(preds) == set(range(11738))
    gt, identity = annotations(a.data_root)
    dets = []
    for i in range(11738):
        row = preds[i]
        assert manifest['order'][row['position']] == i
        values = np.concatenate([row['bboxes'], row['scores'][:, None]], axis=1)
        assert np.isfinite(values).all()
        dets.append([values[row['labels'] == c] for c in range(20)])
    targets = [dict(bboxes=g['boxes'], labels=g['labels']) for g in gt]
    scores = {}
    for key, threshold in [('ap50', .5), ('ap75', .75)]:
        mean, classes = eval_rbbox_map(dets, targets, iou_thr=threshold, use_07_metric=True, nproc=1)
        scores[key] = float(mean)*100
    torch.save(preds, a.run/'predictions.pt')
    (a.run/'metrics.json').write_text(json.dumps({k.upper(): v for k, v in scores.items()}, indent=2))
    result = dict(status='COMPLETE', images=11738, results={'full': scores},
                  identity=identity, cuda_used=torch.cuda.is_initialized(),
                  interpretation='online emit-before-update AP; not final-state static AP')
    assert not result['cuda_used']
    (a.run/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
