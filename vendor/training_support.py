"""Isolated reproducibility support; same optimizer/loss math as carrier."""
import os
import random

import numpy as np
import torch

from engine.core import register
from engine.data.dataloader import DataLoader
from engine.solver.det_solver import DetSolver


@register()
class PaperDataLoader(DataLoader):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        seed = int(os.environ.get('PAPER_DATA_SEED', '0')) + 10000 * int(epoch)
        # Two independent CPU streams; model shape cannot shift sample order.
        self.generator = torch.Generator().manual_seed(seed + 101)
        if hasattr(self.sampler, 'generator'):
            self.sampler.generator = torch.Generator().manual_seed(seed + 102)
        # Denoising/dropout stream is reset at epoch boundaries. Data workers
        # obtain torch/Python/numpy seeds from DataLoader's worker generator.
        random.seed(seed + 103); np.random.seed((seed + 103) % 2**32)
        torch.manual_seed(seed + 104)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 104)


class PaperDetSolver(DetSolver):
    def load_resume_state(self, path):
        # At the no-augmentation transition, upstream refreshes model/optimizer
        # from best_stg1 but also overwrites its completed-epoch counter. Preserve
        # that counter only for this internal refresh, not an explicit resume.
        internal = (str(path) == str(self.output_dir / 'best_stg1.pth')
                    and str(self.cfg.resume) != str(path))
        if internal and os.environ.get('EMA_RESTART_BEST'):
            path = os.environ['EMA_RESTART_BEST']
        if internal and os.environ.get('PROBE_NO_ROLLBACK') == '1':
            # Preregistered probe amendment v2: during the clean-data extension,
            # skip the upstream stumble->best_stg1 rollback and its EMA-decay
            # decrement so the trajectory measures pure adaptation capacity.
            print('[probe-amendment-v2] skip best_stg1 rollback; weights kept')
            return
        previous = self.last_epoch
        super().load_resume_state(path)
        if internal:
            self.last_epoch = previous

    def state_dict(self):
        state = super().state_dict()
        state['paper_reconstruction_rng'] = {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'data_seed': int(os.environ.get('PAPER_DATA_SEED', '0')),
            'resume_contract': 'epoch boundary; next epoch independently seeds data and denoising; no mid-epoch resume claim',
        }
        return state
