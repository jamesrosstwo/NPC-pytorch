from abc import ABC, abstractmethod
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data._utils.worker import WorkerInfo


class MCMCRaySampler(ABC):
    def __init__(self, n_concurrent: int, bounds: np.ndarray, n_frames: int = 1):
        self._HW = bounds
        self._n_frames = n_frames

        # Number of rays dictates our chain length
        self._n_concurrent = n_concurrent


    @property
    def n_concurrent(self):
        return self._n_concurrent

    @property
    def n_frames(self):
        return self._n_frames

    @abstractmethod
    def add_sample(self, network_inputs: Dict, loss: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, frame_idx, sampling_mask) -> np.ndarray:
        raise NotImplementedError()


class MetropolisRaySampler(MCMCRaySampler):
    def __init__(self, max_iters=100, perturb_strength=15, n_roll=20, is_active=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_workers = 16

        # Unfortunately having multiple workers makes this whole process pretty messy.
        # We just store a separate set of markov chains for each worker since it's all processed in parallel.
        # We store one chain per worker per frame, and each chain stores the number of samples we need for any given
        # batch.1
        # Due to this representation, we also roll a few of them at a time and generate a bunch of proposals.
        self._current_samples = np.zeros((self._n_workers, self.n_frames, self.n_concurrent, 2))
        self._n_roll = n_roll
        self._prop_shape = (self._n_workers, self.n_frames, self._n_concurrent, 2)
        self._proposal = np.zeros(self._prop_shape)
        self._last_loss = np.inf
        self._max_iters = max_iters
        self._perturb_strength = perturb_strength
        self._worker_samples = np.array_split(np.arange(self.n_frames), self._n_workers)
        self._initialized = np.zeros((self._n_workers, self.n_frames), dtype=bool)
        self._is_active = is_active

    # For warmup iters
    def activate(self):
        self._is_active = True

    def _idx_to_coords(self, idx: np.ndarray):
        H, W = self._HW
        row = idx // W
        col = idx % W
        return np.column_stack((row, col))

    def _coords_to_idx(self, coords: np.ndarray):
        H, W = self._HW
        return (coords[:, 0] * W + coords[:, 1]).astype(np.int32)

    def _base_case(self, idx, valid_idxs) -> np.ndarray:
        selected_idx = np.random.choice(
            valid_idxs,
            self.n_concurrent,
            replace=False
        )

        return self._idx_to_coords(selected_idx)

    def _generate_perturbations(self):
        ps = np.random.normal(scale=self._perturb_strength, size=(self._n_roll, 2))
        return np.round(ps).astype(int)

    def _generate_proposal(self, worker_idx, frame_idx) -> np.ndarray:
        base = self._current_samples[worker_idx, frame_idx]
        perturbations = self._generate_perturbations()
        new_elems = base[-self._n_roll:] + perturbations
        # Roll the last n_roll elements around so they can be replaced.
        # Essentially we are storing a fixed history of the markov chain and bulk-updating it to get more new samples.
        prop = np.roll(base, -self._n_roll, axis=0)
        prop[-self._n_roll:] = new_elems
        return prop

    def _sample(self, frame_idx, valid_idxs) -> np.ndarray:
        worker_id = torch.utils.data.get_worker_info().id
        if not self._is_active or not self._initialized[worker_id, frame_idx]:
            self._initialized[worker_id, frame_idx] = True
            self._current_samples[worker_id, frame_idx] = self._base_case(frame_idx, valid_idxs)
            return self._current_samples[worker_id, frame_idx]

        p = self._generate_proposal(worker_id, frame_idx)
        self._proposal[worker_id, frame_idx] = p
        return p

    def sample(self, idx, valid_idxs) -> np.ndarray:
        out = np.sort(self._coords_to_idx(self._sample(idx, valid_idxs)))

        while np.unique(out).shape[0] < self.n_concurrent:
            out = np.sort(self._coords_to_idx(self._sample(idx, valid_idxs)))

        return out

    def add_sample(self, network_inputs: Dict, loss: torch.Tensor):
        l = loss.detach().cpu().numpy()
        alpha = self._last_loss / l
        skip = network_inputs["kp3d"].shape[0] // network_inputs["N_unique"]
        frame_idxs = network_inputs["real_kp_idx"][::skip].detach().cpu().numpy()
        r = np.random.rand(self.n_concurrent)
        to_adj = np.where(r < alpha)
        self._current_samples[:, frame_idxs][:, :, to_adj] = self._proposal[:, frame_idxs][:, :, to_adj]
        self._last_loss = l
