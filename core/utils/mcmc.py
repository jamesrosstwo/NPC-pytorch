from abc import ABC, abstractmethod
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data._utils.worker import WorkerInfo


class MCMCRaySampler(ABC):
    def __init__(self, n_concurrent: int, bounds: np.ndarray, sampling_masks, n_frames: int = 1,
                 dilate_mask: bool = False):
        self._HW = bounds
        self._dilate_mask = dilate_mask
        self._n_frames = n_frames

        # Number of rays dictates our chain length
        self._n_concurrent = n_concurrent

        # TODO: check if sampling masks need adjustment
        # assume sampling masks are of shape (N, H, W, 1)
        # time0 = time.time()

        self._valid_idx = []
        self._sampling_masks = []
        for mask in sampling_masks:
            sampling_mask = mask.reshape(-1)
            if self._dilate_mask:
                border = 3
                kernel = np.ones((border, border), np.uint8)
                sampling_mask = cv2.dilate(sampling_mask.reshape(*self._HW), kernel=kernel, iterations=1).reshape(-1)
            # print(f'fetch mask time {time.time()-time0}')
            self._sampling_masks.append(sampling_mask)

            valid_idxs, = np.where(sampling_mask > 0)
            if len(valid_idxs) == 0 or len(valid_idxs) < self.n_concurrent:
                valid_idxs = np.arange(len(sampling_mask))
            self._valid_idx.append(valid_idxs)

    @property
    def n_concurrent(self):
        return self._n_concurrent

    @property
    def n_frames(self):
        return self._n_frames

    def get_sampling_mask(self, idx):
        return self._sampling_masks[idx]

    @abstractmethod
    def add_sample(self, network_inputs: Dict, loss: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, valid_idxs) -> np.ndarray:
        raise NotImplementedError()


class MetropolisRaySampler(MCMCRaySampler):
    def __init__(self, max_iters=100, perturb_strength=15, n_roll=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        worker_info: WorkerInfo = torch.utils.data.get_worker_info()
        self._n_workers = worker_info.num_workers

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

    def _idx_to_coords(self, idx: np.ndarray):
        H, W = self._HW
        row = idx // W
        col = idx % W
        return np.column_stack((row, col))

    def _coords_to_idx(self, coords: np.ndarray):
        H, W = self._HW
        return (coords[:, 0] * W + coords[:, 1]).astype(np.int32)

    def _base_case(self, idx) -> np.ndarray:
        valid_idxs = self._valid_idx[idx]
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
        prop[-self._n_roll:] += perturbations
        return prop

    def _sample(self, idx) -> np.ndarray:
        worker_id = torch.utils.data.get_worker_info().id
        if not self._initialized[worker_id, idx]:
            self._initialized[worker_id, idx] = True
            self._current_samples[worker_id, idx] = self._base_case(idx)
            return self._current_samples[worker_id, idx]

        p = self._generate_proposal(worker_id, idx)
        self._proposal[worker_id, idx] = p
        return p

    def sample(self, idx) -> np.ndarray:
        return np.sort(self._coords_to_idx(self._sample(idx)))
    def add_sample(self, network_inputs: Dict, loss: torch.Tensor):
        alpha = self._last_loss / loss
        r = np.random.rand()
        if r > alpha:
            self._current_samples = self._proposal
            self._last_loss = loss
        # Accept the sample
        self._last_loss = loss
