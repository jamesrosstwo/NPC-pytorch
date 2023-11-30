import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from torch import nn

class GradientCapture:
    _FILENAME = "gradcap.pickle"

    @classmethod
    def from_path(cls, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)

        pkl_path = path / GradientCapture._FILENAME
        if pkl_path.exists() and pkl_path.is_file():
            with open(str(pkl_path), "rb") as file:
                return pickle.load(file)
        return cls(path)

    def __init__(self, save_loc: Path):
        self._out_loc: Path = save_loc
        self._param_grads: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._pred_grads: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._preds: List[np.ndarray] = []
        self.register_pred_hooks()
        self.register_param_hooks()

    def register_pred_hooks(self, module: nn.Module):
        nm, last_layer = module.named_children()[-1]

        def _hook_backward_fn(m, grad_in, grad_out):
            self._pred_grads[nm].append(grad_out.detach().cpu().numpy())

        def _hook_fn(m, i, output):
            self._preds.append(output.detach().cpu().numpy())

        last_layer.register_backward_hook(_hook_backward_fn)
        module.register_forward_hook(_hook_fn)

    def register_param_hooks(self, module):
        def _hook_fn(module, grad_in, grad_out):
            self._param_grads.append(grad_out.detach().cpu().numpy())

        module.register_backward_hook(_hook_fn)

    def generate_hook(self, image_idx: int):
        return lambda grad: self.hook(image_idx, grad)

    def save_results(self):
        with open(str(self._out_loc / GradientCapture._FILENAME), "wb") as file:
            pickle.dump(self, file)
