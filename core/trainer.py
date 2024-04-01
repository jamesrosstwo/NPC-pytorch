import torch
import torch.nn as nn
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig

from core.gradient_capture import GradientCapture
from core.losses import *


def img2mse(img, target):
    return (img - target).pow(2.).mean()


def mse2psnr(mse):
    return -10. * torch.log10(mse) / torch.log10(torch.Tensor([10.]))


def img2psnr(img, target):
    return mse2psnr(img2mse(img, target))


def to_device(data, device='cuda'):
    data_device = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            data_device[k] = v.to(device)
        else:
            data_device[k] = v
    return data_device


def get_lr_decay_fn(decay_type):
    if decay_type == 'standard':
        return decay_optimizer_lr
    elif decay_type == 'tava':
        return decay_optimizer_lr_delay
    elif decay_type == 'standard_delay':
        return decay_optimizer_lr_decay_delay
    else:
        raise ValueError(f'Unknown decay_type {decay_type}')


def decay_optimizer_lr(
        init_lr,
        decay_steps,
        decay_rate,
        optimizer,
        global_step=None,
        group_scale=[1.0, 1.0],
        **kwargs,
):
    optim_step = global_step

    new_lrate = init_lr * (decay_rate ** (optim_step / decay_steps))
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None


def decay_optimizer_lr_decay_delay(
        init_lr,
        decay_steps,
        decay_rate,
        optimizer,
        global_step=None,
        delay_steps=0,
        group_scale=[1.0, 1.0],
        **kwargs,
):
    optim_step = max(global_step - delay_steps, 0)
    new_lrate = init_lr * (decay_rate ** (optim_step / decay_steps))
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None


def decay_optimizer_lr_delay(
        init_lr,
        decay_steps,
        decay_rate,
        optimizer,
        delay_steps=0,
        delay_mult=0.01,
        global_step=None,
        group_scale=[1.0, 0.1],
        **kwargs,
):
    """ From TAVA
    """
    assert global_step is not None
    optim_step = global_step
    if delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = delay_mult + (1 - delay_mult) * np.sin(
            0.5 * np.pi * np.clip(optim_step / delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(optim_step / decay_steps, 0, 1)
    log_lerp = np.exp(np.log(init_lr) * (1 - t) + np.log(init_lr * decay_rate) * t)
    new_lrate = delay_rate * log_lerp
    for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lrate * group_scale[j]
    return new_lrate, None


@torch.no_grad()
def get_gradnorm(module):
    total_norm = 0.0
    cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        cnt += 1
    avg_norm = (total_norm / cnt) ** 0.5
    total_norm = total_norm ** 0.5
    return total_norm, avg_norm


class Trainer(object):
    """ For training models
    """

    def __init__(
            self,
            config,
            loss_config,
            full_config,
            model,
            ckpt=None,
            gradient_capture: Optional[DictConfig] = None,
            **kwargs,
    ):
        self.config = config
        self.loss_config = loss_config
        self.full_config = full_config
        self.model = nn.DataParallel(model)
        self.gradient_capture = None
        if gradient_capture is not None:
            self.gradient_capture: GradientCapture = instantiate(gradient_capture)
            self.gradient_capture.register_pred_hooks(self.model)
            self.gradient_capture.register_param_hooks(self.model)

        # initialize optimizerni
        self.init_optimizer(ckpt)

        # initialize loss function
        self.init_loss_fns()

    def init_optimizer(self, ckpt=None):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, 'opt_param_groups'):
            model_params = model.opt_param_groups()
            self.optimizer = instantiate(self.config.optim, params=model_params[0]['params'])
            if len(model_params) == 2:
                self.optimizer.add_param_group(model_params[1])
            elif len(model_params) > 2:
                raise ValueError('Only support 2 param groups')
        else:
            model_params = model.parameters()
            self.optimizer = instantiate(self.config.optim, params=model_params)

        if ckpt is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])

        self.optimizer.zero_grad()
        self.decay_fn = get_lr_decay_fn(self.config.lr_sched.decay_type)

    def update_optimizer(self, global_iter):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        self.optimizer = instantiate(self.config.optim, params=model.parameters())
        # update LR
        self.optimizer.zero_grad()
        new_lr, _ = self.decay_fn(
            init_lr=self.config.optim.lr,
            optimizer=self.optimizer,
            global_step=global_iter,
            **self.config.lr_sched,
        )

    def init_loss_fns(self):
        self.loss_fns = [
            eval(k)(**v)
            for k, v in self.loss_config.items()]

    def train_batch(self, batch, global_iter=1):
        device_cnt = 1
        if isinstance(self.model, nn.DataParallel):
            if len(self.model.device_ids) > 1:
                device_cnt = len(self.model.device_ids)

        stats = dict()

        # Step 1. model prediction
        batch = to_device(batch, 'cuda')
        batch['N_unique'] = self.full_config.N_sample_images // device_cnt
        batch['device_cnt'] = device_cnt
        batch['global_iter'] = global_iter
        pose_opt = self.config.get('pose_opt', False)
        preds = self.model(batch, pose_opt=pose_opt)

        if pose_opt:
            stats["joint_norms"] = self.model.module.pose_opt.current_joint_norms
            stats["joint_vars"] = self.model.module.pose_opt.current_joint_vars

        # Step 2. compute loss
        # TODO: used to have pose-optimization here ..
        loss, new_stats = self.compute_loss(batch, preds, global_iter=global_iter)
        stats.update(new_stats)

        # clean up after step
        loss.backward()
        self.optimizer.step()
        total_norm, avg_norm = get_gradnorm(self.model)
        self.optimizer.zero_grad()

        # Step 3. post-update stuff

        # change/renew optimizer if needed

        # change learning rate
        new_lr, _ = self.decay_fn(
            init_lr=self.config.optim.lr,
            optimizer=self.optimizer,
            global_step=global_iter,
            **self.config.lr_sched,
        )
        stats.update(lr=new_lr)
        stats.update(avg_norm=avg_norm)

        # TODO: A-NeRF cutoff update
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, 'update_config'):
            need_optim_update = model.update_network(global_iter=global_iter)
            if isinstance(self.model, nn.DataParallel):
                self.model = nn.DataParallel(model)
            if need_optim_update:
                self.update_optimizer(global_iter=global_iter)
            self.model.train()  # somehow training flag is changed..

        return stats

    def compute_loss(self, batch, preds, global_iter=1):

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        total_loss = torch.tensor(0.0)
        loss_stats = {}
        for loss_fn in self.loss_fns:
            loss, loss_stat = loss_fn(batch, preds, global_iter=global_iter, model=model)
            total_loss += loss
            loss_stats.update(**loss_stat)

        pred_rgb_map = preds['rgb_map']
        pred_rgb_zero = preds['rgb0']
        target_s = batch["target_s"]

        if "is_perturbed" in batch:
            pred_rgb_map = pred_rgb_map[batch["is_perturbed"]]
            pred_rgb_zero = pred_rgb_zero[batch["is_perturbed"]]
            target_s = target_s[batch["is_perturbed"]]

        if "kp3d" in preds:
            # https://github.com/pytorch/pytorch/issues/36748
            unique, inverse = torch.unique(batch["real_kp_idx"], sorted=True, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
            pose_error = torch.norm((preds["kp3d"] - batch["base_kp3d"])[perm], dim=2)
            loss_stats.update(pose_error=pose_error)

        # get extra stats that's irrelevant to loss
        loss_stats.update(psnr=img2psnr(pred_rgb_map, target_s).item())
        if 'rgb0' in preds:
            loss_stats.update(psnr0=img2psnr(pred_rgb_zero, target_s).item())
        loss_stats.update(alpha=preds['acc_map'].mean().item())
        if 'acc_map0' in preds:
            loss_stats.update(alpha0=preds['acc0'].mean().item())
        loss_stats.update(total_loss=total_loss.item())

        return total_loss, loss_stats

    def save_ckpt(self, global_iter, path):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_iter': global_iter,
            'gradient_capture': self.gradient_capture
        }

        torch.save(checkpoint_data, path)
