import math
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from core.utils.skeleton_utils import (
    Skeleton,
    SMPLSkeleton,
    calculate_kinematic, )


class PoseOptPSO(nn.Module):
    """
    Pose optimization module
    """

    def __init__(
            self,
            rest_pose: torch.Tensor,
            kp3d: np.ndarray,
            bones: np.ndarray,  # FIXME: shouldn't call it bones anymore
            skel_type: Skeleton,
            emb_dim: int = 16,  # unique-identifier for the per-pose code
            depth: int = 4,
            width: int = 128,
            residual_scale: float = 0.1,  # scale the output
            n_particles: int = 10,
            rot_6d: bool = False,
            n_embs: Optional[int] = None,
            inertia_constant: float = 0.7,
            exploration_constant: float = 1.2,
            exploitation_constant: float = 0.8,
            **kwargs,
    ):
        N_joints = kp3d.shape[1]

        super().__init__()
        assert skel_type == SMPLSkeleton, 'Only SMPLSkeleton is supported for now'
        assert depth >= 2, 'Depth must be at least 2'
        self.skel_type = skel_type
        self.n_embs = n_embs if n_embs is not None else len(kp3d)
        self.residual_scale = residual_scale
        # self._base_model = base_model
        self._n_frames = bones.shape[0]
        self._device = rest_pose.device

        self._inertia = inertia_constant
        self._exploration = exploration_constant
        self._exploitation = exploitation_constant
        self._particles, self._vels, self._p_best, self._g_best = self._init_pso(bones.shape[0], n_particles, N_joints)
        self.current_joint_norms, self.current_joint_vars = None, None

        self._p_best_scores = torch.zeros(self._p_best.shape[:-2])
        self._g_best_scores = torch.zeros(self._g_best.shape[:-2])
        self._p_best_scores[:] = np.inf
        self._g_best_scores[:] = np.inf

        rvecs = torch.tensor(bones)
        # NOTE: this is different from original A-NeRF implementation, but should 
        # work better
        pelvis = torch.tensor(kp3d[:, skel_type.root_id])

        self.register_buffer('rest_pose', rest_pose, persistent=False)
        self.register_buffer('pelvis', pelvis, persistent=False)
        self.register_buffer('rvecs', rvecs, persistent=False)
        self.pose_embs = nn.Embedding(self.n_embs, emb_dim)

        # initialize refinement network
        rvec_dim = 3 if not rot_6d else 6
        rvec_dim = rvec_dim * N_joints
        net = [nn.Linear(rvec_dim + emb_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            net.extend([nn.Linear(width, width), nn.ReLU()])
        net.append(nn.Linear(width, 2 * (rvec_dim + 3)))
        self.refine_net = nn.Sequential(*net)
        # self._base_model.register_forward_hook()

    @property
    def n_particles(self):
        return self._particles.shape[1]

    def _init_pso(self, n_frames, n_particles, n_joints):
        upper_bound = math.pi / 10
        lower_bound = -upper_bound

        # We store the current position & velocity of each particle.
        # n_joints + 1 in order to store root translation as well as joint rotations.
        pos = torch.FloatTensor(n_frames, n_particles, n_joints + 1, 3).uniform_(lower_bound, upper_bound)
        vel = torch.zeros(n_frames, n_particles, n_joints + 1, 3).uniform_(lower_bound, upper_bound)
        # As well as the best pose encountered by each particle
        p_best = pos.detach().clone()
        # And the best particle globally
        g_best = p_best[:, 0]

        return (x.to(self._device) for x in [pos, vel, p_best, g_best])

    def on_loss_evaluated(self, network_inputs: Dict, loss: torch.Tensor):
        N_samples = len(network_inputs["kp3d"])
        skip = N_samples // network_inputs.get("N_unique", 1)
        kp_unique = network_inputs["real_kp_idx"][::skip]
        particle_idx = network_inputs["particle_idx"]

        x_i = self._particles[kp_unique, particle_idx]
        v_i = self._vels[kp_unique, particle_idx]
        p_best = self._p_best[kp_unique, particle_idx]

        did_improve = loss < self._p_best_scores[kp_unique, particle_idx]
        kp_p_improved = kp_unique[did_improve]
        particle_improved = particle_idx[did_improve]
        self._p_best[kp_p_improved, particle_improved] = self._particles[kp_p_improved, particle_improved]
        self._p_best_scores[kp_p_improved, particle_improved] = loss

        g_improved = loss < self._g_best_scores[kp_unique]
        kp_g_improved_idx = kp_unique[g_improved]
        # We only modify the frame with the highest loss
        g_mod_idx = torch.argmax(self._g_best_scores[kp_g_improved_idx])
        g_mod_particle = self._particles[kp_g_improved_idx[g_mod_idx], particle_idx[g_mod_idx]]

        self._g_best[g_mod_idx] = g_mod_particle
        self._g_best_scores[g_mod_idx] = loss

        r1, r2 = torch.rand(2)
        explore = self._exploration * r1 * (p_best - x_i)
        exploit = self._exploitation * r2 * (self._g_best[kp_unique] - x_i)

        self._vels[kp_unique, particle_idx] = self._inertia * v_i + explore + exploit
        self._particles[kp_unique, particle_idx] += self._vels[kp_unique, particle_idx]

    def forward(
            self,
            network_inputs: Dict[str, Any],
            kp3d: torch.Tensor,
            bones: torch.Tensor,  # FIXME: rvec
            kp_idxs: torch.LongTensor,
            N_unique: int = 1,
            unroll: bool = True
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        kp3d: torch.Tensor (B, J, 3) -- joint locations
        bones: torch.Tensor (B, J, 3 or 6) -- rotation vector for each joint/bone
        kp_idxs: torch.LongTensor -- index of the poses
        unroll: bool -- unroll the for loop for kinematic calculation, this gives faster forward pass
        """

        N_samples = len(kp3d)
        skip = N_samples // N_unique
        N, J, _ = bones.shape
        # The body poses are organized in (N_images * N_rays_per_image, ...)
        # meaning that there will be N_rays_per_image redundant poses (the same image has the same pose)
        # -> skip the redundant one
        rvecs = rearrange(bones[::skip], 'b j d -> b (j d)')
        kp_unique = kp_idxs[::skip]
        kp3d_unique = kp3d[::skip]

        pelvis = self.pelvis[kp_unique].clone()
        assert torch.allclose(pelvis, kp3d_unique[:, self.skel_type.root_id],
                              atol=1e-5), 'Pelvis should be the same as root joint'

        particle_idx = torch.zeros(len(kp_unique)).long()
        particle_idx[:] = torch.randint(low=0, high=self.n_particles, size=(1,))

        residuals = self._particles[kp_unique, particle_idx]
        residual_rvecs, residual_pelvis = residuals[:, :-1], residuals[:, -1]
        rvecs = rearrange(rvecs, 'b (j d) -> b j d', j=J) + residual_rvecs
        pelvis = pelvis + residual_pelvis
        rest_pose = self.rest_pose.clone()

        kp, skts = calculate_kinematic(
            rest_pose[None],
            rvecs,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        # to update
        # 1. kp3d, 2. skts, 3. bone_poses
        # -> expand the shape
        kp = rearrange(kp[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        skts = rearrange(skts[:, None].expand(-1, skip, -1, -1, -1), 'i s j k d -> (i s) j k d')
        rvecs = rearrange(rvecs[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        network_inputs.update(
            kp3d=kp,
            skts=skts,
            bones=rvecs,  # FIXIT: nameing
            particle_idx=particle_idx
        )
        return network_inputs

    @torch.no_grad()
    def export_optimized_pose(self, unroll: bool = True):
        pelvis = self.pelvis.clone()
        rvecs = self.rvecs.clone()
        B, J, _ = rvecs.shape
        rest_pose = self.rest_pose.clone()

        embs = self.pose_embs(torch.arange(len(rvecs)))

        x = torch.cat([rvecs, embs], dim=-1)
        residual = self.refine_net(x) * self.residual_scale

        residual_rvecs, residual_pelvis = torch.split(residual, [rvecs.shape[-1], 3], dim=-1)
        rvecs = rearrange(rvecs + residual_rvecs, 'b (j d) -> b j d', j=J)
        pelvis = pelvis + residual_pelvis

        kp, skts = calculate_kinematic(
            rest_pose[None],
            rvecs,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        return {
            'kp3d': kp,
            'skts': skts,
            'bones': rvecs,
        }
