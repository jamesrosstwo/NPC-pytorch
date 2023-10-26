import numpy as np
import torch
import torch.nn as nn
from core.utils.skeleton_utils import (
    Skeleton,
    SMPLSkeleton,
    calculate_kinematic,
)
from typing import Dict, Any


class PoseSampler(nn.Module):
    """
    Pose optimization module
    """

    def __init__(
            self,
            rest_pose: torch.Tensor,
            skel_type: Skeleton,
            n_framecodes: int
    ):
        super().__init__()
        assert skel_type == SMPLSkeleton, 'Only SMPLSkeleton is supported for now'
        self.skel_type = skel_type
        self.rest_pose = rest_pose
        self.perturb_strength = 0
        self.n_framecodes = n_framecodes
        # cov_tensor = torch.stack([torch.eye(self.rest_pose.shape[1]) for _ in range(self.rest_pose.shape[0])])
        self.variance_tensor = torch.ones((self.n_framecodes, *self.rest_pose.shape)) * np.pi / 24
        self.means_tensor = torch.zeros((self.n_framecodes, *self.rest_pose.shape))
        self.variances = nn.Parameter(data=self.variance_tensor, requires_grad=True).to(self.rest_pose.device)
        self.means = nn.Parameter(data=self.means_tensor, requires_grad=True).to(self.rest_pose.device)
        # self.register_buffer('rest_pose', rest_pose, persistent=False)

    def forward(self,
                kp3d: torch.Tensor,
                bones: torch.Tensor,
                kp_idxs: torch.LongTensor,
                unroll: bool = True) -> Dict[str, Any]:
        """
        Parameters
        ----------
        kp3d: torch.Tensor (B, J, 3) -- joint locations
        bone_poses: torch.Tensor (B, J, 3 or 6) -- rotation vector for each joint/bone
        kp_idxs: torch.LongTensor -- index of the poses
        unroll: bool -- unroll the for loop for kinematic calculation, this gives faster forward pass
        """
        bone_poses = torch.clone(bones)
        samples = torch.normal(0, 1, size=bone_poses.shape).to(bones.device)
        means = self.means[kp_idxs]
        vars = self.variance_tensor[kp_idxs]
        perturbations = means + torch.mul(samples, vars)
        bone_poses[:, 17, :] += perturbations[:, 17, :]
        pelvis = torch.tensor(kp3d[:, self.skel_type.root_id])
        rest_pose = self.rest_pose.clone()

        # before_fig = plot_skeleton3d(kp3d[0].detach().cpu().numpy())
        # before_fig.write_html("/home/james/Desktop/Courses/449CPSC/NPC-pytorch/outputs/eval/vis/befor_viz.html")

        kp, skts = calculate_kinematic(
            rest_pose[None],
            bone_poses,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        # after_fig = plot_skeleton3d(kp[0].detach().cpu().numpy())
        # after_fig.write_html("/home/james/Desktop/Courses/449CPSC/NPC-pytorch/outputs/eval/vis/after_viz.html")

        return dict(
            kp3d=kp,
            skts=skts,
            bones=bone_poses
        )
