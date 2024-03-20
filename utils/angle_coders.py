# Copyright (c) OpenMMLab. All rights reserved.
# Modified from open-mmlab: https://github.com/open-mmlab/mmrotate

import math

import torch
from torch import Tensor
import math
import torch.nn.functional as F


def smooth_l1_loss(pred, target, beta=1.0 / 9.0):
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

def focal_loss(pred,
               target,
               gamma=2.0,
               alpha=0.25,):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') #* focal_weight
    #print(pred.shape, target.shape, loss.shape)
    return loss


def gaussian_focal_loss(pred: Tensor,
                        gaussian_target: Tensor,
                        alpha: float = 2.0,
                        gamma: float = 4.0,
                        pos_weight: float = 1.0,
                        neg_weight: float = 1.0) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_weight * pos_loss + neg_weight * neg_loss


class CSLCoder:
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version='le', N=180, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135', 'le']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45, 'le': 0}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.window = window
        self.radius = radius
        self.encode_size = N
        self.omega = int(self.angle_range // self.encode_size)

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            Tensor: The csl encoding of angle offset for each scale
            level. Has shape (num_anchors * H * W, encode_size)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.encode_size
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = torch.exp(-torch.pow(base_radius_range.float(), 2.) / (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)
            
        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, encode_size) or
                (B, num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.


        Returns:
            Tensor: Angle offset for each scale level. When keepdim is true,
            return (num_anchors * H * W, 1) or (B, num_anchors * H * W, 1),
            otherwise (num_anchors * H * W,) or (B, num_anchors * H * W)
        """
        if angle_preds.shape[0] == 0:
            shape = list(angle_preds.size())
            if keepdim:
                shape[-1] = 1
            else:
                shape = shape[:-1]
            return angle_preds.new_zeros(shape)
        angle_cls_inds = torch.argmax(angle_preds, dim=-1, keepdim=keepdim)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)

    def loss(self, pred, target, avg_factor):
        loss = gaussian_focal_loss(pred, target)
        return loss.sum() / avg_factor

class PSCCoder:
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        N (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str = 'le90',
                 dual_freq: bool = True,
                 N: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = N
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * math.pi) - math.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase / 2
        return angle_pred
    
    def loss(self, pred, target, avg_factor):
        loss = smooth_l1_loss(pred, target)
        return loss.sum() / avg_factor / self.encode_size


class PseudoAngleCoder:
    """Pseudo Angle Coder."""

    def __init__(self, N=-1):
        self.encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)
        
    def loss(self, pred, target, avg_factor):
        loss = smooth_l1_loss(pred, target)
        return loss.sum() / avg_factor
        

class ACMCoder:
    """Angle Correct Moule (ACM) Coder.

    Encoding formula:
        z = f (\theta) = e^{j\omega\theta} 
          = \cos(\omega\theta) + j \sin(\omega\theta)

    Decoding formula:
        \theta = f^{-1}(z) = -\frac{j}{\omega} \ln z 
               = \frac{1}{\omega}((\mathrm{arctan2}(z_{im}, z_{re}) + 2\pi) \bmod 2\pi) 

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        N (int, optional): just for aligning coder interface.
    """

    def __init__(self,
                 angle_version: str = 'le90',
                 base_omega=2,
                 N=-1,
                 dual_freq: bool = True):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.encode_size = 4 if self.dual_freq else 2
        self.base_omega = base_omega


    def encode(self, angle_targets: Tensor) -> Tensor:
        """Angle Correct Moule (ACM) Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        angle2 = self.base_omega * angle_targets
        cos2= torch.cos(angle2)
        sin2 = torch.sin(angle2)

        if self.dual_freq:
            angle4 = 2* self.base_omega * angle_targets
            cos4 = torch.cos(angle4)
            sin4 = torch.sin(angle4)

        return torch.cat([cos2, sin2, cos4, sin4], dim=-1) if self.dual_freq else torch.cat([cos2, sin2], dim=-1)


    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Angle Correct Moule (ACM) Decoder.

        Args:
            angle_preds (Tensor): The acm encoded-angle
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        
        if self.dual_freq:
            cos2, sin2, cos4, sin4 = angle_preds.unbind(dim=-1)
            angle_a = torch.atan2(sin2, cos2)     # 2x
            angle_b = torch.atan2(sin4, cos4) / 2 # 4x -> 2x

            idx = torch.cos(angle_a) * torch.cos(angle_b) + torch.sin(
               angle_a) * torch.sin(angle_b) < 0
            angle_b[idx] = angle_b[idx] % (2 * math.pi) - math.pi
    
            angles = angle_b / 2

        else:
            sin2, cos2 = angle_preds.unbind(dim=-1)
            cos2, sin2 = sin2, cos2
            angles = torch.atan2(sin2, cos2) / self.base_omega

        if keepdim:
            angles = angles.unsqueeze(dim=-1)

        return angles
    
    def loss(self, pred, target, avg_factor):
        loss = smooth_l1_loss(pred, target)
        return loss.sum() / avg_factor


if __name__ == '__main__':
    csl = CSLCoder(angle_version='le', N=180, window='gaussian', radius=6)
    angle = torch.tensor([30, 30], dtype=torch.float).reshape((-1, 1)).deg2rad()
    code = csl.encode(angle)
    print(code.shape)
    print(csl.decode(code).rad2deg())
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.arange(180), code[0, :].numpy())
    plt.savefig('gs.png')

    pred = code[[0], :]
    target = code[[1], :]
    print((pred == target).all())
    loss = csl.loss(pred, target, 1)
    print(loss)


    eps = 1e-12
    alpha = 2
    beta = 4
    pos_weight = target.eq(1)
    neg_weight = (1 - target).pow(beta)
    pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred + eps) * pos_weight # 正样本损失 （s_sum）
    neg_loss =  -torch.pow(pred, alpha) * torch.log(1 - pred + eps) * neg_weight # 负样本损失 (N,1,H/4,W/4)

    #print(pos_loss, neg_loss)

    center_loss = (pos_loss.sum() + neg_loss.sum())
    print(center_loss)