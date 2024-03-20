# Copyright (c) OpenMMLab. All rights reserved.
# Modified from open-mmlab: https://github.com/open-mmlab/mmdet
# Modified from open-mmlab: https://github.com/open-mmlab/mmrotate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.ops import diff_iou_rotated_2d

def xy_wh_r_2_xy_sigma(whr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 3).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = whr.shape
    assert _shape[-1] == 3
    wh = whr[..., :2].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = whr[..., 2]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return sigma


def kfiou_loss(pred_box,
            target_box,
            fun='',
            eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    # xy_p = pred[:, :2]
    # xy_t = target[:, :2]
    Sigma_p = xy_wh_r_2_xy_sigma(pred_box)
    Sigma_t = xy_wh_r_2_xy_sigma(target_box)

    # Smooth-L1 norm
    # diff = torch.abs(xy_p - xy_t)
    # xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                     diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().clamp(eps).sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (kf_loss).clamp(0)

    return loss

def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = xy_wh_r_2_xy_sigma(pred)
    Sigma_t = xy_wh_r_2_xy_sigma(target)
    # xy_p, Sigma_p = pred
    # xy_t, Sigma_t = target

    # xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)

def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    Sigma_p = xy_wh_r_2_xy_sigma(pred)
    Sigma_t = xy_wh_r_2_xy_sigma(target)
    # xy_p, Sigma_p = pred
    # xy_t, Sigma_t = target

    # _shape = xy_p.shape

    # xy_p = xy_p.reshape(-1, 2)
    # xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    # dxy = (xy_p - xy_t).unsqueeze(-1)
    # xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = whr_distance
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    # distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)

def rotated_iou_loss(pred, target, linear=False, mode='linear', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')


    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss



class losses(nn.Module):

    use_smooth = True

    def __init__(self, coder, coder_mode='model', box_loss='kfiou'):
        super().__init__()
        self.angle_coder = coder
        self.coder_mode = coder_mode
        self.box_loss = box_loss

    def forward(self, heatmap,scale,offset,theta_p,heatmap_t,label,theta_g):
        '''
        heatmap: (N,15,H/4,W/4) cuda  0~1
        scale: (N,2,H/4,W/4) cuda \\if use gaussian  (N,4,H/4,W/4)
        offset: (N,2,H/4,W/4) cuda \\if use gaussian  (N,4,H/4,W/4) 未经tanh激活
        theta_p: (N,1,H/4,W/4) cuda 0~1 网络的输出
        heatmap_t: (N,15,H/4,W/4) tensor cuda float32
        label: list list中的每个元素是（num_obj,6）[cx,cy,h,w,theta,class] np
        theta_g:list list中的每个元素是（num_obj,1）tensor cpu GT

        '''
        alpha=2.0
        beita=4.0
        eps=0.00001

        label_new=[]
        for i in range(len(label)):
            if len(label[i]) == 0: continue
            l=torch.zeros((label[i].shape[0],7))
            l[:,1:7]=torch.from_numpy(label[i])
            l[:,0]=i
            label_new.append(l)
        if len(label_new) == 0:
            return 0*heatmap.sum(), 0*scale.sum(), 0*offset.sum(), 0*theta_p.sum()
        label_new = torch.cat(label_new, 0)  # (s_num,7) torch.float32 cpu [idx,cx,cy,h,w,theta,class]

        N = max(1, label_new.size(0))

        idx = label_new[:, 0].long()
        cx = (label_new[:, 1]/4.0).long()
        cy = (label_new[:, 2]/4.0).long()
        clss = label_new[:, 6].long()

        '''compute center point loss'''

        pos = heatmap[idx, clss, cy, cx]
        pos_loss = -torch.pow(1 - pos, alpha) * torch.log(pos + eps)  # 正样本损失 （s_sum）

        neg_loss = -torch.pow(1 - heatmap_t, beita) * torch.pow(heatmap, alpha) * torch.log(1 - heatmap + eps)  # 负样本损失 (N,1,H/4,W/4)

        center_loss = (pos_loss.sum() + neg_loss.sum()) / N

        '''compute scale&offset loss'''
        scale_ph = torch.clamp(scale[idx, 0, cy, cx], max=math.log(5000 / 4.0))
        scale_pw = torch.clamp(scale[idx, 1, cy, cx], max=math.log(5000 / 4.0))
        #scale_ph = scale[idx, 0, cy, cx]
        #scale_pw = scale[idx, 1, cy, cx]
        offset_ph = torch.tanh(offset[idx, 0, cy, cx])
        offset_pw = torch.tanh(offset[idx, 1, cy, cx])

        scale_th = torch.log(label_new[:, 3] / 4.0).cuda()
        scale_tw = torch.log(label_new[:, 4] / 4.0).cuda()
        offset_th = (label_new[:, 2]/4.0 - (cy.float() + 0.5)).cuda()
        offset_tw = (label_new[:, 1]/4.0 - (cx.float() + 0.5)).cuda()

        # L1 loss
        if self.use_smooth == False :
            diff_s_h = torch.abs(scale_th - scale_ph)
            diff_s_w = torch.abs(scale_tw - scale_pw)
            diff_o_h = torch.abs(offset_th - offset_ph)
            diff_o_w = torch.abs(offset_tw - offset_pw)

            scale_loss = (diff_s_h.sum() + diff_s_w.sum()) / N
            offset_loss = (diff_o_h.sum() + diff_o_w.sum()) / N
        # smooth L1 loss
        else :
            diff_s_h = torch.abs(scale_th - scale_ph)
            diff_s_w = torch.abs(scale_tw - scale_pw)
            diff_o_h = torch.abs(offset_th - offset_ph)
            diff_o_w = torch.abs(offset_tw - offset_pw)

            scale_loss = (torch.where(torch.le(diff_s_h, 1.0 / 9.0),
                                      0.5 * 9.0 * torch.pow(diff_s_h, 2),
                                      diff_s_h - 0.5 / 9.0).sum() +
                          torch.where(torch.le(diff_s_w, 1.0 / 9.0),
                                      0.5 * 9.0 * torch.pow(diff_s_w, 2),
                                      diff_s_w - 0.5 / 9.0).sum()
                          ) / N

            offset_loss = (torch.where(torch.le(diff_o_h, 1.0 / 9.0),
                                       0.5 * 9.0 * torch.pow(diff_o_h, 2),
                                       diff_o_h - 0.5 / 9.0).sum() +
                           torch.where(torch.le(diff_o_w, 1.0 / 9.0),
                                       0.5 * 9.0 * torch.pow(diff_o_w, 2),
                                       diff_o_w - 0.5 / 9.0).sum()
                           ) / N


        '''compute theta loss'''
        theta_g = torch.cat(theta_g, dim=0).cuda() #(N, 1)
        theta_emd_g = self.angle_coder.encode(theta_g) #(N, emd_dim)

        if self.coder_mode == 'model':
            theta_emd_p = theta_p[idx, :, cy, cx]                        #(N, emd_dim)
            theta_p = self.angle_coder.decode(theta_emd_p, keepdim=True) #(N, 1)
        else:
            theta_p= theta_p[idx, :, cy, cx]               #(N, 1)
            theta_emd_p = self.angle_coder.encode(theta_p) #(N, emd_dim)
            #print(theta_emd_p.shape, theta_p.shape)

        theta_loss =  self.angle_coder.loss(theta_emd_p, theta_emd_g, N)
        
        '''compute box loss'''
        high_p = torch.exp(scale_ph).cuda()  # torch.Size([11])
        width_p = torch.exp(scale_pw).cuda() # torch.Size([11])
        high_g = (label_new[:, 3] / 4.0).cuda() # torch.Size([11])
        width_g = (label_new[:, 4] / 4.0).cuda() # torch.Size([11])

        box_p = torch.cat([width_p[:, None], high_p[:, None], theta_p], dim=1)
        box_g = torch.cat([width_g[:, None], high_g[:, None], theta_g], dim=1)

        if self.box_loss == 'gwd':
            box_loss = gwd_loss(box_p, box_g).sum() / N
        elif self.box_loss == 'kld':
            box_loss = kld_loss(box_p, box_g).sum() / N
        elif self.box_loss == 'kfiou':
            box_loss = kfiou_loss(box_p, box_g).sum() / N
        elif self.box_loss == 'riou':
            box_loss = rotated_iou_loss(box_p, box_g).sum() / N
        else:
            box_loss = 0

        theta_loss = theta_loss + box_loss
        #theta_loss = box_loss

        return center_loss, scale_loss, offset_loss, theta_loss
