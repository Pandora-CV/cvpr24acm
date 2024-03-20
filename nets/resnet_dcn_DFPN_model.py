import torch.nn as nn
import torch
import math
import sys
sys.path.append('./nets')
from utils.Gaussian_focal_loss import losses
from resnet_dcn_DFPN import get_pose_net
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import Softmax
import numpy as np
from utils.angle_coders import ACMCoder, PSCCoder, CSLCoder, PseudoAngleCoder

class ResNet(nn.Module):

    def __init__(self, num_layers, num_cls=15, coder='acm', coder_cfg=1, coder_mode='model', box_loss='kfiou'):
        super(ResNet, self).__init__()

        self.coder_type = coder
        self.coder_mode = coder_mode
        if coder == 'acm':
            self.angle_coder = ACMCoder(dual_freq=(coder_cfg >= 0))
        elif coder == 'psc':
            self.angle_coder = PSCCoder(N=coder_cfg)
        elif coder == 'csl':
            self.angle_coder = CSLCoder(N=coder_cfg)
        else:
            self.angle_coder = PseudoAngleCoder()

        heads = {'hm': num_cls,'wh': 2 ,'reg': 2, 'theta':self.angle_coder.encode_size if coder_mode == 'model' else 1}
        print(heads)
        self.num_cls = num_cls
        self.backbone=get_pose_net(num_layers=num_layers,heads=heads)
        self.losses=losses(self.angle_coder, coder_mode=coder_mode, box_loss=box_loss)

        self.sig = nn.Sigmoid()
        self.softmax = Softmax(dim=1)

    def forward(self, input):
        if self.training:
            x=input['img'] #cuda
            label=input['label']
            heatmap_t=input['heatmap_t'] #cuda
            theta_g= input['theta']
        else :
            x = input

        out=self.backbone(x)[0]
        heatmap=self.sig(out['hm'])#(N,15,H/4,W/4)
        scale=out['wh']#(N,2,H/4,W/4)
        offset=out['reg']#(N,2,H/4,W/4)

        if self.coder_type in ['acm', 'psc']:
            theta_p=torch.tanh(out['theta'])#(N,2,H/4,W/4)
        elif self.coder_type in ['csl']:
            theta_p=torch.sigmoid(out['theta'])#(N,2,H/4,W/4)
        else:
            theta_p=out['theta']

        if self.training:
            return self.losses(heatmap,scale,offset,theta_p,heatmap_t,label,theta_g)
        else:
            return [heatmap,scale,offset,theta_p]
        
    def _decode(self, heatmap, scale, offset,theta, process_H, process_W, scorethr):
        '''
        heatmap (process_H/4,process_W/4) tensor cpu
        scale (1,2,process_H/4,process_W/4) tensor cpu
        offset (1,2,process_H/4,process_W/4) tensor cpu
        theta (1,180,process_H/4,process_W/4) tensor cpu
        process_H,process_W 输入网络中的图片尺寸
        '''
        heatmap = heatmap.squeeze().numpy()  # (process_H/4,process_W/4)
        scale0, scale1 = scale[0, 0, :, :].numpy(), scale[0, 1, :, :].numpy()  # (process_H/4,process_W/4)
        offset0, offset1 = offset[0, 0, :, :].numpy(), offset[0, 1, :,:].numpy()  # (process_H/4,process_W/4)
        theta = theta[0,:, :, :] #(2,process_H/4,process_W/4)

        c0, c1 = np.where(heatmap > scorethr)
        boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                cx, cy = max(0, (c1[i] + o1 + 0.5) * 4), max(0, (c0[i] + o0 + 0.5) * 4)
                cx, cy = min(cx, process_W), min(cy, process_H)
                
                if self.coder_mode == 'loss':
                    angle = theta[0, c0[i], c1[i]].item()
                elif self.coder_mode == 'model':
                    angle_ebd = theta[:, c0[i], c1[i]].reshape((1, -1)) #(1, emd_dim)
                    angle = self.angle_coder.decode(angle_ebd).item() # (1,)
                else:
                    raise NotImplementedError
            
                angle = angle / math.pi * 180

                boxes.append([cx, cy, s0, s1, angle, s])

            boxes = np.asarray(boxes, dtype=np.float32)
        return boxes #boxes (num_objs,6) (cx,cy,h,w,theta,s)  均为process_H,process_W尺度上的预测结果

    def decode_per_img(self, heatmap,scale,offset,theta,H,W,scorethr):
        '''
        :param heatmap: (1,NUM_CLASSES,H/4,W/4) CUDA //after sigmoid
        :param scale: (1,2,H/4,W/4) CUDA
        :param offset: (1,2,H/4,W/4) CUDA //after tanh
        :param theta: (1,180,H/4,W/4) CUDA //after sigmoid
        '''
        pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        h_p = pooling(heatmap)
        heatmap[heatmap != h_p] = 0

        results=[]
        for i in range(self.num_cls):
            bboxs=self._decode(heatmap[0,i,:,:].cpu(),scale.cpu(),offset.cpu(),theta.cpu(),H,W,scorethr)#(num_objs,6) (cx,cy,h,w,theta,s)
            if len(bboxs)>0:
                sigle_result = np.zeros((len(bboxs),7),dtype=np.float32)
                sigle_result[:,:5] = bboxs[:,:5]
                sigle_result[:,5] = i
                sigle_result[:,6] = bboxs[:,5]
                results.append(sigle_result)
        if len(results) > 0:
            results = np.concatenate(results, axis=0)
        return results
        #(total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

