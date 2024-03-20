import numpy as np
import math
import cv2
import os
import datetime
import torch
import random
from contextlib import contextmanager
from easydict import EasyDict

def gaussian_radius(det_size, min_overlap):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def creat_label_heatmap(img,label,num_classes=15,min_overlap=0.5):
    '''
    img (N,C,H,W) tensor
    label:list list中的每个元素是（num_obj,6）[cx,cy,h,w,theta,class] np
    '''
    N = img.size(0)
    heatmap_t = np.zeros((N, num_classes, int(img.size(2) / 4), int(img.size(3) / 4))).astype(np.float32)
    for i in range(N):
        for j in range(len(label[i])):
            cx, cy, h, w, theta, c = label[i][j]
            #xmin, ymin, h, w, c = label[i][j][0], label[i][j][1], label[i][j][3], label[i][j][2], int(label[i][j][4])
            radius = gaussian_radius((math.ceil(h / 4.0), math.ceil(w / 4.0)),min_overlap=min_overlap)
            radius = max(0, int(radius))
            
            ct = np.array([cx / 4.0, cy / 4.0], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            
            heatmap_t[i, int(c), :, :] = draw_umich_gaussian(heatmap_t[i, int(c), :, :], ct_int, radius)
    return torch.from_numpy(heatmap_t) #(N,15,H/4,W/4)  tensor

#postprocess


'''
def decode(heatmap, scale, offset,theta, process_H, process_W, scorethr):
    #
    #heatmap (process_H/4,process_W/4) tensor cpu
    #scale (1,2,process_H/4,process_W/4) tensor cpu
    #offset (1,2,process_H/4,process_W/4) tensor cpu
    #theta (1,180,process_H/4,process_W/4) tensor cpu
    #process_H,process_W 输入网络中的图片尺寸
    #
    dual_freq = True
    freq = 2
    heatmap = heatmap.squeeze().numpy()  # (process_H/4,process_W/4)
    scale0, scale1 = scale[0, 0, :, :].numpy(), scale[0, 1, :, :].numpy()  # (process_H/4,process_W/4)
    offset0, offset1 = offset[0, 0, :, :].numpy(), offset[0, 1, :,:].numpy()  # (process_H/4,process_W/4)
    theta = theta.squeeze().numpy() #(2,process_H/4,process_W/4)

    c0, c1 = np.where(heatmap > scorethr)
    boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            cx, cy = max(0, (c1[i] + o1 + 0.5) * 4), max(0, (c0[i] + o0 + 0.5) * 4)
            cx, cy = min(cx, process_W), min(cy, process_H)
            # angle =  theta[:, c0[i], c1[i]].argmax()
            if dual_freq:
                cos_p2 = theta[0, c0[i], c1[i]]
                sin_p2 = theta[1, c0[i], c1[i]]
                cos_p4 = theta[2, c0[i], c1[i]]
                sin_p4 = theta[3, c0[i], c1[i]]
                # import pdb; pdb.set_trace()
                phase = torch.atan2(torch.tensor(sin_p2), torch.tensor(cos_p2))
                phase2 = torch.atan2(torch.tensor(sin_p4), torch.tensor(cos_p4)) / 2
                idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                    phase) * torch.sin(phase2) < 0
                # Add pi to phase2 and keep it in range [-pi,pi)
                phase2[idx] = phase2[idx] % (2 * np.pi) - np.pi
                angle = phase2.numpy() / 2
            else:
                cos_p = theta[0, c0[i], c1[i]]
                sin_p = theta[1, c0[i], c1[i]]
                angle = np.arctan2(sin_p, cos_p) / freq

            angle = angle / math.pi * 180
            # print(theta[0, c0[i], c1[i]].shape)
            # import pdb; pdb.set_trace()
            # angle = theta[c0[i], c1[i]] / math.pi * 180
            #print(angle)
            #print(torch.from_numpy(theta[:, c0[i], c1[i]]))

            boxes.append([cx, cy, s0, s1, angle, s])

        boxes = np.asarray(boxes, dtype=np.float32)
    return boxes
    #boxes (num_objs,6) (cx,cy,h,w,theta,s)  均为process_H,process_W尺度上的预测结果

def decode_per_img(heatmap,scale,offset,theta,H,W,scorethr,NUM_CLASSES = 15):
    #
    #:param heatmap: (1,NUM_CLASSES,H/4,W/4) CUDA //after sigmoid
    #:param scale: (1,2,H/4,W/4) CUDA
    #param offset: (1,2,H/4,W/4) CUDA //after tanh
    #:param theta: (1,180,H/4,W/4) CUDA //after sigmoid
    #
    pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    h_p = pooling(heatmap)
    heatmap[heatmap != h_p] = 0

    results=[]
    for i in range(NUM_CLASSES):
        bboxs=decode(heatmap[0,i,:,:].cpu(),scale.cpu(),offset.cpu(),theta.cpu(),H,W,scorethr)#(num_objs,6) (cx,cy,h,w,theta,s)
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

'''


if __name__ == "__main__" :
    x=torch.randn(1,3,512,512)
    label=[np.array([[24,44,80,40,0],[192,192,128,128,0]])]
    ht=creat_label_heatmap(x,label)
    

    cv2.imshow('ht', (ht.numpy()[0,0,:,:]*255).astype(np.uint8) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()