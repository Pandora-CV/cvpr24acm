import numpy as np
import torch
import json
import cv2
import os
import math
import random
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
import sys
from torch.nn import functional as F
from torch.utils.data import Dataset
from utils.utils import creat_label_heatmap
from utils.smooth_label import gaussian_label
from utils.aug import rotate_image
from DOTA_devkit.DOTA import DOTA

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

class DotaSetv1(Dataset):
    
    #rgb
    ori_img_size = 1024
    mean = np.array([[[0.485, 0.456, 0.406]]],dtype=np.float32)
    std  = np.array([[[0.229, 0.224, 0.225]]],dtype=np.float32)

    def __init__(self, root_dir,img_size=1024):
        self.numclasses = len(wordname_15)
        self.class2label = {}
        self.label2class = {}
        for i in range(self.numclasses):
            self.class2label[wordname_15[i]] = i
            self.label2class[i] = wordname_15[i]
        self.imgsize = img_size


        self.DOTA = DOTA(root_dir)
        self.imgids = self.DOTA.getImgIds()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img = self.DOTA.loadImgs(imgid)[0] #bgr 0~255 np
        ann = self.DOTA.loadAnns(imgId=imgid)

        ann = self.convert_to_minAreaRect(ann) #将原始不规则四边形标注转化为最小旋转外接矩形标注
        H,W,_ = img.shape
        if H != self.ori_img_size or W != self.ori_img_size:
            new_img = np.zeros((self.ori_img_size,self.ori_img_size,3),dtype=np.float32)
            new_img[:H, :W, :] = img
            img = new_img

        img,ann = self.flip_aug(img,ann)
        img,ann = self.rot_aug(img,ann,max_angle=10)
        img = self.gray_aug(img)
        #img,ann = self.resize(img,ann,self.imgsize)

        converted_ann = self.convert_poly2cxcyhw(ann) #(N,6) np.float32 [cx,cy,h,w,theta(0~179),class]
        img = self.normalize(img) #rgb (h,w,c) 标准化 np

        return torch.from_numpy(img.transpose(2,0,1)), converted_ann


    def convert_poly2cxcyhw(self,ann):
        #h ->long side， w ->short side
        converted_ann = np.zeros((0,6),dtype=np.float32) #cx,cy,h,w,theta,class
        for i in range(len(ann)):
            p1,p2,p3,p4 =  ann[i]['poly']
            cx = ((p1[0]+p3[0])/2.0 + (p2[0]+p4[0])/2.0)/2.0
            cy = ((p1[1]+p3[1])/2.0 + (p2[1]+p4[1])/2.0)/2.0
            side1 = self.cal_line_length(p1,p2)
            side2 = self.cal_line_length(p2,p3)
            if side1>side2:
                r1,r2 = p1,p2
                long_side = side1
                short_side = side2
            else:
                r1,r2 = p2,p3
                long_side = side2
                short_side = side1
            if long_side < 2.0 or short_side < 2.0:
                continue

            if r1[1]<r2[1]:
                xx = r1[0] - r2[0]
            else:
                xx = r2[0] - r1[0]
            theta = round((math.acos(xx/long_side)/math.pi)*180)%180 #[0,179]
            cls = self.class2label[ann[i]['name']]

            a = np.zeros((1, 6),dtype=np.float32)
            a[0] = cx,cy,long_side,short_side,theta,cls
            converted_ann = np.append(converted_ann, a, axis=0)
        return converted_ann

    def convert_cxcyhw2poly(self,converted_ann):
        ann = []
        for i in range(len(converted_ann)):
            cx,cy,long_side,short_side,theta,cls = converted_ann[i]
            half_cross_line = math.sqrt(math.pow(long_side, 2) + math.pow(short_side, 2))/2.0 #半对角线长度
            p1 = (cx + half_cross_line * math.cos(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)),
                  cy - half_cross_line * math.sin(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)))
            p2 = (cx + half_cross_line * math.cos(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)),
                  cy - half_cross_line * math.sin(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)))
            p3 = (cx - half_cross_line * math.cos(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)),
                  cy + half_cross_line * math.sin(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)))
            p4 = (cx - half_cross_line * math.cos(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)),
                  cy + half_cross_line * math.sin(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)))
            ann.append(dict(poly=[p1,p2,p3,p4],name=self.label2class[int(cls)]))

        return ann

    def display(self,img,ann, img_name='demo.png'):
        # img (h,w,c) bgr 0~255
        # ann poly type
        np.random.seed(1)

        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []

        base_color = []
        for i in range(15):
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            base_color.append(c)
        base_color[0] = [0.2, 1.0, 0.976]
        base_color[1] = [1.0, 0.6, 0]
        base_color[2] = [1, 0.8, 0.8]
        base_color[3] = [0.4, 0.2, 0]
        base_color[4] = [1, 1, 0.2]
        base_color[5] = [1, 0, 0.8]
        base_color[6] = [1, 0, 0]
        base_color[7] = [0, 0.8, 1]
        base_color[8] = [1, 0.8, 0]
        base_color[9] = [1, 0.4, 0]
        base_color[10] = [0, 0.4, 0]
        base_color[11] = [0.6, 0.8, 1]
        base_color[12] = [0, 1, 0]
        base_color[13] = [0.8, 0.6, 0.8]
        base_color[14] = [0.6, 0.8, 0.2]

        for obj in ann:
            # if self.class2label[obj['name']] != 4:continue
            c = base_color[int(self.class2label[obj['name']])]
            # print(obj['name'])
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)

        # p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=0.8)
        ax.add_collection(p)

        plt.savefig(img_name, bbox_inches='tight')
        plt.clf()
        plt.close('all')

    def cal_line_length(self,point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    def flip_aug(self,img,ann,flip_x=0.5,flip_y=0.5):

        if np.random.rand() < flip_x:
            img = img[:,::-1,:]
            for i in range(len(ann)):
                p1,p2,p3,p4 = ann[i]['poly']
                p1_aug = (self.ori_img_size - p1[0], p1[1])
                p2_aug = (self.ori_img_size - p2[0], p2[1])
                p3_aug = (self.ori_img_size - p3[0], p3[1])
                p4_aug = (self.ori_img_size - p4[0], p4[1])
                ann[i]['poly'] = [p1_aug,p2_aug,p3_aug,p4_aug]

        if np.random.rand() < flip_y:
            img = img[::-1,:,:]
            for i in range(len(ann)):
                p1,p2,p3,p4 = ann[i]['poly']
                p1_aug = (p1[0], self.ori_img_size - p1[1])
                p2_aug = (p2[0], self.ori_img_size - p2[1])
                p3_aug = (p3[0], self.ori_img_size - p3[1])
                p4_aug = (p4[0], self.ori_img_size - p4[1])
                ann[i]['poly'] = [p1_aug, p2_aug, p3_aug, p4_aug]

        return img,ann

    def resize(self,img,ann,size):
        img = cv2.resize(img,(size,size))
        ratio = size * 1. / self.ori_img_size
        for i in range(len(ann)):
            p1, p2, p3, p4 = ann[i]['poly']
            p1_aug = (p1[0]*ratio , p1[1]*ratio)
            p2_aug = (p2[0]*ratio , p2[1]*ratio)
            p3_aug = (p3[0]*ratio , p3[1]*ratio)
            p4_aug = (p4[0]*ratio , p4[1]*ratio)
            ann[i]['poly'] = [p1_aug, p2_aug, p3_aug, p4_aug]

        return img,ann

    def normalize(self,img):
        #img bgr (h,w,c) 0~255
        img = img[:,:,::-1].astype(np.float32)/255.0
        img = (img-self.mean)/self.std

        return img

    def convert_to_minAreaRect(self,ann):
        for i in range(len(ann)):
            p1,p2,p3,p4 = ann[i]['poly']
            ori_box = np.array([[p1[0], p1[1]],[p2[0], p2[1]],[p3[0], p3[1]],[p4[0], p4[1]]],dtype=np.int32)
            rect = cv2.minAreaRect(ori_box)
            rec_box = cv2.boxPoints(rect)
            p1_new = (rec_box[0][0], rec_box[0][1])
            p2_new = (rec_box[1][0], rec_box[1][1])
            p3_new = (rec_box[2][0], rec_box[2][1])
            p4_new = (rec_box[3][0], rec_box[3][1])
            ann[i]['poly'] = [p1_new, p2_new, p3_new, p4_new]

        return ann

    def rot_aug(self,img,ann,max_angle=10):
        H, W, _ = img.shape
        assert H == W

        list_ann = []
        for a in ann:
            p1,p2,p3,p4 = a['poly']
            bbox = [p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1]]
            list_ann.append(([],bbox))
        angle = np.random.randint(-1*max_angle,max_angle+1) + np.random.choice([0,90])

        img, bbox_new_list = rotate_image(img,list_ann,angle=angle)

        new_H, new_W, c = img.shape
        assert new_H == new_W
        img = cv2.resize(img,(W,H))
        for i in range(len(bbox_new_list)):
            bbox = bbox_new_list[i][1]
            p1_new = (bbox[0]*1.*H/new_H, bbox[1]*1.*H/new_H)
            p2_new = (bbox[2]*1.*H/new_H, bbox[3]*1.*H/new_H)
            p3_new = (bbox[4]*1.*H/new_H, bbox[5]*1.*H/new_H)
            p4_new = (bbox[6]*1.*H/new_H, bbox[7]*1.*H/new_H)
            ann[i]['poly'] = [p1_new, p2_new, p3_new, p4_new]

        return img,ann

    def gray_aug(self,img,p=0.1):
        if np.random.rand() < p:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

def collater(data):
    ori_size = 1024
    input_size = 1024
    #input_size = random.choice([960,992,1024,1056,1088])

    imgs=[]
    annos=[]
    for img,anno in data :
        imgs.append(img)
        if len(anno) > 0:
            anno[:,:4] = anno[:,:4]*1.*input_size / ori_size
            mask = (anno[:,3] >= 4.0)
            anno = anno[mask]
        annos.append(anno)
    imgs = torch.stack(imgs,dim=0) #(N,C,H,W)
    imgs = F.interpolate(imgs, size=(input_size,input_size), mode='bilinear', align_corners=True)

    heatmap_t = creat_label_heatmap(imgs,annos,num_classes=15,min_overlap=0.5) #(N,15,H/4,W/4)
    theta=[]
    for anno in annos:
        smooth = np.zeros((len(anno),1),dtype=np.float32)
        for i in range(len(anno)):
            # smooth[i,:] = gaussian_label(int(anno[i][4]),180,sig=6).astype(np.float32)
            smooth[i, :] = anno[i][4] / 180 * math.pi
        theta.append(torch.from_numpy(smooth))

    return {'img':imgs,'label':annos,'heatmap_t':heatmap_t,'theta':theta}



if __name__ == "__main__":
    sys.path.append('..')
    sys.path.append('../DOTA_devkit')
    from torch.utils.data import DataLoader

    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    trainset = DotaSetv1('/home/wj/Detection/DOTA/datasets/trainvalsplit-1024-256')

    img, label = trainset[2000]
    print(img.size())

    img = (img.numpy().transpose(1, 2, 0) * std + mean) * 255.0
    img = img[:, :, ::-1].astype(np.uint8)
    ann = trainset.convert_cxcyhw2poly(label)
    trainset.display(img, ann)
