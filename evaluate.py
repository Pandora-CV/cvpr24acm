from __future__ import print_function, division
import sys
sys.path.append('DOTA_devkit')
import os
import glob
import random
import math
import argparse
from datetime import datetime
import numpy as np
import cv2
import json
import shutil
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from nets.resnet_dcn_DFPN_model import ResNet
#from utils.utils import decode_per_img
from datasets.DotaDataset import DotaSetv1
import json

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=int, default=50)
parser.add_argument('--operation', help='must be one of [visualize,DOTA_test,DOTA_MS_test,HRSC_test,HRSC_MS_test]')
#parser.add_argument("--heads", default={'hm': 1,'wh': 2 ,'reg': 2, 'theta':4})
parser.add_argument("--heads", type=int, default=15)
parser.add_argument("--coder", default='acm')
parser.add_argument("--coder_cfg", default=1, type=int)
parser.add_argument("--coder_mode", default='model')
parser.add_argument("--box_loss", default='riou')
parser.add_argument('--weight_path', default='')
parser.add_argument('--score_thr', default=0.15)
#dota
parser.add_argument("--traindir", type=str, default='./DOTA_devkit/datasets/DOTA/trainvalsplit-1024-256')
parser.add_argument("--test_image_dir", type=str, default='./DOTA_devkit/datasets/DOTA/test4000/images')
parser.add_argument("--test_size", type=int, default=4000)
parser.add_argument("--output_id", type=int, default=0)
parser.add_argument('--img_path', default='./result/P0007.png')
#hrsc
parser.add_argument("--hrsc_test_size", type=int, default=640)
parser.add_argument("--hrsc_test_image_dir", type=str, default='/shared/datasets/HRSC2016/test/images')
parser.add_argument("--hrsc_test_annopath", type=str, default='/shared/datasets/HRSC2016/test/labelTxt/{:s}.txt')
parser.add_argument("--use_07_metric", action='store_true')
parser.add_argument("--ap_thres", type=float, default=0.50)

args = parser.parse_args()
print(args.weight_path, args.ap_thres)

TEST_IMG_SIZE = 4000

def inference_sigle_with_nms():
    workdir = './current'
    if not os.path.isdir(workdir):
        os.mkdir('./current')

    print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    img = cv2.imread(args.img_path)[:, :, ::-1]  # (H,W,C) RGB 0~255
    #img = cv2.imread(args.img_path)[:, :, ::-1]  # (H,W,C) RGB 0~255
    image = ((img.astype(np.float32) / 255.0) - mean) / std
    image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
    _, img_h, img_w = image.shape
    input_h, input_w = int(32 * np.ceil(img_h / 32.0)), int(32 * np.ceil(img_w / 32.0))
    input = np.zeros((3, input_h, input_w), dtype=np.float32)
    input[:, :img_h, :img_w] = image
    input = torch.from_numpy(input).unsqueeze(0).cuda()

    with torch.no_grad():
        heatmap, scale, offset, theta = model(input)
        results = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_h, input_w,args.score_thr)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

    assert len(results) != 0
    if os.path.isdir(os.path.join(workdir,'before_merge')):
        shutil.rmtree(os.path.join(workdir,'before_merge'))
    os.mkdir(os.path.join(workdir,'before_merge'))

    trainset = DotaSetv1(args.traindir)
    poly_ann = trainset.convert_cxcyhw2poly(results[:, :6])

    names = locals()
    for n in range(15):
        names['f' + str(n)] = open(os.path.join(workdir,'before_merge', trainset.label2class[n] + ".txt"), 'w')

    for i in range(len(results)):
        cx, cy, h, w, theta, cls, score = results[i]
        p1, p2, p3, p4 = poly_ann[i]['poly']
        names['f' + str(int(cls))].write("test__1__0___0")
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(score))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p1[0]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p1[1]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p2[0]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p2[1]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p3[0]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p3[1]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p4[0]))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(p4[1]))
        names['f' + str(int(cls))].write('\n')
    for n in range(15):
        names['f' + str(n)].close()

    from DOTA_devkit.ResultMerge_multi_process import mergebypoly
    if os.path.isdir(os.path.join(workdir, 'after_merge')):
        shutil.rmtree(os.path.join(workdir, 'after_merge'))
    os.mkdir(os.path.join(workdir,'after_merge'))
    mergebypoly(os.path.join(workdir,'before_merge'), os.path.join(workdir,'after_merge'))

    results = []
    for i in range(15):
        path = workdir + '/after_merge/' + trainset.label2class[i] + '.txt'
        with open(path, 'r') as f:
            sigle_class_data = f.readlines()
        if len(sigle_class_data) == 0: continue

        for j in range(len(sigle_class_data)):
            line = sigle_class_data[j].strip().split(' ')
            sigle = dict(poly=[(float(line[2]), float(line[3])), (float(line[4]), float(line[5])),
                               (float(line[6]), float(line[7])), (float(line[8]), float(line[9]))], name=trainset.label2class[i])
            if float(line[1]) > args.score_thr:
                results.append(sigle)

    trainset.display(img, results)
    shutil.rmtree(workdir)

def genarate_img_subimage_dict(sub_img_dir=args.test_image_dir):
    imgid=[]
    with open('./result/testID.txt','r') as f:
        file = f.readlines()
    for line in file:
        imgid.append(line.strip())
    #imgid ['P0006','P0009',...]

    img_subimage_dict = {}
    for id in imgid:
        img_subimage_dict[id] = []

    subimage_list = os.listdir(sub_img_dir)
    for imgname in subimage_list:
        curid = imgname.split('__')[0]
        img_subimage_dict[curid].append(imgname)

    sum = 0
    for key in img_subimage_dict.keys():
        sum += len(img_subimage_dict[key])
    assert sum == len(subimage_list)

    print("imgid-subimage-dict haved been genarated.")
    return img_subimage_dict

def inference_testset(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir=args.test_image_dir):
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1(args.traindir)

    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()

def inference_testset_FLIP(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir='../datasets/testsplit/images'):
    print('*****current test mood is H FLIP test*****')
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1(args.traindir)

    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            image = np.ascontiguousarray(image[:,:,::-1])

            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32
                #out = class_specific_decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size, 0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:, 0] = input_size - 1 - out[:, 0]
            out[:, 4] = (180 - out[:, 4]) % 180
            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()

def inference_testset_VFLIP(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir='../datasets/testsplit/images'):
    print('*****current test mood is V FLIP test*****')
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1(args.traindir)

    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            image = np.ascontiguousarray(image[:,::-1,:])

            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32
                #out = class_specific_decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size, 0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:, 1] = input_size - 1 - out[:, 1]
            out[:, 4] = (180 - out[:, 4]) % 180
            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()

def divide_into_class_specific(id_path_dict,src_path='./result/test1_ori',dst_path='./result/test1_before_merge'):
    names = locals()
    trainset = DotaSetv1(args.traindir)
    os.mkdir(dst_path)

    for imgid in id_path_dict.keys():
        src_txt_path = os.path.join(src_path,imgid+'.txt')
        with open(src_txt_path,'r') as f:
            data = f.readlines()
        for n in range(15):
            names['f'+str(n)] = open(os.path.join(dst_path,imgid+"__"+trainset.label2class[n]+".txt"),'w')

        for i in range(len(data)):
            line = data[i].strip().split(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[0])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[2])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[3])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[4])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[5])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[6])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[7])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[8])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[9])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[10])
            names['f' + str(trainset.class2label[line[1]])].write('\n')
        for n in range(15):
            names['f' + str(n)].close()

def merged2final(id_path_dict,final_dir='./result/test1_final',merged_dir='./result/test1_after_merge'):
    label2class = {0: 'plane', 1: 'baseball-diamond', 2: 'bridge', 3: 'ground-track-field', 4: 'small-vehicle',
                   5: 'large-vehicle', 6: 'ship', 7: 'tennis-court',
                   8: 'basketball-court', 9: 'storage-tank', 10: 'soccer-ball-field', 11: 'roundabout', 12: 'harbor',
                   13: 'swimming-pool', 14: 'helicopter'}

    for class_id in range(len(label2class)):
        classname = label2class[class_id]
        file = open(os.path.join(final_dir,'Task1_'+classname+'.txt'),'w')

        for img_id in id_path_dict.keys():
            ori_txt_path = merged_dir + "/" + img_id +"__" + classname+'.txt'
            with open(ori_txt_path,'r') as f:
                src = f.readlines()
            if len(src) == 0:continue

            for line_i in range(len(src)):
                line = src[line_i].strip().split(' ')
                file.write(line[0]), file.write(' ')
                file.write(line[1]), file.write(' ')
                file.write(line[2]), file.write(' ')
                file.write(line[3]), file.write(' ')
                file.write(line[4]), file.write(' ')
                file.write(line[5]), file.write(' ')
                file.write(line[6]), file.write(' ')
                file.write(line[7]), file.write(' ')
                file.write(line[8]), file.write(' ')
                file.write(line[9]), file.write('\n')

        file.close()

def eval_dota(out_id,input_size,is_flip=False,is_V=False):
    from DOTA_devkit.ResultMerge_multi_process import mergebypoly
    print("***********out id: ",out_id,"***********ori size: 4000, ***********image size: ",input_size)

    if os.path.isdir(os.path.join('./result', 'test'+str(out_id))):
        shutil.rmtree(os.path.join('./result', 'test'+str(out_id)))
    os.mkdir(os.path.join('./result', 'test'+str(out_id)))
    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'))# test results for each image without NMS. num = image_num * num_classes

    id_path_dict = genarate_img_subimage_dict(sub_img_dir=args.test_image_dir)
    if is_flip:
        if is_V:
            inference_testset_VFLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=TEST_IMG_SIZE, input_size=input_size)
        else:
            inference_testset_FLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=TEST_IMG_SIZE, input_size=input_size)
    else:
        inference_testset(id_path_dict, out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'), ori_size=TEST_IMG_SIZE, input_size=input_size)
    divide_into_class_specific(id_path_dict, src_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'),dst_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'))


    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))# test results for each image after NMS. num = image_num * num_classes
    mergebypoly(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'), os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))

    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'))# test results for each class. num = num_classes
    merged2final(id_path_dict, final_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'),merged_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))

def multiscale_test(id_path_dict,in_list,out_dir):
    for id in id_path_dict.keys():
        out_file = open(os.path.join(out_dir, id+'.txt'),'w')
        for result in in_list:
            path = os.path.join("./result", result.split('-')[0], result.split('-')[0]+'_ori', id+'.txt')
            with open(path,'r') as f:
                data = f.readlines()
            for line in data:
                out_file.write(line)
        out_file.close()

#for HRSC2016
def inference_testset_for_hrsc(img_size=640):
    outdir = './current'
    #args.heads['hm'] = 1
    #print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    trainset = DotaSetv1(args.traindir)
    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()


    imgpath_list = os.listdir(args.hrsc_test_image_dir)
    print("{} images for testing".format(len(imgpath_list)))
    if os.path.isdir(outdir):
        shutil.rmtree(os.path.join(outdir))
    os.mkdir(os.path.join(outdir))
    os.mkdir(os.path.join(outdir, 'before'))
    os.mkdir(os.path.join(outdir, 'after'))
    os.mkdir(os.path.join(outdir, 'final'))



    for img_path in tqdm(imgpath_list):
        img_id = img_path.split('.')[0]

        img = cv2.imread(os.path.join(args.hrsc_test_image_dir,img_path))[:,:,::-1]
        H, W, _ = img.shape
        long_side = max(H, W)
        ratio = img_size * 1. / long_side
        image = cv2.resize(img, (round(W * ratio), round(H * ratio)))

        image = ((image.astype(np.float32) / 255.0) - mean) / std
        image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
        _, img_h, img_w = image.shape
        input_h, input_w = int(32 * np.ceil(img_h / 32.0)), int(32 * np.ceil(img_w / 32.0))
        input = np.zeros((3, input_h, input_w), dtype=np.float32)
        input[:, :img_h, :img_w] = image
        input = torch.from_numpy(input).unsqueeze(0).cuda()

        with torch.no_grad():
            heatmap, scale, offset, theta = model(input)
            results = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_h, input_w,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            if len(results) != 0: results[:,:4] = results[:,:4]*1./ ratio



        f = open(os.path.join(outdir,'before',img_id+'.txt'), 'w')
        if len(results) == 0:
            f.close()
            continue
        poly_ann = trainset.convert_cxcyhw2poly(results[:, :6])

        for i in range(len(results)):
            cx, cy, h, w, theta, cls, score = results[i]
            p1, p2, p3, p4 = poly_ann[i]['poly']
            f.write(img_id+"__1__0___0"), f.write(" ")
            f.write(str(score)), f.write(" ")
            f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
            f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
            f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
            f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()
    print("finished generate before merged results")

    from DOTA_devkit.ResultMerge_multi_process import mergebypoly
    mergebypoly(os.path.join(outdir, 'before'), os.path.join(outdir, 'after'))
    print("finished generate after merged results")

    f = open(os.path.join(outdir, 'final', 'Task1_ship.txt'), 'w')

    for txtfile in os.listdir(os.path.join(outdir, 'after')):
        with open(os.path.join(outdir, 'after',txtfile),'r') as c:
            data = c.readlines()
        for i in range(len(data)):
            line = data[i].strip().split(' ')
            f.write(line[0]), f.write(" ")
            f.write(line[1]), f.write(" ")
            f.write(line[2]), f.write(" ")
            f.write(line[3]), f.write(" ")
            f.write(line[4]), f.write(" ")
            f.write(line[5]), f.write(" ")
            f.write(line[6]), f.write(" ")
            f.write(line[7]), f.write(" ")
            f.write(line[8]), f.write(" ")
            f.write(line[9]), f.write('\n')
    f.close()

    print("finished generate final merged results")

    from DOTA_devkit.dota_evaluation_task1 import voc_eval
    detpath = './current/final/Task1_{:s}.txt'
    annopath = args.hrsc_test_annopath
    imagesetfile = './result/hrsc_testID.txt'

    classnames = ['ship']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=args.use_07_metric)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)
        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
    # plt.show()
    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)

    shutil.rmtree(outdir)

def inference_testset_for_hrsc_MS(img_size=[512,640]):
    outdir = './current'
    #args.heads['hm'] = 1
    #print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    trainset = DotaSetv1(args.traindir)
    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()


    imgpath_list = os.listdir(args.hrsc_test_image_dir)
    print("{} images for testing".format(len(imgpath_list)))
    if os.path.isdir(outdir):
        shutil.rmtree(os.path.join(outdir))
    os.mkdir(os.path.join(outdir))
    os.mkdir(os.path.join(outdir, 'before'))
    os.mkdir(os.path.join(outdir, 'after'))
    os.mkdir(os.path.join(outdir, 'final'))



    for img_path in tqdm(imgpath_list):
        img_id = img_path.split('.')[0]

        f = open(os.path.join(outdir, 'before', img_id + '.txt'), 'w')
        img = cv2.imread(os.path.join(args.hrsc_test_image_dir,img_path))[:,:,::-1]
        H, W, _ = img.shape
        long_side = max(H, W)

        for size in range(img_size[0],img_size[1]+1,32):
            ratio = size * 1. / long_side
            image = cv2.resize(img, (round(W * ratio), round(H * ratio)))

            image = ((image.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            _, img_h, img_w = image.shape
            input_h, input_w = int(32 * np.ceil(img_h / 32.0)), int(32 * np.ceil(img_w / 32.0))
            input = np.zeros((3, input_h, input_w), dtype=np.float32)
            input[:, :img_h, :img_w] = image
            input = torch.from_numpy(input).unsqueeze(0).cuda()

            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                results = model.decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_h, input_w,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

                if len(results) != 0: results[:,:4] = results[:,:4]*1./ ratio




            if len(results) == 0:
                continue
            poly_ann = trainset.convert_cxcyhw2poly(results[:, :6])

            for i in range(len(results)):
                cx, cy, h, w, theta, cls, score = results[i]
                p1, p2, p3, p4 = poly_ann[i]['poly']
                f.write(img_id+"__1__0___0"), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()
    print("finished generate before merged results")

    from DOTA_devkit.ResultMerge_multi_process import mergebypoly
    mergebypoly(os.path.join(outdir, 'before'), os.path.join(outdir, 'after'))
    print("finished generate after merged results")

    f = open(os.path.join(outdir, 'final', 'Task1_ship.txt'), 'w')

    for txtfile in os.listdir(os.path.join(outdir, 'after')):
        with open(os.path.join(outdir, 'after',txtfile),'r') as c:
            data = c.readlines()
        for i in range(len(data)):
            line = data[i].strip().split(' ')
            f.write(line[0]), f.write(" ")
            f.write(line[1]), f.write(" ")
            f.write(line[2]), f.write(" ")
            f.write(line[3]), f.write(" ")
            f.write(line[4]), f.write(" ")
            f.write(line[5]), f.write(" ")
            f.write(line[6]), f.write(" ")
            f.write(line[7]), f.write(" ")
            f.write(line[8]), f.write(" ")
            f.write(line[9]), f.write('\n')
    f.close()

    print("finished generate final merged results")

    from DOTA_devkit.dota_evaluation_task1 import voc_eval
    detpath = './current/final/Task1_{:s}.txt'
    annopath = args.hrsc_test_annopath
    imagesetfile = './result/hrsc_testID.txt'

    classnames = ['ship']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=args.ap_thres,
                                 use_07_metric=args.use_07_metric)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)
        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
    # plt.show()
    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)

    shutil.rmtree(outdir)


if __name__ =="__main__":
    if args.operation == 'visualize':
        inference_sigle_with_nms()
    elif args.operation == 'DOTA_test':
        eval_dota(out_id=args.output_id,input_size=args.test_size,is_flip=False,is_V=False)
        '''
        is_flip=True & is_V=False : using horizontal flip testing.
        is_flip=True & is_V=True  : using vertical flip testing.
        '''
    elif args.operation == 'DOTA_MS_test':
        from DOTA_devkit.ResultMerge_multi_process import mergebypoly

        eval_dota(out_id=args.output_id + 0, input_size=2016)
        eval_dota(out_id=args.output_id + 1, input_size=2528)
        eval_dota(out_id=args.output_id + 2, input_size=3008)
        eval_dota(out_id=args.output_id + 3, input_size=3520)
        eval_dota(out_id=args.output_id + 4, input_size=4000)
        eval_dota(out_id=args.output_id + 5, input_size=4512)
        eval_dota(out_id=args.output_id + 6, input_size=5024)
        eval_dota(out_id=args.output_id + 7, input_size=5504)
        eval_dota(out_id=args.output_id + 8, input_size=6016)

        if os.path.isdir(os.path.join('./result', 'test')):
            shutil.rmtree(os.path.join('./result', 'test'))
        os.mkdir(os.path.join('./result', 'test'))
        os.mkdir(os.path.join('./result', 'test', 'test_ori'))

        id_path_dict = genarate_img_subimage_dict()
        multiscale_test(id_path_dict,in_list=['test{}-2016'.format(str(args.output_id + 0)), 'test{}-2528'.format(str(args.output_id + 1)),
                                              'test{}-3008'.format(str(args.output_id + 2)), 'test{}-3520'.format(str(args.output_id + 3)),
                                              'test{}-4000'.format(str(args.output_id + 4)), 'test{}-4512'.format(str(args.output_id + 5)),
                                              'test{}-5024'.format(str(args.output_id + 6)), 'test{}-5504'.format(str(args.output_id + 7)),
                                              'test{}-6016'.format(str(args.output_id + 8))],
                        out_dir='./result/test/test_ori')
        divide_into_class_specific(id_path_dict, src_path='./result/test/test_ori',dst_path='./result/test/test_before_merge')
        os.mkdir('./result/test/test_after_merge')
        mergebypoly('./result/test/test_before_merge', './result/test/test_after_merge')
        os.mkdir('./result/test/test_final')
        merged2final(id_path_dict, final_dir='./result/test/test_final',merged_dir='./result/test/test_after_merge')
    elif args.operation == 'HRSC_test':
        inference_testset_for_hrsc(img_size=args.hrsc_test_size)
    elif args.operation == 'HRSC_MS_test':
        inference_testset_for_hrsc_MS(img_size=[512,640])# test size are 512, 544, 576, 608, 640
    else:
        print('ERROR OCCUR! --operation must be one of [visualize, DOTA_test, DOTA_MS_test, HRSC_test, HRSC_MS_test]')
