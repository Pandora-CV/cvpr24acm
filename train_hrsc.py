from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
import random
import os
import yaml
import collections
import argparse
import sys
sys.path.append('DOTA_devkit')
from tqdm import tqdm

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp


from nets.resnet_dcn_DFPN_model import ResNet
from datasets.HRSCDataset import HRSCSetv1,collater



parser = argparse.ArgumentParser()

parser.add_argument("--input_size", default = 640 , type=int)
parser.add_argument("--datadir", type=str, default='/home/data/xuhang/datasets/HRSC2016/train')
#parser.add_argument("--heads", default={'hm': 1,'wh': 2 ,'reg': 2, 'theta':4})
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--coder", default='acm')
parser.add_argument("--coder_cfg", default=1, type=int)
parser.add_argument("--coder_mode", default='model')
parser.add_argument("--box_loss", default='kfiou')
parser.add_argument("--model", type=int, default=50)

parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument('--seed', default=2021, type=int,help='random seed')



parser.add_argument("--epochs", type=int, default=140)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--batch_size", type=int, default = 32, help="size of each image batch")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lr_decay",  default=[100,130])
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--resume_weight_path', default="")
parser.add_argument("--save_interval", type=int, default=70)
parser.add_argument('--log_path', default="./result/debug.txt")

parser.add_argument('--dist-url', default='tcp://127.0.0.1:2556', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--nodes', default=1, type=int,
                    help='total number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='total number of process for distributed training')
parser.add_argument('--local_rank', default=0, type=int)

def cal(epoch):
    if epoch < 10:
        return 0.0
    else:
        return 1.0

def main():
    args = parser.parse_args()
    print(args)

    """随机数种子"""
    random.seed(args.seed)
    #np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True


    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.nodes

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):

    #torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(args.seed)
    np.random.seed(args.seed+gpu)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)  
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    train_loss = collections.deque(maxlen=10)

    print("=> creating model.")
    #print(args.heads)

    model = ResNet(num_layers=args.model, num_cls=args.heads, coder=args.coder, coder_cfg=args.coder_cfg, coder_mode=args.coder_mode, box_loss=args.box_loss).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)


    if args.resume:
        model.load_state_dict(torch.load(args.resume_weight_path,map_location={"cuda:0":"cuda:{}".format(args.gpu)}))
        print("==>finished loading weight")

    cudnn.benchmark = True

    print("=> preparing data")
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    trainset = HRSCSetv1(root_dir=args.datadir,img_size=args.input_size)
    print("training images: {}".format(len(trainset)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),collate_fn=collater,num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    if args.rank == 0 : f = open(args.log_path, 'w')

    for epoch in range(args.start_epoch,args.epochs) :

        model.train()
        train_sampler.set_epoch(epoch)


        for batch_idx, data in enumerate(train_loader):
            img,label,heatmap_t,theta=data['img'].cuda(args.gpu, non_blocking=True),data['label'],data['heatmap_t'].cuda(args.gpu, non_blocking=True),data['theta']


            if 1:
                center_loss, scale_loss, offset_loss, theta_loss = model({'img':img , 'label':label , 'heatmap_t':heatmap_t,'theta':theta})
                total_loss = center_loss + scale_loss + offset_loss + theta_loss

                optimizer.zero_grad()
                total_loss.backward()
                for p in model.parameters():
                    torch.nn.utils.clip_grad_norm_(p,10)
                optimizer.step()

                train_loss.append(float(total_loss))
                if args.rank==0:
                    print(
                        '{}\{} | Center loss: {:1.5f} | scale loss: {:1.5f} | offset loss: {:1.5f}| theta loss:{:1.5f} | running loss: {:1.5f}'.format(
                            epoch, batch_idx, float(center_loss), float(scale_loss), float(offset_loss),float(theta_loss), np.mean(train_loss))
                    )

                    f.write(str(float(center_loss))) , f.write(" ") , f.write(str(float(scale_loss))) , f.write(" ") , f.write(str(float(offset_loss))), f.write(" ") ,f.write(str(float(theta_loss))), f.write(" "), f.write(str(float(np.mean(train_loss))))
                    f.write('\n')



        if (epoch+1) in args.lr_decay :
            args.lr = args.lr/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        if args.rank == 0 and (epoch + 1) % args.save_interval == 0:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if not os.path.isdir('checkpoint/hrsc'):
                os.mkdir('checkpoint/hrsc')
            print("Saving...")
            torch.save(model.state_dict(), f"checkpoint/hrsc/{args.box_loss}_{args.coder}_{args.coder_cfg}_{args.coder_mode}_mon_%d.pth" % (epoch+1))

    if args.rank == 0 : f.close()





if __name__ == "__main__":
    main()


