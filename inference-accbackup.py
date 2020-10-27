#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models#, losses, datasets
from utils import flow_utils, tools
import time

from scipy.misc import imread
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from fpn_reg import fpn_reg
from lib.model.fpn.fpn_resnet import FPNResNet
# fp32 copy of parameters for update
global param_copy
# import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default = 255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--milestones', default=[150,250,300], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')



    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments

    args = parser.parse_args()
    if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

    # Get argument defaults (hastag #thisisahack)
    parser.add_argument('--IGNORE',  action='store_true')
    defaults = vars(parser.parse_args(['--IGNORE']))

    # Print all arguments, color the non-defaults
    #for argument, value in sorted(vars(args).items()):
    #    reset = colorama.Style.RESET_ALL
    #    color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
    #    block.log('{}{}: {}{}'.format(color, argument, value, reset))

    args.model_class = tools.module_to_dict(models)[args.model]

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # dict to collect activation gradients (for training debug purpose)
    args.grads = {}

    if args.inference:
        args.skip_validation = True
        args.skip_training = True
        args.total_epochs = 1
        args.inference_dir = "{}/inference".format(args.save)

    print('Source Code')
    #print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments

    args.effective_batch_size = args.batch_size * args.number_gpus
    args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
    args.effective_number_workers = args.number_workers * args.number_gpus
    gpuargs = {'num_workers': args.effective_number_workers, 
               'pin_memory': True, 
               'drop_last' : True} if args.cuda else {}
    inf_gpuargs = gpuargs.copy()
    inf_gpuargs['num_workers'] = args.number_workers

    class ModelAndLoss(nn.Module):
        def __init__(self, args):
            super(ModelAndLoss, self).__init__()
            kwargs = tools.kwargs_from_args(args, 'model')
            self.model = args.model_class(args, **kwargs)
            
        def forward(self, inputs=None, rois=None, flow=None, mode='flow'):
            results = self.model(inputs=inputs, rois=rois, flow=flow, mode=mode)
            return results

    model_and_loss = ModelAndLoss(args)
    model_and_loss = torch.nn.DataParallel(model_and_loss).cuda()

    # Load weights if needed, otherwise randomly initialize
    if args.resume and os.path.isfile(args.resume):
        #block.log("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if not args.inference:
            args.start_epoch = 0
        pretrained_dict = checkpoint['state_dict']
        model_and_loss.module.model.load_state_dict(pretrained_dict)
    elif args.resume and args.inference:
        #block.log("No checkpoint found at '{}'".format(args.resume))
        quit()

    else:
        print('Random initialization')
        #block.log("Random initialization")


    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou

    def nms(boxes, scores, overlap=0.5, top_k=200):
        #print(boxes)
        #print(scores)
        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w*h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]
        #print(keep, count)
        return keep, count

    def IoU(box,boxes):
        '''
              计算detect box和 gt boxes的IoU值
              形参:
            box:numpy array,shape(5,):x1,y1,x2,y2,score
                input box
            boxes:numpy array,shape (n,4):x1,y1,x2,y2
                input ground truth boxes
                返回值：
             ovr: numpy.array, shape (n, )
             IoU
        '''
        box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        xx1=np.maximum(box[0],boxes[:,0])
        yy1=np.maximum(box[1],boxes[:,1])
        xx2=np.minimum(box[2],boxes[:,2])
        yy2=np.minimum(box[3],boxes[:,3])
        #print(area.dtype, yy2.dtype)
        #print((xx2-xx1+1).dtype)
        #print(torch.tensor(0.).type(torch.DoubleTensor).dtype)
        #计算 bounding box的长宽
        w=np.maximum(0.,xx2-xx1+1)
        h=np.maximum(0.,yy2-yy1+1)

        inter=w*h
        ovr= inter/(box_area+area-inter)
        return ovr#.detach().cpu().numpy()

    # Reusable function for inference
    def inference(image_dir, model_flow, FPN):
        thresh_score = 0.5#0.01
        thresh_overlap = 0.5
        thresh_overlap_uma = 0.5
        thresh_nms = 0.5#0.3
        stop = 30 #20 #30 #50
        step = 1
        variances = [0.1, 0.2]

        model_flow.eval()
        FPN.eval()

        imglist = os.listdir(image_dir)
        imglist.sort()
        num = len(imglist)-1

        # T = [] # track
        # t means time
        image_set = []
        tracks = 2000
        lens = 1600
        T = np.zeros((tracks, lens, 5))
        T.fill(-1)
        current_num = 0
        for t in range(num):
            print(t)
            im_file_1 = os.path.join(image_dir, imglist[t]) # eg. 000001.jpg
            im_file_2 = os.path.join(image_dir, imglist[t+1]) # eg. 000002.jpg
            # annotation use the same name as the first image and it contains the box-first box_next-second
            annotation = im_file_1.replace('img_input', 'annotation').replace('.jpg','.npy') # eg. 000001.npy
            annos = np.load(annotation,allow_pickle=True).tolist()
            det_1 = np.array(annos['bbox'])
            det_2 = np.array(annos['bbox_next'])

            img1 = imread(im_file_1)
            img2 = imread(im_file_2)
            

            images = [img1, img2]
            images = np.array(images).transpose(3,0,1,2)
            images = torch.from_numpy(images.astype(np.float32))
            images = images.unsqueeze(0).cuda() # for flownet2

            # get the image for fpn
            if t == 0:
                image_set.append(img1)

                with torch.no_grad():
                    #print(det_1.shape)
                    det_1 = fpn_reg(FPN, img1, det_1, thresh_score, thresh_nms)
                det_1 = det_1.detach().cpu().numpy()
                det_1 = np.insert(det_1[:,:4], 4, t, axis=1)
                num_det_1 = len(det_1)
                current_num = num_det_1
                T[:num_det_1,t,:] = det_1
                # for d in det_1:
                #     track = []
                #     track.append(d)
                #     T.append(track)
            image_set.append(img2)
            # a = input("input1:")
            # now deal with t+1 aka tracking
            if len(det_2) != 0:
                with torch.no_grad():
                    det_2 = fpn_reg(FPN, img2, det_2, thresh_score, thresh_nms) # refined detection for the second image
                    # a = input("input2:")
                    flow = model_flow(inputs=images, mode='flow') # this mode just to generate flow from t to t+1
                det_2 = det_2.detach().cpu().numpy()
            else:
                print('no detection')

            # B = []
            # ID = []
            # a = input("input3:")
            b_lasts_ = T[:, t, :] # get the bounding box at time t
            b_idx = np.where(b_lasts_[:,4] == t)[0]
            b_lasts = b_lasts_[b_idx, :4] # n x 4

            rois = torch.tensor(b_lasts.astype(np.float32)).unsqueeze(0)
            rois = rois.cuda()
            with torch.no_grad():
                offset_pred = model_flow(rois=rois, flow=flow, mode='reg')
                print(offset_pred.shape)
                # a = input("input4:")
            # offset_pred = offset_pred.squeeze(0).detach().cpu().numpy()
            offset_pred = offset_pred.detach().cpu().numpy()

            b_last = rois.squeeze(0).detach().cpu().numpy()
            assert offset_pred.shape == b_last.shape

            xc_l = np.reshape((b_last[:,0]+b_last[:,2])/2., (-1,1))
            yc_l = np.reshape((b_last[:,1]+b_last[:,3])/2., (-1,1))
            w_l = np.reshape(b_last[:,2]-b_last[:,0], (-1,1))
            h_l = np.reshape(b_last[:,3]-b_last[:,1], (-1,1))
            b_last_ = np.concatenate((xc_l, yc_l, w_l, h_l), 1)

            b_pred = np.concatenate((b_last_[:, :2] + offset_pred[:, :2] * variances[0] * b_last_[:, 2:],
                b_last_[:, 2:] * np.exp(offset_pred[:, 2:] * variances[1])),1)

            b_pred[:, :2] -= b_pred[:, 2:] / 2
            with torch.no_grad():
                print(b_pred.shape)
                # b_pred = np.array([b_pred])
                b_pred, inds = fpn_reg(FPN, img2, b_pred, thresh_score, thresh_nms, True)
                # a = input("input5:")
                print(b_pred.shape)
                print(inds)
            inds = inds.detach().cpu().numpy()
            b_idx = b_idx[inds]
            b_pred = b_pred.squeeze(0).detach().cpu().numpy()
            print('b',b_pred.shape)
            for k_p, b_p in enumerate(b_pred):
                print(b_p.shape, det_2.shape)
                ovlp = IoU(b_p, det_2)
                id_ovlp = np.argmax(ovlp)
                assert ovlp[id_ovlp] == np.max(ovlp)
                if np.max(ovlp) > thresh_overlap:
                    if det_2[id_ovlp][4] > b_p[4]:
                        b_pred[k_p] = det_2[id_ovlp]
                    print('ov',id_ovlp)
                    print(det_2.shape)
                    det_2 = np.delete(det_2, id_ovlp, 0)
                    print(det_2.shape)
                    print('de2',len(det_2))
                if len(det_2) == 0:
                    break
                    # del det_2[id_ovlp]

            if len(b_pred) != 0:
                ids, count = nms(torch.tensor(b_pred[:,:4]), torch.tensor(b_pred[:,4]), overlap=thresh_nms)
                idx_nms = ids[:count].numpy().tolist()
                idx_nms.sort()
                b_pred = b_pred[idx_nms]
                b_idx = b_idx[idx_nms]

                b_pred = np.insert(b_pred[:,:4], 4, t+1, axis=1)
                T[b_idx,t+1,:] = b_pred

            # det_2 is det_unmatched
            if len(det_2) != 0:
                all_img = []
                inact_idx = np.where(T[:current_num, t+1, 4]!=t+1)[0]
                inact_T = T[inact_idx, :, :]

                if t == 0:
                    det_2_unm = np.insert(det_2[:,:4], 4, t+1, axis=1)
                    num_um = len(det_2)
                    T[current_num:current_num+num_um,t+1,:] = det_2_unm
                    current_num = current_num + num_um
                    continue

                if t < stop-1:
                    rois_BT = np.zeros((t, 200, 4))
                else:
                    rois_BT = np.zeros((stop-1, 200, 4))

                rois_BT.fill(-1)
                # mark = []
                rois_BT_ = []
                idx_ori = []
                flag = 0
                for v in range(1,stop,step): # stop is the furthest to go, and step is interval
                    if v > t:
                        break

                    v_idx = np.where(inact_T[:,t-v,4]==t-v)[0]
                    print('v',v)
                    print(v_idx)
                    print(len(v_idx))
                    if len(v_idx) == 0:
                        images_ = np.array([image_set[t-v],image_set[t+1]]).transpose(3,0,1,2)
                        all_img.append(images_)
                        continue
                    else:
                        flag = 1
                        print('ok')
                    assert len(v_idx) <= 200

                    idx_ori = idx_ori + inact_idx[v_idx].tolist() #track

                    v_T = inact_T[v_idx, t-v, :4] # n x 4

                    rois_BT[v-1,:len(v_idx),:] = v_T
                    print(v_T)

                    # mark = mark + [v-1]*len(v_idx) #time t

                    rois_BT_.append(v_T)

                    images_ = np.array([image_set[t-v],image_set[t+1]]).transpose(3,0,1,2)
                    all_img.append(images_)

                if flag == 0:
                    det_2_unm = np.insert(det_2[:,:4], 4, t+1, axis=1)
                    num_um = len(det_2)
                    T[current_num:current_num+num_um,t+1,:] = det_2_unm
                    current_num = current_num + num_um
                    continue

                # mark = np.array(mark)
                idx_ori = np.array(idx_ori)
                
                all_img = np.array(all_img)
                all_img = torch.from_numpy(all_img.astype(np.float32))
                all_img = all_img.cuda()

                rois_BT = torch.tensor(rois_BT.astype(np.float32))
                rois_BT = rois_BT.cuda()
                with torch.no_grad():
                    print(all_img.shape)
                    print(rois_BT.shape)
                    offset_pred_ = model_flow(inputs=all_img, rois=rois_BT , mode='full')
                    print(offset_pred_.shape)
                offset_pred_ = offset_pred_.detach().cpu().numpy()
                rois_BT_ = np.vstack(rois_BT_)
                # assert len(rois_BT_) == len(mark) and len(offset_pred_) == len(mark) and len(mark) == len(idx_ori)
                assert len(rois_BT_) == len(offset_pred_) and len(offset_pred_) == len(idx_ori)

                xc_l_ = np.reshape((rois_BT_[:,0]+rois_BT_[:,2])/2., (-1,1))
                yc_l_ = np.reshape((rois_BT_[:,1]+rois_BT_[:,3])/2., (-1,1))
                w_l_ = np.reshape(rois_BT_[:,2]-rois_BT_[:,0], (-1,1))
                h_l_ = np.reshape(rois_BT_[:,3]-rois_BT_[:,1], (-1,1))
                b_last_BT = np.concatenate((xc_l_, yc_l_, w_l_, h_l_), 1)

                b_pred_BT = np.concatenate((b_last_BT[:, :2] + offset_pred_[:, :2] * variances[0] * b_last_BT[:, 2:],
                    b_last_BT[:, 2:] * np.exp(offset_pred_[:, 2:] * variances[1])),1)

                b_pred_BT[:, :2] -= b_pred_BT[:, 2:] / 2
                with torch.no_grad():
                    print(b_pred_BT.shape)
                    b_pred_BT, inds_BT = fpn_reg(FPN, img2, b_pred_BT, thresh_score, thresh_nms, True)

                # mark = mark[inds_BT]
                print(b_pred_BT.shape)
                inds_BT = inds_BT.detach().cpu().numpy()
                print(inds_BT)
                print(idx_ori)
                idx_ori = idx_ori[inds_BT]
                print(idx_ori.shape)
                # b_pred_BT = b_pred_BT.squeeze(0).detach().cpu().numpy()
                b_pred_BT = b_pred_BT.detach().cpu().numpy()
                assert len(b_pred_BT) == len(idx_ori)

                for k_d, d in enumerate(det_2):
                    ovlp_d = IoU(d, b_pred_BT)
                    idx_m = np.searchsorted(ovlp_d, thresh_overlap, side='right')
                    #这里还是先总从原版逻辑，但是之后还是可以优化的，比如多帧的匹配，如果一个track上最多帧能和这个det配上才选这个track，而不是现在选最近一个time上能配上的最近的track
                    if idx_m != len(ovlp_d):
                        index = idx_ori[idx_m] # for track
                        if d[4] > b_pred_BT[idx_m][4]:
                            temp = d[:4].tolist()
                            T[index,t+1,:] = np.array(temp.append(t+1))
                        else:
                            temp = b_pred_BT[idx_m][:4].tolist()
                            T[index,t+1,:] = np.array(temp.append(t+1))
                        # delete track from options
                        del_idx = np.where(idx_ori == index)[0]
                        idx_ori = np.delete(idx_ori, del_idx)
                        b_pred_BT = np.delete(b_pred_BT, del_idx, 0)
                    else:
                        #not exist
                        temp = d[:4].tolist()
                        T[current_num,t+1,:] = np.array(temp.append(t+1))
                        current_num = current_num + 1

        np.save('tracking_result.npy', T)

        return

    set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    cfg_file = "cfgs/res101_fpn.yml"

    cfg_from_file(cfg_file)
    cfg_from_list(set_cfgs)

    np.random.seed(cfg.RNG_SEED)
    load_name = 'models/faster_rcnn_1_20_74.pth'
    # load_name = 'pretrained/fpn_1_27.pth'
    classes = ('__background__', 'person')
    # initilize the network here.
    FPN = FPNResNet(classes, 101, pretrained=False)
    FPN.create_architecture()
    checkpoint = torch.load(load_name)
    FPN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    image_dir = '/data/OPT4/MOT17/test_flowtrack/MOT17-01-DPM/img_input'
    inference(image_dir, model_and_loss, FPN)






