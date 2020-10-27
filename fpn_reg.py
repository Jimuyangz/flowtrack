# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.utils.blob import im_list_to_blob

from lib.model.fpn.fpn_resnet import FPNResNet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


# def parse_args():
#   """
#   Parse input arguments
#   """
#   parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
#   parser.add_argument('--dataset', dest='dataset',
#                       help='training dataset',
#                       default='pascal_voc', type=str)
#   parser.add_argument('--cfg', dest='cfg_file',
#                       help='optional config file',
#                       default='cfgs/vgg16.yml', type=str)
#   parser.add_argument('--net', dest='net',
#                       help='vgg16, res50, res101, res152',
#                       default='res101', type=str)
#   parser.add_argument('--set', dest='set_cfgs',
#                       help='set config keys', default=None,
#                       nargs=argparse.REMAINDER)
#   parser.add_argument('--load_dir', dest='load_dir',
#                       help='directory to load models',
#                       default="/srv/share/jyang375/models")
#   parser.add_argument('--image_dir', dest='image_dir',
#                       help='directory to load images for demo',
#                       default="images")
#   parser.add_argument('--cuda', dest='cuda',
#                       help='whether use CUDA',
#                       action='store_true')
#   parser.add_argument('--mGPUs', dest='mGPUs',
#                       help='whether use multiple GPUs',
#                       action='store_true')
#   parser.add_argument('--cag', dest='class_agnostic',
#                       help='whether perform class_agnostic bbox regression',
#                       action='store_true')
#   parser.add_argument('--parallel_type', dest='parallel_type',
#                       help='which part of model to parallel, 0: all, 1: model before roi pooling',
#                       default=0, type=int)
#   parser.add_argument('--checksession', dest='checksession',
#                       help='checksession to load model',
#                       default=1, type=int)
#   parser.add_argument('--checkepoch', dest='checkepoch',
#                       help='checkepoch to load network',
#                       default=1, type=int)
#   parser.add_argument('--checkpoint', dest='checkpoint',
#                       help='checkpoint to load network',
#                       default=10021, type=int)
#   parser.add_argument('--bs', dest='batch_size',
#                       help='batch_size',
#                       default=1, type=int)
#   parser.add_argument('--vis', dest='vis',
#                       help='visualization mode',
#                       action='store_true')

#   args = parser.parse_args()
#   return args

# lr = cfg.TRAIN.LEARNING_RATE
# momentum = cfg.TRAIN.MOMENTUM
# weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def fpn_reg(FPN, img, rois_gt, thresh_score, thresh_nms, flag=False):
  # input:FPN, img, rois_gt(x1,y1,w,h), thresh_score

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  im_data = im_data.cuda()
  im_info = im_info.cuda()
  num_boxes = num_boxes.cuda()
  gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  FPN.cuda()

  FPN.eval()
  max_per_image = 100
  #thresh = 0.05
  classes = ('__background__', 'person')
  # Set up webcam or get image directories
  im_in = np.array(img)
  # rgb -> bgr
  im = im_in[:,:,::-1]

  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  # label_file = os.path.join(args.image_dir, imglist[num_images].replace('.jpg','.npy')).replace('images','label')
  # rois_gt = np.load(label_file)
  rois_gt[:,2] = rois_gt[:,0]+rois_gt[:,2]
  rois_gt[:,3] = rois_gt[:,1]+rois_gt[:,3]

  ratio_w = 1500./1920.
  ratio_h = 800./1024.

  rois_gt[:,0] = rois_gt[:,0]*ratio_w
  rois_gt[:,1] = rois_gt[:,1]*ratio_h
  rois_gt[:,2] = rois_gt[:,2]*ratio_w
  rois_gt[:,3] = rois_gt[:,3]*ratio_h

  _z = np.zeros((rois_gt.shape[0],1))
  rois_gt = torch.tensor(np.append(_z,rois_gt,axis=1)).type(torch.float32).cuda()

  with torch.no_grad():
      im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.resize_(1, 1, 5).zero_()
      num_boxes.resize_(1).zero_()

  rois, cls_prob, bbox_pred, \
  rpn_loss_cls, rpn_loss_box, \
  RCNN_loss_cls, RCNN_loss_bbox, \
  rois_label = FPN(im_data, im_info, rois_gt, gt_boxes, num_boxes)

  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]

  if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Optionally normalize targets by a precomputed mean and stdev
          #print('here')
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

          box_deltas = box_deltas.view(1, -1, 4 * len(classes))
      #print(box_deltas.shape)
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

  pred_boxes /= im_scales[0]

  scores = scores.squeeze()
  pred_boxes = pred_boxes.squeeze()
  #det_toc = time.time()
  #detect_time = det_toc - det_tic
  #misc_tic = time.time()

  # for j in xrange(1, len(pascal_classes)):
  #print(len(scores.shape))
  #print(pred_boxes.unsqueeze(0).shape)
  #print(scores[:,1])
  if len(scores.shape) == 1:
    scores = scores.unsqueeze(0)
    pred_boxes = pred_boxes.unsqueeze(0)
  #print(scores.shape)
  j = 1
  inds = torch.nonzero(scores[:,1]>thresh_score).view(-1)
  # print(inds)
  # if there is det
  if inds.numel() > 0:
    cls_scores = scores[:,1][inds]
    _, order = torch.sort(cls_scores, 0, True)
    
    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
    
    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
    if flag == True:
      # print('???',cls_dets, inds)
      return cls_dets, inds
    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
    cls_dets = cls_dets[order]
    # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)

    keep = nms(cls_boxes[order, :], cls_scores[order], thresh_nms)
    cls_dets = cls_dets[keep.view(-1).long()]
    #print(cls_dets.shape)

    return cls_dets # num x 5 (x1,y1,x2,y2,score)

  else:
    if flag == True:
      return torch.tensor([]).cuda(), torch.tensor([]).cuda()
    else:
      return torch.tensor([]).cuda()




