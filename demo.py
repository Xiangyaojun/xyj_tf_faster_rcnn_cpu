# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by YaoJun Xiang, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.config import cfg
from lib.model.test import im_detect
from lib.model.nms_wrapper import nms
from lib.utils.timer import Timer
from pylab import *

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2, argparse

from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
from lib.nets.mobilenet_v1 import mobilenetv1

CLASSES = ('__background__',  # always index 0
                     "北汽", "福特", "斯柯达", "启辰", "本田", "日产", "凯迪拉克", "铃木",
                     "吉利", "保时捷", "jeep", "宝骏", "荣威", "林肯", "丰田", "别克",
                     "奇瑞", "起亚", "哈弗", "奥迪", "路虎", "大众", "广汽传祺", "长安",
                     "名爵", "雷诺", "雷克萨斯", "宝马", "马自达", "奔驰")

def parse_args():
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--net', dest='network_name', default="vgg16", help='The network of name: vgg16, res50, res101, res152, mobile')
  parser.add_argument('--train', dest='train_dataset_name', default="voc_2013_train", help='training data set of name')
  parser.add_argument('--iter', dest='iter_number', default="30000", help='test the iter of model')
  args = parser.parse_args()
  return args

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # plt.draw()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    project_path = os.path.abspath('.')
    save_path = os.path.join(project_path, 'data', "demo",image_name[:len(image_name)-4]+"_detect" + ".jpg")
    plt.savefig(save_path)

if __name__ == '__main__':
    args = parse_args()
    project_path = os.path.abspath('.')
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # 载入中文字体
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # model path
    tfmodel = os.path.join(project_path, "output", args.network_name, args.train_dataset_name,
                                         args.network_name + "_faster_rcnn_iter_" + args.iter_number + ".ckpt")


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.network_name == 'vgg16':
        net = vgg16()
    elif args.network_name == 'res50':
        net = resnetv1(num_layers=50)
    elif args.network_name == 'res101':
        net = resnetv1(num_layers=101)
    elif args.network_name == 'res152':
        net = resnetv1(num_layers=152)
    elif args.network_name == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)


    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['0001.jpg', '0002.jpg', '0003.jpg',
                '0004.jpg', '0005.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
