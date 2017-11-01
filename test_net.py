# -*- coding: utf-8 -*-
# @Time    : 2017/10/25 0025 11:10
# @Author  : xiangyaojun
# @Email   : maisca1920@163.com
# @File    : train_net.py
from lib.model.test import test_net
from lib.model.config import cfg, cfg_from_file, cfg_from_list
from lib.datasets.factory import get_imdb
from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.nets.mobilenet_v1 import mobilenetv1

import tensorflow as tf
import os
import pprint
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--net', dest='network_name', default="vgg16", help='The network of name: vgg16, res50, res101, res152, mobile')
  parser.add_argument('--test', dest='test_dataset_name', default="voc_2013_test", help='test data set of name')
  parser.add_argument('--train', dest='train_dataset_name', default="voc_2013_train", help='training data set of name')
  parser.add_argument('--iter', dest='iter_number', default="10000", help='test the iter of model')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  test_dataset_name = args.test_dataset_name#测试数据集名称
  set_cfgs = ['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[50000]']
  project_path = os.path.abspath('.')
  #测试迭代训练好的参数路径
  test_iter_weight_path = os.path.join(project_path, "output", args.network_name, args.train_dataset_name, args.network_name+"_faster_rcnn_iter_"+args.iter_number+".ckpt")

  cfg_from_file(project_path + "/experiments/cfgs/"+args.network_name+".yml")  # 载入参数配置
  cfg_from_list(set_cfgs)#修改参数配置
  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  filename = 'default/'+args.network_name

  imdb = get_imdb(test_dataset_name)

  #配置Session参数
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
  # load model
  net.create_architecture("TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  print(('Loading model check point from {:s}').format(test_iter_weight_path))
  saver = tf.train.Saver()
  saver.restore(sess, test_iter_weight_path)
  print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=1)

  sess.close()
