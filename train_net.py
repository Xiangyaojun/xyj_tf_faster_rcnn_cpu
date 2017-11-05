# -*- coding: utf-8 -*-
# @Time    : 2017/10/25 0025 11:10
# @Author  : xiangyaojun
# @Email   : maisca1920@163.com
# @File    : train_net.py
from lib.model.train_val import get_training_roidb, train_net
from lib.model.config import cfg_from_list, cfg_from_file, get_output_dir, get_output_tb_dir, cfg
from lib.datasets.factory import get_imdb
from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.nets.mobilenet_v1 import mobilenetv1
import lib.datasets.imdb as datasets_imdb
import os
import pprint
import argparse

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets_imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb
def parse_args():
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--net', dest='network_name', default="vgg16", help='The network of name: vgg16, res50, res101, res152, mobile')
  parser.add_argument('--train', dest='train_dataset_name', default="voc_2013_train", help='training data set of name')
  parser.add_argument('--val', dest='val_dataset_name', default="voc_2013_train", help='validation data set of name')
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  args = parser.parse_args()
  return args
if __name__ == '__main__':
  args = parse_args()
  arg_net = args.network_name
  train_datset_name = args.train_dataset_name  # 测试数据集名称
  val_dataset_name = args.val_dataset_name  # 验证数据集名称
  set_cfgs = ['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[50000]',]
  project_path = os.path.abspath('.')
  pre_train_weight = project_path+"/data/pre_train_weight/"+arg_net+".ckpt"

  cfg_from_file(project_path+"/experiments/cfgs/"+arg_net+".yml")#载入参数配置
  cfg_from_list(set_cfgs)#修改参数配置
  print('Using config:')
  pprint.pprint(cfg)

  # roidb：所有训练图片的gt_boxes
  # imdb：训练数据集的相关信息：包括类别列表，所有的图片名称的索引，数据集名称等等
  imdb, roidb = combined_roidb("gridsum_car_train")
  print(roidb[0]['boxes'])
  print(roidb[0])
  print('{:d} roidb entries'.format(len(roidb)))
  # output directory where the models are saved
  output_dir = get_output_dir(imdb, "")
  print('Output will be saved to `{:s}`'.format(output_dir))
  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, "")
  # 同样的方法载入val数据集
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))
  _, valroidb = combined_roidb("gridsum_car_train")
  print('{:d} validation roidb entries'.format(len(valroidb)))

  if arg_net == 'vgg16':
    net = vgg16()
  elif arg_net == 'res50':
    net = resnetv1(num_layers=50)
  elif arg_net == 'res101':
    net = resnetv1(num_layers=101)
  elif arg_net == 'res152':
    net = resnetv1(num_layers=152)
  elif arg_net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError

  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=pre_train_weight,
            max_iters=args.max_iters)