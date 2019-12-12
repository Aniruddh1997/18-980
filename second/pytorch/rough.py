import pickle
import pathlib
from google.protobuf import text_format
import numpy as np
import torch
import time
import torchplus

import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar
from train import example_convert_to_torch, predict_kitti_to_anno

import warnings
warnings.filterwarnings('ignore')

info_path = '/home/scratch/anirudds/data/kitti_infos_train.pkl'
with open(info_path, 'rb') as f:
    kitti_infos_train = pickle.load(f)

info_path = '/home/scratch/anirudds/data/kitti_infos_test.pkl'
with open(info_path, 'rb') as f:
    kitti_infos_test = pickle.load(f)


print (kitti_infos_train[0]['annos'])

config_path = 'second/configs/pointpillars/car/xyres_16.proto'
model_dir = '/zfsauton2/home/anirudds/model_base/'
num_examples = 50

gt_annos = np.load('gt.npy')
dt_annos = np.load('dt.npy')

print (dt_annos[0])
