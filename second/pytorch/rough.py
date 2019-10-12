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

# def evaluate(config_path,
#              model_dir,
#              ckpt_path=None,
#              ref_detfile=None):
#     model_dir = pathlib.Path(model_dir)
    
#     config = pipeline_pb2.TrainEvalPipelineConfig()
#     with open(config_path, "r") as f:
#         proto_str = f.read()
#         text_format.Merge(proto_str, config)

#     input_cfg = config.eval_input_reader
#     model_cfg = config.model.second
#     train_cfg = config.train_config
#     class_names = list(input_cfg.class_names)
#     center_limit_range = model_cfg.post_center_limit_range
#     ######################
#     # BUILD VOXEL GENERATOR
#     ######################
#     voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
#     bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
#     box_coder = box_coder_builder.build(model_cfg.box_coder)
#     target_assigner_cfg = model_cfg.target_assigner
#     target_assigner = target_assigner_builder.build(target_assigner_cfg,
#                                                     bv_range, box_coder)

#     net = second_builder.build(model_cfg, voxel_generator, target_assigner)
#     net.cuda()
#     if train_cfg.enable_mixed_precision:
#         net.half()
#         net.metrics_to_float()
#         net.convert_norm_to_float(net)

#     if ckpt_path is None:
#         torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
#     else:
#         torchplus.train.restore(ckpt_path, net)

#     eval_dataset = input_reader_builder.build(
#         input_cfg,
#         model_cfg,
#         training=False,
#         voxel_generator=voxel_generator,
#         target_assigner=target_assigner)

#     eval_dataloader = torch.utils.data.DataLoader(
#         eval_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=input_cfg.num_workers,
#         pin_memory=False,
#         collate_fn=merge_second_batch)

#     if train_cfg.enable_mixed_precision:
#         float_dtype = torch.float16
#     else:
#         float_dtype = torch.float32

#     net.eval()
    
#     t = time.time()
#     dt_annos = []
#     global_set = None
#     print("Generate output labels...")
#     bar = ProgressBar()
#     bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

#     for i, example in enumerate(eval_dataloader):
#         if i == num_examples: break
#         example = example_convert_to_torch(example, float_dtype)
#         dt_annos += predict_kitti_to_anno(
#             net, example, class_names, center_limit_range,
#             model_cfg.lidar_input, global_set)
#         bar.print_bar()

#     sec_per_example = len(eval_dataset) / (time.time() - t)
#     print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

#     print(f"avg forward time per example: {net.avg_forward_time:.3f}")
#     print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")

#     gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
#     gt_annos = gt_annos[:num_examples]
    
#     result = get_official_eval_result(gt_annos, dt_annos, class_names)
#     print(result)
#     np.save('gt', gt_annos)
#     np.save('dt', dt_annos)

    

# if __name__ == "__main__":
# 	evaluate(config_path, model_dir)
