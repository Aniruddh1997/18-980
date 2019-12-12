import time
from enum import Enum
from functools import reduce

import numpy as np
# import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from second.pytorch.utils import get_paddings_indicator, get_alpha, _sigmoid

from second.pytorch.models.centernet import get_pose_net
from second.pytorch.models.losses import FocalLoss, L1Loss, BinRotLoss
from second.pytorch.models.decode import ddd_decode


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(
            0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0)


class MiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm3d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            # BatchNorm3d = change_default_args(
            #     group=32, eps=1e-3, momentum=0.01)(GroupBatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape
        self.middle_conv = Sequential(
            ZeroPad3d(1),
            Conv3d(num_input_features, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d([1, 1, 1, 1, 0, 0]),
            Conv3d(64, 64, 3, stride=1),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d(1),
            Conv3d(64, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        # print('scatter_nd fw:', time.time() - t)
        ret = ret.permute(0, 4, 1, 2, 3)
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="MiddleExtractor",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='voxelnet', K=100,
                 hm_weight = 1,
                 dim_weight = 1,
                 rot_weight = 1,
                 off_weight = 1,
                 centernet_layers = 50):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        
        self.K = K
        self.hm_weight = hm_weight
        self.dim_weight = dim_weight
        self.rot_weight = rot_weight
        self.off_weight = off_weight
        self.centernet_layers = centernet_layers;
        self.pc_range = pc_range


        vfe_class_dict = {
            "VoxelFeatureExtractor": VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": VoxelFeatureExtractorV2,
            "PillarFeatureNet": PillarFeatureNet
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        if vfe_class_name == "PillarFeatureNet":
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance,
                voxel_size=voxel_size,
                pc_range=pc_range
            )
        else:
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance)

        print("middle_class_name", middle_class_name)
        if middle_class_name == "PointPillarsScatter":
            self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                                num_input_features=vfe_num_filters[-1])
        else:
            mid_class_dict = {
                "MiddleExtractor": MiddleExtractor
                # "SparseMiddleExtractor": SparseMiddleExtractor,
            }
            mid_class = mid_class_dict[middle_class_name]
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=vfe_num_filters[-1],
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            

        #======================= New detection heads ==========================
        heads = {'hm': self._num_class, 'rot': 8, 'dim': 3, 'reg': 2}
        self.centernet = get_pose_net(num_layers=self.centernet_layers, heads=heads)
        #=======================================================================

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.hm_loss = metrics.Scalar()
        self.dim_loss = metrics.Scalar()
        self.rot_loss = metrics.Scalar()
        self.total_loss = metrics.Scalar()

        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        t = time.time()
        batch_size = example["image_idx"].shape[0]

        # VFE + Pointpillar Scatter + Keypoint detection network
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size)
        outputs = self.centernet(spatial_features)

        self._total_forward_time += time.time() - t
        if self.training:
            return self.get_loss(example, outputs)
        else:
            return self.predict(example, outputs[-1], self.pc_range)

    def get_loss(self, example, outputs):
        crit = torch.nn.MSELoss() #or FocalLoss()
        crit_reg = L1Loss()
        crit_rot = BinRotLoss()

        hm_loss, rot_loss, dim_loss = 0, 0, 0
        wh_loss, off_loss = 0, 0
        num_stacks = 1 #this is 2 only for hourglass backbone
        for s in range(num_stacks):
            output = outputs[s]
            # print ("output ", output['hm'].size())
            # print (output['hm'])
            # print ("example ", example['hm'].size())
            # print (example['hm'])
            output['hm'] = _sigmoid(output['hm'])

            hm_loss += crit(output['hm'], example['hm']) / num_stacks
            if self.dim_weight > 0:
                dim_loss += crit_reg(output['dim'], example['reg_mask'],
                              example['ind'], example['dim']) / num_stacks
            if self.rot_weight > 0:
                rot_loss += crit_rot(output['rot'], example['rot_mask'],
                              example['ind'], example['rotbin'],
                              example['rotres']) / num_stacks
            if self.off_weight > 0:
                off_loss += crit_reg(output['reg'], example['rot_mask'],
                              example['ind'], example['reg']) / num_stacks

        loss = self.hm_weight * hm_loss + \
        self.dim_weight * dim_loss + self.rot_weight * rot_loss + \
        self.off_weight * off_loss

        return {'loss': loss, 'hm_loss': hm_loss,
          'dim_loss': dim_loss, 'rot_loss': rot_loss, 
          # 'wh_loss': wh_loss, 
          'off_loss': off_loss}

    def predict(self, example, output, pc_range):
        t = time.time()
        dets = ddd_decode(output['hm'], output['rot'], output['dim'],
                            pc_range, example["ground"], reg=output['reg'], K=self.K)

        batch_size = example['rect'].shape[0]
        self._total_inference_count += batch_size

        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        batch_imgidx = example['image_idx']

        predictions_dicts = []
        for det, rect, Trv2c, P2, img_idx in zip(dets, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx):

            final_alpha = get_alpha(det[:,6:14])
            final_rot_y = final_alpha + torch.atan2(-det[:,1], det[:,0])
            final_box_preds = torch.cat([det[:,:6], final_rot_y.unsqueeze(1)], dim = -1)

            final_scores = det[:, -2]
            final_labels = det[:, -1]

            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            # change angles
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)
            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                "label_preds": final_labels,
                "image_idx": img_idx,
            }
            predictions_dicts.append(predictions_dict)

        self._total_postprocess_time += time.time() - t
        return predictions_dicts

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.hm_loss.float()
        self.dim_loss.float()
        self.rot_loss.float()
        self.total_loss.float()

    def update_metrics(self,
                       hm_loss,
                       dim_loss,
                       rot_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        hm_loss_ = self.hm_loss(hm_loss).numpy()[0]
        dim_loss_ = self.dim_loss(dim_loss).numpy()[0]
        rot_loss_ = self.rot_loss(rot_loss).numpy()[0]
        ret = {
            "hm_loss": float(hm_loss_),
            "hm_loss_rt": float(hm_loss.data.cpu().numpy()),
            'dim_loss': float(dim_loss_),
            "dim_loss_rt": float(dim_loss.data.cpu().numpy()),
            "rot_loss": float(rot_loss_),
            "rot_loss_rt": float(rot_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc)
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.hm_loss.clear()
        self.dim_loss.clear()
        self.rot_loss.clear()
        self.total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(net)
        return net