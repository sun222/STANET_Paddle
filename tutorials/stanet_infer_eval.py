import sys
#加入环境
sys.path.append('./STANET_Paddle/')
import os
import cv2
import argparse
import paddlers as pdrs
import paddle
import numpy as np
import tqdm
import paddlers
from paddlers import transforms as T
from paddlers.transforms import arrange_transforms
from paddlers.utils import (seconds_to_hms, get_single_card_bs, dict2str,
                            get_pretrain_weights, load_pretrain_weights,
                            load_checkpoint, SmoothedValue, TrainingStats,
                            _get_shared_memory_size_in_M, EarlyStop)
                            
from paddle.io import DataLoader, DistributedBatchSampler
from collections import OrderedDict
import paddlers.models.ppseg as paddleseg

from paddlers.tasks.utils import seg_metrics as metrics
import paddlers.utils.logging as logging


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infer_dir',
        '-m',
        type=str,
        default=None,
        help='model directory path')
    parser.add_argument(
        '--img_dir',
        '-s',
        type=str,
        default=None,
        help='path to save inference model')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./tutorials/infer/output',
        help='path to save inference result')
    parser.add_argument(
        '--warmup_iters', type=int, default=0, help='warmup_iters')

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser.add_argument('--repeats', type=int, default=1, help='repeats')

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser


def quantize(arr):
    return (arr * 255).astype('uint8')

def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = paddle.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = paddle.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = paddle.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = paddle.logical_and(pred_i, label_i)
        pred_area.append(paddle.sum(paddle.cast(pred_i, "int32")))
        label_area.append(paddle.sum(paddle.cast(label_i, "int32")))
        intersect_area.append(paddle.sum(paddle.cast(intersect_i, "int32")))

    pred_area = paddle.concat(pred_area)
    label_area = paddle.concat(label_area)
    intersect_area = paddle.concat(intersect_area)

    return intersect_area, pred_area, label_area


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()


    batch_size=1
    DATA_DIR =     args.img_dir
    # "./dataset/"


    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR,'val.txt')


    eval_transforms = T.Compose([
        T.Resize(target_size=256),
        T.Normalize(
          mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR+'/val',
        file_list=EVAL_FILE_LIST_PATH,
        label_list=None,
        transforms=eval_transforms,
        num_workers=0,
        binarize_labels=True,
        shuffle=False)

    predictor = pdrs.deploy.Predictor(args.infer_dir, use_gpu=args.use_gpu)

    T.arrange_transforms(
            model_type='changedetector',
            transforms=eval_dataset.transforms,
            mode='eval')

    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not (paddle.distributed.parallel.parallel_helper.
                _is_parallel_ctx_initialized()):
            paddle.distributed.init_parallel_env()

    batch_size_each_card = get_single_card_bs(batch_size)
    if batch_size_each_card > 1:
        batch_size_each_card = 1
        batch_size = batch_size_each_card * paddlers.env_info['num']
        logging.warning(
            "Segmenter only supports batch_size=1 for each gpu/cpu card " \
            "during evaluation, so batch_size " \
            "is forcibly set to {}.".format(batch_size)
        )
    # predictor._model
    eval_data_loader = predictor._model.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    conf_mat_all = []
    with paddle.no_grad():
        for step, data in enumerate(eval_data_loader):
            # print(step)
            data.append(eval_dataset.transforms.transforms)
            net=predictor._model.net
            mode='eval'
             
            preprocessed_samples = {
                'image': np.asarray(data[0]),
                'image2':  np.asarray(data[1]),
                'ori_shape':  np.asarray(data[2])
            }
           
            net_outputs = predictor.raw_predict(preprocessed_samples)

            # label_map, score_map = predictor._model._postprocess(
            #     net_outputs,
            #     batch_origin_shape=ori_shape,
            #     transforms=transforms.transforms)
         

            # net_out = net(inputs[0], inputs[1])
            # logit = net_out[0]
            outputs = OrderedDict()
            self=predictor._model


            # if self.status == 'Infer':
            #     pred = paddle.unsqueeze(net_out[0], axis=1)  # NCHW
            # else:
            #     pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            label = data[2]
            # print(label.shape)

            origin_shape = [label.shape[-2:]]
            pred = self._postprocess(
                net_outputs, origin_shape, transforms=data[3])[0][0]  # NCHW

            pred=paddle.to_tensor(pred)
            pred = paddle.unsqueeze(pred, axis=[0, 1])
 
            intersect_area, pred_area, label_area = calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
            outputs['conf_mat'] = metrics.confusion_matrix(pred, label,
                                                           self.num_classes)
       
       
  

            pred_area = outputs['pred_area']
            label_area = outputs['label_area']
            intersect_area = outputs['intersect_area']
            conf_mat = outputs['conf_mat']

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                conf_mat_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                                  intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)
                paddle.distributed.all_gather(conf_mat_list, conf_mat)

                # Some image has been evaluated and should be eliminated in last iter
                if (step + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - step * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]
                    conf_mat_list = conf_mat_list[:valid]

                    intersect_area_all += sum(intersect_area_list)
                    pred_area_all += sum(pred_area_list)
                    label_area_all += sum(label_area_list)
                    conf_mat_all.extend(conf_mat_list)

            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
                conf_mat_all.append(conf_mat)
        # class_iou, miou = paddleseg.utils.metrics.mean_iou(
        #     intersect_area_all, pred_area_all, label_area_all)
        # # TODO 确认是按oacc还是macc
        # class_acc, oacc = paddleseg.utils.metrics.accuracy(intersect_area_all,
        #                                                    pred_area_all)
        # kappa = paddleseg.utils.metrics.kappa(intersect_area_all, pred_area_all,
        #                                       label_area_all)

        category_f1score = metrics.f1_score(intersect_area_all, pred_area_all,
                                            label_area_all)
        # eval_metrics = OrderedDict(
        #     zip([
        #         'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
        #         'category_F1-score'
        #     ], [miou, class_iou, oacc, class_acc, kappa, category_f1score]))
        print(category_f1score)


