import sys
import paddle
import os
import argparse
#加入环境
sys.path.append('./STANET_Paddle/')
import paddlers as pdrs
from paddlers import transforms as T
import paddle.nn as nn
import paddle
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--out_dir', '-s', type=str, default=None, help='path to save inference model')
    parser.add_argument('-lr', type=float, default=0.001, help='lr')
    parser.add_argument('--decay_step', type=int, default=5000, help='epoch number')  
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_epoch', type=int, default=3, help='save epoch')
    parser.add_argument('--state_dict_path', type=str, default=None ,help='where model params')
    parser.add_argument('--pruned_flops', type=float, default=0.5 ,help='where model params')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST_PATH = os.path.join(DATA_DIR,'train.txt')
    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR,'val.txt')
    TESTLE_LIST_PATH = os.path.join(DATA_DIR,'test.txt')

    EXP_DIR = args.out_dir
    LR = args.lr
    DECAY_STEP = args.decay_step
    NUM_EPOCHS = args.num_epoch
    # 每多少个epoch保存一次模型权重参数
    SAVE_INTERVAL_EPOCHS = args.save_epoch
    #训练阶段 batch size
    TRAIN_BATCH_SIZE = args.batch_size
    state_dict_path= args.state_dict_path
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
    # 初始化模型，并进行训练
    # 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/paddlers/blob/develop/docs/visualdl.md

    num_classes = 2
    model = pdrs.tasks.STANet( in_channels=3, num_classes=num_classes, att_type='PAM', ds_factor=1)     
    model.net_initialize(pretrain_weights = state_dict_path)
    model.net.eval()

    LR=0.001
    DECAY_STEP=5000
    # 制定定步长学习率衰减策略
    lr_scheduler = paddle.optimizer.lr.StepDecay(
        LR,
        step_size=DECAY_STEP,
        # 学习率衰减系数，这里指定每次减半
        gamma=0.5
    )
    # 构造Adam优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.net.parameters()
    )

    # model.get_prune(eval_dataset,save_dir="./prunemodel_pre")
    # model.prune_nosensi(pruned_flops=args.pruned_flops,save_dir="./prunemodel",eval_dataset=eval_dataset,optimizer=optimizer)


    model.get_sensitivity(eval_dataset,save_dir="./prunemodel_pre")
    model.prune(pruned_flops=args.pruned_flops,save_dir="./prunemodel",eval_dataset=eval_dataset,optimizer=optimizer)   


    # eval_metrics = model.evaluate(eval_dataset)
    # print(str(eval_metrics))
  