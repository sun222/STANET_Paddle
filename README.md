# 基于Paddle复现
## 1.论文简介
STANET: [A Spatial-T emporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)，

<img src=./docs/stanetmodel.png></img>

本次复现的模型为时空注意神经网络(STANet)。
STANet设计了两种类型的自我注意模块。基本时空注意模块(BAM)。金字塔时空注意模块(PAM)

**参考实现**：https://github.com/justchenhao/STANet

## 2.复现精度

在LEVIR的测试集的测试效果如下表,达到验收指标，F1-Score=0.91  满足精度要求 
model.pdiparams文件大小为14.6m
model.pdmodel文件大小为5.11m，和为19.7m 满足要求
综上所述 满足挑战指标要求





## 3.环境依赖
通过以下命令安装对应依赖
```shell
cd STANET_Paddle/
pip install -r requirements.txt
pip install https://paddle-model-ecology.bj.bcebos.com/whl/paddleclas-0.0.0-py3-none-any.whl
```

## 4.数据集

下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/136610](https://aistudio.baidu.com/aistudio/datasetdetail/136610)

数据集下载解压后需要生成.txt文件用于训练。执行以下命令。

```shell
#切片
python ./STANET_Paddle/tools/spliter-cd.py  --image_folder  data/LEVIR-CD --block_size 256 --save_folder dataset
```
**参数介绍**：

- image_folder:数据集路径
- block_size:切片大小
- save_folder:保存路径
 ```shell
# 创建列表
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/train --A A --B B --label label --save_txt train.txt
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/val --A A --B B --label label --save_txt val.txt
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/test --A A --B B --label label --save_txt test.txt
```
**参数介绍**：

- image_folder:切片后数据集路径
- -A  -B  -label :A时相、B时相、label的子路径名
- save_txt:保存名

## 5.快速开始

### 模型训练

运行一下命令进行模型训练，在训练过程中会对模型进行评估，启用了VisualDL日志功能，运行之后在`/output/stanet/vdl_log` 文件夹下找到对应的日志文件

训练过程中的vdl文件为vdlrecords.1655664431.log

```shell
!python ./STANET_Paddle/tutorials/train/stanet_train_bonesmall.py --data_dir=./dataset/   --out_dir=./output1/stanet/   --batch_size=8     --num_epoch=100
```

**参数介绍**：

- data_dir:数据集路径

- out_dir:模型输出文件夹
- batch_size：batch大小

其他超参数已经设置好。最后一个epoch结束 



达到验收指标。


### 模型验证

除了可以再训练过程中验证模型精度，可以使用stanet_eval_bone.py脚本进行测试，

动态模型为：
链接：https://pan.baidu.com/s/1LHsQ6nIUwoJVfHkDSIHHLA 
提取码：y90l

```shell
!python ./STANET_Paddle/tutorials/eval/stanet_eval_bone.py --data_dir=./dataset/   --state_dict_path=./output1/stanet/best_model/model.pdparams
```
**参数介绍**：

- data_dir:数据集路径

- weight_path:模型权重所在路径

输出如下：

```shell
2022-06-20 16:39:52 [INFO]	Loading pretrained model from best_model/model.pdparams
2022-06-20 16:39:52 [INFO]	There are 393/393 variables loaded into STANet.
2022-06-20 16:39:52 [INFO]	Start to evaluate(total_samples=1024, total_steps=1024)...
OrderedDict([('miou', 0.9182049414147296), ('category_iou', array([0.99265375, 0.84375614])), ('oacc', 0.992934063076973), ('category_acc', array([0.99602765, 0.92133201])), ('kappa', 0.9115713464149582), ('category_F1-score', array([0.99631333, 0.91525785]))])
```



### 导出

可以将模型导出，动态图转静态图，使模型预测更快，可以使用stanet_export_bone..py脚本进行测试

在这里因为动静态模型转化的问题，修改了stanet的模型代码使其可以转出静态模型。

调试过程中参考这份文档   [报错调试](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/debugging_cn.html)

```shell
!python ./STANET_Paddle/deploy/export/stanet_export_bone.py    --state_dict_path=./output1/stanet/best_model/model.pdparams    --save_dir=./inference_model_bone/  --fixed_input_shape=[1,3,256,256]  
```
**参数介绍**：
- fixed_input_shape:预测图的形状
-save_dir  静态图导出路径
- state_dict_path:模型权重所在路径



### 使用静态图推理

可以使用stanet_infer.py脚本进行测试

```shell
!python ./STANET_Paddle/tutorials/infer/stanet_infer.py   --infer_dir=./inference_model_bone   --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset/ --output_dir=./STANET_Paddle/test_tipc/result/predict_output
```
**参数介绍**：
- infer_dir:模型文件路径
- img_dir：用于推理的图片路径
- output_dir：预测结果输出路径




### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


```shell
bash  ./STANET_Paddle/test_tipc/prepare.sh  ./STANET_Paddle/test_tipc/configs/stanet/train_infer_python.txt 'lite_train_lite_infer'

bash  ./STANET_Paddle/test_tipc/test_train_inference_python.sh ./STANET_Paddle/test_tipc/configs/stanet/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示

```shell

aistudio@jupyter-2315405-4166493:~$ bash  ./STANET_Paddle/test_tipc/test_train_inference_python.sh ./STANET_Paddle/test_tipc/configs/stanet/train_infer_python.txt 'lite_train_lite_infer'
[06-15 18:31:47 MainThread @logger.py:242] Argv: ./STANET_Paddle/tutorials/train/stanet_train_bonesmall.py --data_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --out_dir=./STANET_Paddle/test_tipc/result/stanet/ --num_epoch=6 --save_epoch=2 --batch_size=2
[06-15 18:31:47 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:31:47,700-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:31:47,701-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                     Models supported by PaddleClas                                                                     
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|       Series      |                                                                       Name                                                                       |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                                                     AlexNet                                                                      |
|      DarkNet      |                                                                    DarkNet53                                                                     |
|        DeiT       | DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224  DeiT_base_patch16_384  DeiT_small_distilled_patch16_224 |
|                   |                                  DeiT_small_patch16_224  DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                                  |
|      DenseNet     |                                         DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                                          |
|        DLA        |                                    DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169                                    |
|        DPN        |                                                       DPN68  DPN92  DPN98  DPN107  DPN131                                                        |
|    EfficientNet   |       EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4  EfficientNetB5  EfficientNetB6       |
|                   |                                                                  EfficientNetB7                                                                  |
|       ESNet       |                                                 ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                                                 |
|      GhostNet     |                                         GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                                          |
|      HarDNet      |                                                 HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                                                 |
|       HRNet       |          HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C  HRNet_W18_C_ssld  HRNet_W48_C_ssld           |
|     Inception     |                                                       GoogLeNet  InceptionV3  InceptionV4                                                        |
|       MixNet      |                                                           MixNet_S  MixNet_M  MixNet_L                                                           |
|    MobileNetV1    |                              MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld                               |
|    MobileNetV2    |            MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5  MobileNetV2_x2_0  MobileNetV2_ssld             |
|    MobileNetV3    |            MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0  MobileNetV3_small_x1_25             |
|                   |            MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75  MobileNetV3_large_x1_0  MobileNetV3_large_x1_25             |
|                   |                                             MobileNetV3_small_x1_0_ssld  MobileNetV3_large_x1_0_ssld                                             |
|      PPLCNet      |                PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  PPLCNet_x2_5                 |
|       RedNet      |                                                RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                                                |
|       RegNet      |                                                                   RegNetX_4GF                                                                    |
|      Res2Net      |          Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  Res2Net50_vd_26w_4s_ssld           |
|                   |                                               Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld                                               |
|      ResNeSt      |                                                        ResNeSt50  ResNeSt50_fast_1s1x64d                                                         |
|       ResNet      |       ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2  ResNet101  ResNet101_vd  ResNet152       |
|                   |         ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2          |
|                   |                                                              ResNet50_ACNet_deploy                                                               |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d  ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl       |
|                   |      ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl  ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d       |
|                   |                                   ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d                                   |
|       ReXNet      |                                            ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                                            |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d  SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d |
|    ShuffleNetV2   |      ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0  ShuffleNetV2_x1_5  ShuffleNetV2_x2_0      |
|     SqueezeNet    |                                                           SqueezeNet1_0  SqueezeNet1_1                                                           |
|  SwinTransformer  |                       SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k                        |
|                   |   SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k  SwinTransformer_base_patch4_window12_384    |
|                   |            SwinTransformer_base_patch4_window7_224  SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224            |
|       Twins       |                                 pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large                                 |
|        VGG        |                                                            VGG11  VGG13  VGG16  VGG19                                                            |
| VisionTransformer |      ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224  ViT_large_patch16_384  ViT_large_patch32_384       |
|                   |                                                              ViT_small_patch16_224                                                               |
|      Xception     |                                    Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71                                    |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
                                                                                                                                                Powered by PaddlePaddle!
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2022-06-15 18:31:47 [INFO]      16 samples in file ./STANET_Paddle/test_tipc/data/mini_levir_dataset/train.txt
2022-06-15 18:31:47 [INFO]      8 samples in file ./STANET_Paddle/test_tipc/data/mini_levir_dataset/val.txt
W0615 18:31:47.809813  6982 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0615 18:31:47.812942  6982 gpu_context.cc:306] device: 0, cuDNN Version: 8.2.
INFO 2022-06-15 18:31:50,207 logger.py:79] unique_endpoints {''}
[2022/06/15 18:31:50] root INFO: unique_endpoints {''}
INFO 2022-06-15 18:31:50,207 logger.py:79] Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
[2022/06/15 18:31:50] root INFO: Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
2022-06-15 18:31:55 [INFO]      [TRAIN] Epoch 1 finished, loss=0.43874335 .
2022-06-15 18:31:56 [INFO]      [TRAIN] Epoch 2 finished, loss=0.1586074 .
2022-06-15 18:31:56 [WARNING]   Segmenter only supports batch_size=1 for each gpu/cpu card during evaluation, so batch_size is forcibly set to 1.
2022-06-15 18:31:56 [INFO]      Start to evaluate(total_samples=8, total_steps=8)...
2022-06-15 18:31:57 [INFO]      [EVAL] Finished, Epoch=2, miou=0.493079, category_iou=[0.98615837 0.        ], oacc=0.986158, category_acc=[0.98615837 0.        ], kappa=0.000000, category_F1-score=[0.99303095 0.        ] .
2022-06-15 18:31:57 [INFO]      Model saved in ./STANET_Paddle/test_tipc/result/stanet/best_model.
2022-06-15 18:31:57 [INFO]      Current evaluated best model on eval_dataset is epoch_2, miou=0.49307918548583984
2022-06-15 18:31:57 [INFO]      Model saved in ./STANET_Paddle/test_tipc/result/stanet/epoch_2.
2022-06-15 18:31:59 [INFO]      [TRAIN] Epoch 3 finished, loss=0.15517747 .
2022-06-15 18:32:00 [INFO]      [TRAIN] Epoch 4 finished, loss=0.13931209 .
2022-06-15 18:32:00 [WARNING]   Segmenter only supports batch_size=1 for each gpu/cpu card during evaluation, so batch_size is forcibly set to 1.
2022-06-15 18:32:00 [INFO]      Start to evaluate(total_samples=8, total_steps=8)...
2022-06-15 18:32:01 [INFO]      [EVAL] Finished, Epoch=4, miou=0.493079, category_iou=[0.98615837 0.        ], oacc=0.986158, category_acc=[0.98615837 0.        ], kappa=0.000000, category_F1-score=[0.99303095 0.        ] .
2022-06-15 18:32:01 [INFO]      Current evaluated best model on eval_dataset is epoch_2, miou=0.49307918548583984
2022-06-15 18:32:01 [INFO]      Model saved in ./STANET_Paddle/test_tipc/result/stanet/epoch_4.
2022-06-15 18:32:03 [INFO]      [TRAIN] Epoch 5 finished, loss=0.17409432 .
2022-06-15 18:32:04 [INFO]      [TRAIN] Epoch 6 finished, loss=0.13709755 .
2022-06-15 18:32:04 [WARNING]   Segmenter only supports batch_size=1 for each gpu/cpu card during evaluation, so batch_size is forcibly set to 1.
2022-06-15 18:32:04 [INFO]      Start to evaluate(total_samples=8, total_steps=8)...
2022-06-15 18:32:05 [INFO]      [EVAL] Finished, Epoch=6, miou=0.492938, category_iou=[0.98587608 0.        ], oacc=0.985876, category_acc=[0.98615446 0.        ], kappa=-0.000554, category_F1-score=[0.99288782        nan] .
2022-06-15 18:32:05 [INFO]      Current evaluated best model on eval_dataset is epoch_2, miou=0.49307918548583984
2022-06-15 18:32:05 [INFO]      Model saved in ./STANET_Paddle/test_tipc/result/stanet/epoch_6.
 Run successfully with command - python3.7 ./STANET_Paddle/tutorials/train/stanet_train_bonesmall.py --data_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset   --out_dir=./STANET_Paddle/test_tipc/result/stanet/   --num_epoch=6 --save_epoch=2  --batch_size=2 !  
[06-15 18:32:12 MainThread @logger.py:242] Argv: ./STANET_Paddle/tutorials/eval/stanet_eval_bone.py --data_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --state_dict_path=./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams
[06-15 18:32:12 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:32:12,838-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:32:12,838-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                     Models supported by PaddleClas                                                                     
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|       Series      |                                                                       Name                                                                       |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                                                     AlexNet                                                                      |
|      DarkNet      |                                                                    DarkNet53                                                                     |
|        DeiT       | DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224  DeiT_base_patch16_384  DeiT_small_distilled_patch16_224 |
|                   |                                  DeiT_small_patch16_224  DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                                  |
|      DenseNet     |                                         DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                                          |
|        DLA        |                                    DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169                                    |
|        DPN        |                                                       DPN68  DPN92  DPN98  DPN107  DPN131                                                        |
|    EfficientNet   |       EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4  EfficientNetB5  EfficientNetB6       |
|                   |                                                                  EfficientNetB7                                                                  |
|       ESNet       |                                                 ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                                                 |
|      GhostNet     |                                         GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                                          |
|      HarDNet      |                                                 HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                                                 |
|       HRNet       |          HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C  HRNet_W18_C_ssld  HRNet_W48_C_ssld           |
|     Inception     |                                                       GoogLeNet  InceptionV3  InceptionV4                                                        |
|       MixNet      |                                                           MixNet_S  MixNet_M  MixNet_L                                                           |
|    MobileNetV1    |                              MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld                               |
|    MobileNetV2    |            MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5  MobileNetV2_x2_0  MobileNetV2_ssld             |
|    MobileNetV3    |            MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0  MobileNetV3_small_x1_25             |
|                   |            MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75  MobileNetV3_large_x1_0  MobileNetV3_large_x1_25             |
|                   |                                             MobileNetV3_small_x1_0_ssld  MobileNetV3_large_x1_0_ssld                                             |
|      PPLCNet      |                PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  PPLCNet_x2_5                 |
|       RedNet      |                                                RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                                                |
|       RegNet      |                                                                   RegNetX_4GF                                                                    |
|      Res2Net      |          Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  Res2Net50_vd_26w_4s_ssld           |
|                   |                                               Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld                                               |
|      ResNeSt      |                                                        ResNeSt50  ResNeSt50_fast_1s1x64d                                                         |
|       ResNet      |       ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2  ResNet101  ResNet101_vd  ResNet152       |
|                   |         ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2          |
|                   |                                                              ResNet50_ACNet_deploy                                                               |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d  ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl       |
|                   |      ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl  ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d       |
|                   |                                   ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d                                   |
|       ReXNet      |                                            ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                                            |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d  SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d |
|    ShuffleNetV2   |      ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0  ShuffleNetV2_x1_5  ShuffleNetV2_x2_0      |
|     SqueezeNet    |                                                           SqueezeNet1_0  SqueezeNet1_1                                                           |
|  SwinTransformer  |                       SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k                        |
|                   |   SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k  SwinTransformer_base_patch4_window12_384    |
|                   |            SwinTransformer_base_patch4_window7_224  SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224            |
|       Twins       |                                 pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large                                 |
|        VGG        |                                                            VGG11  VGG13  VGG16  VGG19                                                            |
| VisionTransformer |      ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224  ViT_large_patch16_384  ViT_large_patch32_384       |
|                   |                                                              ViT_small_patch16_224                                                               |
|      Xception     |                                    Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71                                    |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
                                                                                                                                                Powered by PaddlePaddle!
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2022-06-15 18:32:12 [INFO]      8 samples in file ./STANET_Paddle/test_tipc/data/mini_levir_dataset/val.txt
W0615 18:32:12.947710  7110 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0615 18:32:12.950922  7110 gpu_context.cc:306] device: 0, cuDNN Version: 8.2.
INFO 2022-06-15 18:32:15,346 logger.py:79] unique_endpoints {''}
[2022/06/15 18:32:15] root INFO: unique_endpoints {''}
INFO 2022-06-15 18:32:15,346 logger.py:79] Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
[2022/06/15 18:32:15] root INFO: Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
2022-06-15 18:32:15 [INFO]      Loading pretrained model from ./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams
2022-06-15 18:32:15 [INFO]      There are 393/393 variables loaded into STANet.
2022-06-15 18:32:15 [INFO]      Start to evaluate(total_samples=8, total_steps=8)...
OrderedDict([('miou', 0.49307918548583984), ('category_iou', array([0.98615837, 0.        ])), ('oacc', 0.9861583709716797), ('category_acc', array([0.98615837, 0.        ])), ('kappa', 0.0), ('category_F1-score', array([0.99303095, 0.        ]))])
 Run successfully with command - python3.7  ./STANET_Paddle/tutorials/eval/stanet_eval_bone.py --data_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset    --state_dict_path=./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams !  
[06-15 18:32:24 MainThread @logger.py:242] Argv: ./STANET_Paddle/deploy/export/stanet_export_bone.py --state_dict_path=./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams --save_dir=./STANET_Paddle/test_tipc/result/inference_model/ --fixed_input_shape=[1,3,256,256]
[06-15 18:32:24 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:32:24,681-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:32:24,681-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                     Models supported by PaddleClas                                                                     
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|       Series      |                                                                       Name                                                                       |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                                                     AlexNet                                                                      |
|      DarkNet      |                                                                    DarkNet53                                                                     |
|        DeiT       | DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224  DeiT_base_patch16_384  DeiT_small_distilled_patch16_224 |
|                   |                                  DeiT_small_patch16_224  DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                                  |
|      DenseNet     |                                         DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                                          |
|        DLA        |                                    DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169                                    |
|        DPN        |                                                       DPN68  DPN92  DPN98  DPN107  DPN131                                                        |
|    EfficientNet   |       EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4  EfficientNetB5  EfficientNetB6       |
|                   |                                                                  EfficientNetB7                                                                  |
|       ESNet       |                                                 ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                                                 |
|      GhostNet     |                                         GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                                          |
|      HarDNet      |                                                 HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                                                 |
|       HRNet       |          HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C  HRNet_W18_C_ssld  HRNet_W48_C_ssld           |
|     Inception     |                                                       GoogLeNet  InceptionV3  InceptionV4                                                        |
|       MixNet      |                                                           MixNet_S  MixNet_M  MixNet_L                                                           |
|    MobileNetV1    |                              MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld                               |
|    MobileNetV2    |            MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5  MobileNetV2_x2_0  MobileNetV2_ssld             |
|    MobileNetV3    |            MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0  MobileNetV3_small_x1_25             |
|                   |            MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75  MobileNetV3_large_x1_0  MobileNetV3_large_x1_25             |
|                   |                                             MobileNetV3_small_x1_0_ssld  MobileNetV3_large_x1_0_ssld                                             |
|      PPLCNet      |                PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  PPLCNet_x2_5                 |
|       RedNet      |                                                RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                                                |
|       RegNet      |                                                                   RegNetX_4GF                                                                    |
|      Res2Net      |          Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  Res2Net50_vd_26w_4s_ssld           |
|                   |                                               Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld                                               |
|      ResNeSt      |                                                        ResNeSt50  ResNeSt50_fast_1s1x64d                                                         |
|       ResNet      |       ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2  ResNet101  ResNet101_vd  ResNet152       |
|                   |         ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2          |
|                   |                                                              ResNet50_ACNet_deploy                                                               |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d  ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl       |
|                   |      ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl  ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d       |
|                   |                                   ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d                                   |
|       ReXNet      |                                            ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                                            |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d  SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d |
|    ShuffleNetV2   |      ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0  ShuffleNetV2_x1_5  ShuffleNetV2_x2_0      |
|     SqueezeNet    |                                                           SqueezeNet1_0  SqueezeNet1_1                                                           |
|  SwinTransformer  |                       SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k                        |
|                   |   SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k  SwinTransformer_base_patch4_window12_384    |
|                   |            SwinTransformer_base_patch4_window7_224  SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224            |
|       Twins       |                                 pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large                                 |
|        VGG        |                                                            VGG11  VGG13  VGG16  VGG19                                                            |
| VisionTransformer |      ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224  ViT_large_patch16_384  ViT_large_patch32_384       |
|                   |                                                              ViT_small_patch16_224                                                               |
|      Xception     |                                    Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71                                    |
+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
                                                                                                                                                Powered by PaddlePaddle!
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
W0615 18:32:24.788726  7290 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0615 18:32:24.791882  7290 gpu_context.cc:306] device: 0, cuDNN Version: 8.2.
INFO 2022-06-15 18:32:27,175 logger.py:79] unique_endpoints {''}
[2022/06/15 18:32:27] root INFO: unique_endpoints {''}
INFO 2022-06-15 18:32:27,176 logger.py:79] Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
[2022/06/15 18:32:27] root INFO: Found /home/aistudio/.paddleclas/weights/ESNet_x1_0_pretrained.pdparams
2022-06-15 18:32:27 [INFO]      Loading pretrained model from ./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams
2022-06-15 18:32:27 [INFO]      There are 393/393 variables loaded into STANet.
2022-06-15 18:32:32 [INFO]      The model for the inference deployment is saved in ./STANET_Paddle/test_tipc/result/inference_model/.
 Run successfully with command - python3.7  ./STANET_Paddle/deploy/export/stanet_export_bone.py     --state_dict_path=./STANET_Paddle/test_tipc/result/stanet/best_model/model.pdparams   --save_dir=./STANET_Paddle/test_tipc/result/inference_model/  --fixed_input_shape=[1,3,256,256]     !  
[06-15 18:32:39 MainThread @logger.py:242] Argv: ./STANET_Paddle/tutorials/infer/stanet_infer.py --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=True --precision=fp32 --enable_benchmark=True --use_tensorrt=False
[06-15 18:32:39 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:32:39,611-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:32:39,611-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
----------------------------------------------------------------------------------------------------------------------------
                                               Models supported by PaddleClas                                               
+-------------------+------------------------------------------------------------------------------------------------------+
|       Series      |                                                 Name                                                 |
+-------------------+------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                               AlexNet                                                |
|      DarkNet      |                                              DarkNet53                                               |
|        DeiT       |       DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224        |
|                   |           DeiT_base_patch16_384  DeiT_small_distilled_patch16_224  DeiT_small_patch16_224            |
|                   |                        DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                        |
|      DenseNet     |                   DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                    |
|        DLA        |              DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169              |
|        DPN        |                                 DPN68  DPN92  DPN98  DPN107  DPN131                                  |
|    EfficientNet   | EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4 |
|                   |                            EfficientNetB5  EfficientNetB6  EfficientNetB7                            |
|       ESNet       |                           ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                           |
|      GhostNet     |                   GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                    |
|      HarDNet      |                           HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                           |
|       HRNet       |      HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C       |
|                   |                                  HRNet_W18_C_ssld  HRNet_W48_C_ssld                                  |
|     Inception     |                                 GoogLeNet  InceptionV3  InceptionV4                                  |
|       MixNet      |                                     MixNet_S  MixNet_M  MixNet_L                                     |
|    MobileNetV1    |        MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld         |
|    MobileNetV2    |        MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5         |
|                   |                                  MobileNetV2_x2_0  MobileNetV2_ssld                                  |
|    MobileNetV3    |   MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0   |
|                   |  MobileNetV3_small_x1_25  MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75   |
|                   |             MobileNetV3_large_x1_0  MobileNetV3_large_x1_25  MobileNetV3_small_x1_0_ssld             |
|                   |                                     MobileNetV3_large_x1_0_ssld                                      |
|      PPLCNet      | PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  |
|                   |                                             PPLCNet_x2_5                                             |
|       RedNet      |                          RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                          |
|       RegNet      |                                             RegNetX_4GF                                              |
|      Res2Net      | Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  |
|                   |            Res2Net50_vd_26w_4s_ssld  Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld            |
|      ResNeSt      |                                  ResNeSt50  ResNeSt50_fast_1s1x64d                                   |
|       ResNet      |   ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2   |
|                   |  ResNet101  ResNet101_vd  ResNet152  ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  |
|                   |        ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2  ResNet50_ACNet_deploy        |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d      |
|                   |       ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl  ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl        |
|                   |       ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d        |
|                   |             ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d             |
|       ReXNet      |                      ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                      |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d |
|                   |                              SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d                              |
|    ShuffleNetV2   |   ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0   |
|                   |                                 ShuffleNetV2_x1_5  ShuffleNetV2_x2_0                                 |
|     SqueezeNet    |                                     SqueezeNet1_0  SqueezeNet1_1                                     |
|  SwinTransformer  | SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k  |
|                   |  SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k   |
|                   |          SwinTransformer_base_patch4_window12_384  SwinTransformer_base_patch4_window7_224           |
|                   |          SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224           |
|       Twins       |           pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large           |
|        VGG        |                                      VGG11  VGG13  VGG16  VGG19                                      |
| VisionTransformer |       ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224        |
|                   |                 ViT_large_patch16_384  ViT_large_patch32_384  ViT_small_patch16_224                  |
|      Xception     |              Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71              |
+-------------------+------------------------------------------------------------------------------------------------------+
                                                                                                    Powered by PaddlePaddle!
----------------------------------------------------------------------------------------------------------------------------
2022-06-15 18:32:39 [INFO]      Model[STANet] loaded.
total file number is 16
------------------ Inference Time Info ----------------------
total_time(ms): 3240.7, img_num: 1, batch_size: 1
average latency time(ms): 3240.70, QPS: 0.308575
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 3231.50, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 54.199999999999996, img_num: 1, batch_size: 1
average latency time(ms): 54.20, QPS: 18.450185
preprocess_time_per_im(ms): 8.80, inference_time_per_batch(ms): 45.20, postprocess_time_per_im(ms): 0.20
------------------ Inference Time Info ----------------------
total_time(ms): 58.9, img_num: 1, batch_size: 1
average latency time(ms): 58.90, QPS: 16.977929
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 50.70, postprocess_time_per_im(ms): 0.20
------------------ Inference Time Info ----------------------
total_time(ms): 53.3, img_num: 1, batch_size: 1
average latency time(ms): 53.30, QPS: 18.761726
preprocess_time_per_im(ms): 8.40, inference_time_per_batch(ms): 44.80, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.3, img_num: 1, batch_size: 1
average latency time(ms): 52.30, QPS: 19.120459
preprocess_time_per_im(ms): 7.90, inference_time_per_batch(ms): 44.30, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.5, img_num: 1, batch_size: 1
average latency time(ms): 52.50, QPS: 19.047619
preprocess_time_per_im(ms): 8.10, inference_time_per_batch(ms): 44.30, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.3, img_num: 1, batch_size: 1
average latency time(ms): 52.30, QPS: 19.120459
preprocess_time_per_im(ms): 7.90, inference_time_per_batch(ms): 44.30, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.3, img_num: 1, batch_size: 1
average latency time(ms): 52.30, QPS: 19.120459
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.2, img_num: 1, batch_size: 1
average latency time(ms): 52.20, QPS: 19.157088
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.10, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 51.9, img_num: 1, batch_size: 1
average latency time(ms): 51.90, QPS: 19.267823
preprocess_time_per_im(ms): 7.70, inference_time_per_batch(ms): 44.10, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.3, img_num: 1, batch_size: 1
average latency time(ms): 52.30, QPS: 19.120459
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.2, img_num: 1, batch_size: 1
average latency time(ms): 52.20, QPS: 19.157088
preprocess_time_per_im(ms): 7.80, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.20
------------------ Inference Time Info ----------------------
total_time(ms): 52.3, img_num: 1, batch_size: 1
average latency time(ms): 52.30, QPS: 19.120459
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.1, img_num: 1, batch_size: 1
average latency time(ms): 52.10, QPS: 19.193858
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.00, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.1, img_num: 1, batch_size: 1
average latency time(ms): 52.10, QPS: 19.193858
preprocess_time_per_im(ms): 7.80, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.10
------------------ Inference Time Info ----------------------
total_time(ms): 52.400000000000006, img_num: 1, batch_size: 1
average latency time(ms): 52.40, QPS: 19.083969
preprocess_time_per_im(ms): 8.00, inference_time_per_batch(ms): 44.20, postprocess_time_per_im(ms): 0.20
 Run successfully with command - python3.7  ./STANET_Paddle/tutorials/infer/stanet_infer.py   --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=True --precision=fp32   --enable_benchmark=True --use_tensorrt=False      > ./test_tipc/output/STANET/python_infer_gpu_usetrt_fp32_precision_null_batchsize_False.log 2>&1 !  
[06-15 18:32:50 MainThread @logger.py:242] Argv: ./STANET_Paddle/tutorials/infer/stanet_infer.py --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --enable_benchmark=True --use_tensorrt=False
[06-15 18:32:50 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:32:50,416-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:32:50,416-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
----------------------------------------------------------------------------------------------------------------------------
                                               Models supported by PaddleClas                                               
+-------------------+------------------------------------------------------------------------------------------------------+
|       Series      |                                                 Name                                                 |
+-------------------+------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                               AlexNet                                                |
|      DarkNet      |                                              DarkNet53                                               |
|        DeiT       |       DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224        |
|                   |           DeiT_base_patch16_384  DeiT_small_distilled_patch16_224  DeiT_small_patch16_224            |
|                   |                        DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                        |
|      DenseNet     |                   DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                    |
|        DLA        |              DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169              |
|        DPN        |                                 DPN68  DPN92  DPN98  DPN107  DPN131                                  |
|    EfficientNet   | EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4 |
|                   |                            EfficientNetB5  EfficientNetB6  EfficientNetB7                            |
|       ESNet       |                           ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                           |
|      GhostNet     |                   GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                    |
|      HarDNet      |                           HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                           |
|       HRNet       |      HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C       |
|                   |                                  HRNet_W18_C_ssld  HRNet_W48_C_ssld                                  |
|     Inception     |                                 GoogLeNet  InceptionV3  InceptionV4                                  |
|       MixNet      |                                     MixNet_S  MixNet_M  MixNet_L                                     |
|    MobileNetV1    |        MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld         |
|    MobileNetV2    |        MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5         |
|                   |                                  MobileNetV2_x2_0  MobileNetV2_ssld                                  |
|    MobileNetV3    |   MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0   |
|                   |  MobileNetV3_small_x1_25  MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75   |
|                   |             MobileNetV3_large_x1_0  MobileNetV3_large_x1_25  MobileNetV3_small_x1_0_ssld             |
|                   |                                     MobileNetV3_large_x1_0_ssld                                      |
|      PPLCNet      | PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  |
|                   |                                             PPLCNet_x2_5                                             |
|       RedNet      |                          RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                          |
|       RegNet      |                                             RegNetX_4GF                                              |
|      Res2Net      | Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  |
|                   |            Res2Net50_vd_26w_4s_ssld  Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld            |
|      ResNeSt      |                                  ResNeSt50  ResNeSt50_fast_1s1x64d                                   |
|       ResNet      |   ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2   |
|                   |  ResNet101  ResNet101_vd  ResNet152  ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  |
|                   |        ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2  ResNet50_ACNet_deploy        |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d      |
|                   |       ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl  ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl        |
|                   |       ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d        |
|                   |             ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d             |
|       ReXNet      |                      ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                      |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d |
|                   |                              SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d                              |
|    ShuffleNetV2   |   ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0   |
|                   |                                 ShuffleNetV2_x1_5  ShuffleNetV2_x2_0                                 |
|     SqueezeNet    |                                     SqueezeNet1_0  SqueezeNet1_1                                     |
|  SwinTransformer  | SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k  |
|                   |  SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k   |
|                   |          SwinTransformer_base_patch4_window12_384  SwinTransformer_base_patch4_window7_224           |
|                   |          SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224           |
|       Twins       |           pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large           |
|        VGG        |                                      VGG11  VGG13  VGG16  VGG19                                      |
| VisionTransformer |       ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224        |
|                   |                 ViT_large_patch16_384  ViT_large_patch32_384  ViT_small_patch16_224                  |
|      Xception     |              Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71              |
+-------------------+------------------------------------------------------------------------------------------------------+
                                                                                                    Powered by PaddlePaddle!
----------------------------------------------------------------------------------------------------------------------------
2022-06-15 18:32:50 [INFO]      Model[STANet] loaded.
---    fused 0 elementwise_add with relu activation
---    fused 0 elementwise_add with tanh activation
---    fused 0 elementwise_add with leaky_relu activation
---    fused 0 elementwise_add with swish activation
---    fused 0 elementwise_add with hardswish activation
---    fused 0 elementwise_add with sqrt activation
---    fused 0 elementwise_add with abs activation
---    fused 0 elementwise_add with clip activation
---    fused 0 elementwise_add with gelu activation
---    fused 0 elementwise_add with relu6 activation
---    fused 0 elementwise_add with sigmoid activation
---    fused 0 elementwise_sub with relu activation
---    fused 0 elementwise_sub with tanh activation
---    fused 0 elementwise_sub with leaky_relu activation
---    fused 0 elementwise_sub with swish activation
---    fused 0 elementwise_sub with hardswish activation
---    fused 0 elementwise_sub with sqrt activation
---    fused 1 elementwise_sub with abs activation
---    fused 0 elementwise_sub with clip activation
---    fused 0 elementwise_sub with gelu activation
---    fused 0 elementwise_sub with relu6 activation
---    fused 0 elementwise_sub with sigmoid activation
---    fused 0 elementwise_mul with relu activation
---    fused 0 elementwise_mul with tanh activation
---    fused 0 elementwise_mul with leaky_relu activation
---    fused 0 elementwise_mul with swish activation
---    fused 0 elementwise_mul with hardswish activation
---    fused 0 elementwise_mul with sqrt activation
---    fused 0 elementwise_mul with abs activation
---    fused 0 elementwise_mul with clip activation
---    fused 0 elementwise_mul with gelu activation
---    fused 0 elementwise_mul with relu6 activation
---    fused 0 elementwise_mul with sigmoid activation
total file number is 16
------------------ Inference Time Info ----------------------
total_time(ms): 981.1, img_num: 1, batch_size: 1
average latency time(ms): 981.10, QPS: 1.019264
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 971.70, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 677.2, img_num: 1, batch_size: 1
average latency time(ms): 677.20, QPS: 1.476669
preprocess_time_per_im(ms): 9.00, inference_time_per_batch(ms): 667.70, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 1377.0, img_num: 1, batch_size: 1
average latency time(ms): 1377.00, QPS: 0.726216
preprocess_time_per_im(ms): 9.40, inference_time_per_batch(ms): 1367.20, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 1015.9, img_num: 1, batch_size: 1
average latency time(ms): 1015.90, QPS: 0.984349
preprocess_time_per_im(ms): 9.40, inference_time_per_batch(ms): 1006.20, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 756.1, img_num: 1, batch_size: 1
average latency time(ms): 756.10, QPS: 1.322576
preprocess_time_per_im(ms): 9.00, inference_time_per_batch(ms): 746.70, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 747.4, img_num: 1, batch_size: 1
average latency time(ms): 747.40, QPS: 1.337972
preprocess_time_per_im(ms): 9.00, inference_time_per_batch(ms): 738.00, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 876.3, img_num: 1, batch_size: 1
average latency time(ms): 876.30, QPS: 1.141162
preprocess_time_per_im(ms): 9.10, inference_time_per_batch(ms): 866.70, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 781.2, img_num: 1, batch_size: 1
average latency time(ms): 781.20, QPS: 1.280082
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 772.00, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 1018.4, img_num: 1, batch_size: 1
average latency time(ms): 1018.40, QPS: 0.981932
preprocess_time_per_im(ms): 8.70, inference_time_per_batch(ms): 1009.30, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 1153.0, img_num: 1, batch_size: 1
average latency time(ms): 1153.00, QPS: 0.867303
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 1143.70, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 663.8, img_num: 1, batch_size: 1
average latency time(ms): 663.80, QPS: 1.506478
preprocess_time_per_im(ms): 9.10, inference_time_per_batch(ms): 654.40, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 687.2, img_num: 1, batch_size: 1
average latency time(ms): 687.20, QPS: 1.455180
preprocess_time_per_im(ms): 8.40, inference_time_per_batch(ms): 678.40, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 837.3000000000001, img_num: 1, batch_size: 1
average latency time(ms): 837.30, QPS: 1.194315
preprocess_time_per_im(ms): 9.30, inference_time_per_batch(ms): 827.60, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 803.2, img_num: 1, batch_size: 1
average latency time(ms): 803.20, QPS: 1.245020
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 793.90, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 781.8000000000001, img_num: 1, batch_size: 1
average latency time(ms): 781.80, QPS: 1.279100
preprocess_time_per_im(ms): 9.10, inference_time_per_batch(ms): 772.40, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 828.3000000000001, img_num: 1, batch_size: 1
average latency time(ms): 828.30, QPS: 1.207292
preprocess_time_per_im(ms): 8.70, inference_time_per_batch(ms): 819.30, postprocess_time_per_im(ms): 0.30
 Run successfully with command - python3.7  ./STANET_Paddle/tutorials/infer/stanet_infer.py   --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --enable_benchmark=True --use_tensorrt=False         > ./test_tipc/output/STANET/python_infer_cpu_usemkldnn_False_threads_1_precision_null_batchsize_False.log 2>&1 !  
[06-15 18:33:10 MainThread @logger.py:242] Argv: ./STANET_Paddle/tutorials/infer/stanet_infer.py --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=False --enable_mkldnn=False --cpu_threads=2 --enable_benchmark=True --use_tensorrt=False
[06-15 18:33:10 MainThread @utils.py:79] WRN paddlepaddle version: 2.3.0. The dynamic graph version of PARL is under development, not fully tested and supported
2022-06-15 18:33:10,834-WARNING: type object 'QuantizationTransformPass' has no attribute '_supported_quantizable_op_type'
2022-06-15 18:33:10,834-WARNING: If you want to use training-aware and post-training quantization, please use Paddle >= 1.8.4 or develop version
----------------------------------------------------------------------------------------------------------------------------
                                               Models supported by PaddleClas                                               
+-------------------+------------------------------------------------------------------------------------------------------+
|       Series      |                                                 Name                                                 |
+-------------------+------------------------------------------------------------------------------------------------------+
|      AlexNet      |                                               AlexNet                                                |
|      DarkNet      |                                              DarkNet53                                               |
|        DeiT       |       DeiT_base_distilled_patch16_224  DeiT_base_distilled_patch16_384  DeiT_base_patch16_224        |
|                   |           DeiT_base_patch16_384  DeiT_small_distilled_patch16_224  DeiT_small_patch16_224            |
|                   |                        DeiT_tiny_distilled_patch16_224  DeiT_tiny_patch16_224                        |
|      DenseNet     |                   DenseNet121  DenseNet161  DenseNet169  DenseNet201  DenseNet264                    |
|        DLA        |              DLA46_c  DLA60x_c  DLA34  DLA60  DLA60x  DLA102  DLA102x  DLA102x2  DLA169              |
|        DPN        |                                 DPN68  DPN92  DPN98  DPN107  DPN131                                  |
|    EfficientNet   | EfficientNetB0  EfficientNetB0_small  EfficientNetB1  EfficientNetB2  EfficientNetB3  EfficientNetB4 |
|                   |                            EfficientNetB5  EfficientNetB6  EfficientNetB7                            |
|       ESNet       |                           ESNet_x0_25  ESNet_x0_5  ESNet_x0_75  ESNet_x1_0                           |
|      GhostNet     |                   GhostNet_x0_5  GhostNet_x1_0  GhostNet_x1_3  GhostNet_x1_3_ssld                    |
|      HarDNet      |                           HarDNet39_ds  HarDNet68_ds  HarDNet68  HarDNet85                           |
|       HRNet       |      HRNet_W18_C  HRNet_W30_C  HRNet_W32_C  HRNet_W40_C  HRNet_W44_C  HRNet_W48_C  HRNet_W64_C       |
|                   |                                  HRNet_W18_C_ssld  HRNet_W48_C_ssld                                  |
|     Inception     |                                 GoogLeNet  InceptionV3  InceptionV4                                  |
|       MixNet      |                                     MixNet_S  MixNet_M  MixNet_L                                     |
|    MobileNetV1    |        MobileNetV1_x0_25  MobileNetV1_x0_5  MobileNetV1_x0_75  MobileNetV1  MobileNetV1_ssld         |
|    MobileNetV2    |        MobileNetV2_x0_25  MobileNetV2_x0_5  MobileNetV2_x0_75  MobileNetV2  MobileNetV2_x1_5         |
|                   |                                  MobileNetV2_x2_0  MobileNetV2_ssld                                  |
|    MobileNetV3    |   MobileNetV3_small_x0_35  MobileNetV3_small_x0_5  MobileNetV3_small_x0_75  MobileNetV3_small_x1_0   |
|                   |  MobileNetV3_small_x1_25  MobileNetV3_large_x0_35  MobileNetV3_large_x0_5  MobileNetV3_large_x0_75   |
|                   |             MobileNetV3_large_x1_0  MobileNetV3_large_x1_25  MobileNetV3_small_x1_0_ssld             |
|                   |                                     MobileNetV3_large_x1_0_ssld                                      |
|      PPLCNet      | PPLCNet_x0_25  PPLCNet_x0_35  PPLCNet_x0_5  PPLCNet_x0_75  PPLCNet_x1_0  PPLCNet_x1_5  PPLCNet_x2_0  |
|                   |                                             PPLCNet_x2_5                                             |
|       RedNet      |                          RedNet26  RedNet38  RedNet50  RedNet101  RedNet152                          |
|       RegNet      |                                             RegNetX_4GF                                              |
|      Res2Net      | Res2Net50_14w_8s  Res2Net50_26w_4s  Res2Net50_vd_26w_4s  Res2Net200_vd_26w_4s  Res2Net101_vd_26w_4s  |
|                   |            Res2Net50_vd_26w_4s_ssld  Res2Net101_vd_26w_4s_ssld  Res2Net200_vd_26w_4s_ssld            |
|      ResNeSt      |                                  ResNeSt50  ResNeSt50_fast_1s1x64d                                   |
|       ResNet      |   ResNet18  ResNet18_vd  ResNet34  ResNet34_vd  ResNet50  ResNet50_vc  ResNet50_vd  ResNet50_vd_v2   |
|                   |  ResNet101  ResNet101_vd  ResNet152  ResNet152_vd  ResNet200_vd  ResNet34_vd_ssld  ResNet50_vd_ssld  |
|                   |        ResNet50_vd_ssld_v2  ResNet101_vd_ssld  Fix_ResNet50_vd_ssld_v2  ResNet50_ACNet_deploy        |
|      ResNeXt      |      ResNeXt50_32x4d  ResNeXt50_vd_32x4d  ResNeXt50_64x4d  ResNeXt50_vd_64x4d  ResNeXt101_32x4d      |
|                   |       ResNeXt101_vd_32x4d  ResNeXt101_32x8d_wsl  ResNeXt101_32x16d_wsl  ResNeXt101_32x32d_wsl        |
|                   |       ResNeXt101_32x48d_wsl  Fix_ResNeXt101_32x48d_wsl  ResNeXt101_64x4d  ResNeXt101_vd_64x4d        |
|                   |             ResNeXt152_32x4d  ResNeXt152_vd_32x4d  ResNeXt152_64x4d  ResNeXt152_vd_64x4d             |
|       ReXNet      |                      ReXNet_1_0  ReXNet_1_3  ReXNet_1_5  ReXNet_2_0  ReXNet_3_0                      |
|       SENet       | SENet154_vd  SE_HRNet_W64_C_ssld  SE_ResNet18_vd  SE_ResNet34_vd  SE_ResNet50_vd  SE_ResNeXt50_32x4d |
|                   |                              SE_ResNeXt50_vd_32x4d  SE_ResNeXt101_32x4d                              |
|    ShuffleNetV2   |   ShuffleNetV2_swish  ShuffleNetV2_x0_25  ShuffleNetV2_x0_33  ShuffleNetV2_x0_5  ShuffleNetV2_x1_0   |
|                   |                                 ShuffleNetV2_x1_5  ShuffleNetV2_x2_0                                 |
|     SqueezeNet    |                                     SqueezeNet1_0  SqueezeNet1_1                                     |
|  SwinTransformer  | SwinTransformer_large_patch4_window7_224_22kto1k  SwinTransformer_large_patch4_window12_384_22kto1k  |
|                   |  SwinTransformer_base_patch4_window7_224_22kto1k  SwinTransformer_base_patch4_window12_384_22kto1k   |
|                   |          SwinTransformer_base_patch4_window12_384  SwinTransformer_base_patch4_window7_224           |
|                   |          SwinTransformer_small_patch4_window7_224  SwinTransformer_tiny_patch4_window7_224           |
|       Twins       |           pcpvt_small  pcpvt_base  pcpvt_large  alt_gvt_small  alt_gvt_base  alt_gvt_large           |
|        VGG        |                                      VGG11  VGG13  VGG16  VGG19                                      |
| VisionTransformer |       ViT_base_patch16_224  ViT_base_patch16_384  ViT_base_patch32_384  ViT_large_patch16_224        |
|                   |                 ViT_large_patch16_384  ViT_large_patch32_384  ViT_small_patch16_224                  |
|      Xception     |              Xception41  Xception41_deeplab  Xception65  Xception65_deeplab  Xception71              |
+-------------------+------------------------------------------------------------------------------------------------------+
                                                                                                    Powered by PaddlePaddle!
----------------------------------------------------------------------------------------------------------------------------
2022-06-15 18:33:10 [INFO]      Model[STANet] loaded.
---    fused 0 elementwise_add with relu activation
---    fused 0 elementwise_add with tanh activation
---    fused 0 elementwise_add with leaky_relu activation
---    fused 0 elementwise_add with swish activation
---    fused 0 elementwise_add with hardswish activation
---    fused 0 elementwise_add with sqrt activation
---    fused 0 elementwise_add with abs activation
---    fused 0 elementwise_add with clip activation
---    fused 0 elementwise_add with gelu activation
---    fused 0 elementwise_add with relu6 activation
---    fused 0 elementwise_add with sigmoid activation
---    fused 0 elementwise_sub with relu activation
---    fused 0 elementwise_sub with tanh activation
---    fused 0 elementwise_sub with leaky_relu activation
---    fused 0 elementwise_sub with swish activation
---    fused 0 elementwise_sub with hardswish activation
---    fused 0 elementwise_sub with sqrt activation
---    fused 1 elementwise_sub with abs activation
---    fused 0 elementwise_sub with clip activation
---    fused 0 elementwise_sub with gelu activation
---    fused 0 elementwise_sub with relu6 activation
---    fused 0 elementwise_sub with sigmoid activation
---    fused 0 elementwise_mul with relu activation
---    fused 0 elementwise_mul with tanh activation
---    fused 0 elementwise_mul with leaky_relu activation
---    fused 0 elementwise_mul with swish activation
---    fused 0 elementwise_mul with hardswish activation
---    fused 0 elementwise_mul with sqrt activation
---    fused 0 elementwise_mul with abs activation
---    fused 0 elementwise_mul with clip activation
---    fused 0 elementwise_mul with gelu activation
---    fused 0 elementwise_mul with relu6 activation
---    fused 0 elementwise_mul with sigmoid activation
total file number is 16
------------------ Inference Time Info ----------------------
total_time(ms): 1012.8, img_num: 1, batch_size: 1
average latency time(ms): 1012.80, QPS: 0.987362
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 1003.40, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 678.9, img_num: 1, batch_size: 1
average latency time(ms): 678.90, QPS: 1.472971
preprocess_time_per_im(ms): 9.20, inference_time_per_batch(ms): 669.40, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 885.6, img_num: 1, batch_size: 1
average latency time(ms): 885.60, QPS: 1.129178
preprocess_time_per_im(ms): 8.80, inference_time_per_batch(ms): 876.50, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 802.0, img_num: 1, batch_size: 1
average latency time(ms): 802.00, QPS: 1.246883
preprocess_time_per_im(ms): 9.30, inference_time_per_batch(ms): 792.10, postprocess_time_per_im(ms): 0.60
------------------ Inference Time Info ----------------------
total_time(ms): 669.1, img_num: 1, batch_size: 1
average latency time(ms): 669.10, QPS: 1.494545
preprocess_time_per_im(ms): 9.70, inference_time_per_batch(ms): 659.10, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 671.4, img_num: 1, batch_size: 1
average latency time(ms): 671.40, QPS: 1.489425
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 662.20, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 687.0, img_num: 1, batch_size: 1
average latency time(ms): 687.00, QPS: 1.455604
preprocess_time_per_im(ms): 8.80, inference_time_per_batch(ms): 677.70, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 845.7, img_num: 1, batch_size: 1
average latency time(ms): 845.70, QPS: 1.182452
preprocess_time_per_im(ms): 9.70, inference_time_per_batch(ms): 835.70, postprocess_time_per_im(ms): 0.30
------------------ Inference Time Info ----------------------
total_time(ms): 794.5, img_num: 1, batch_size: 1
average latency time(ms): 794.50, QPS: 1.258653
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 785.20, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 951.6, img_num: 1, batch_size: 1
average latency time(ms): 951.60, QPS: 1.050862
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 942.20, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 930.0, img_num: 1, batch_size: 1
average latency time(ms): 930.00, QPS: 1.075269
preprocess_time_per_im(ms): 9.50, inference_time_per_batch(ms): 920.10, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 1092.5, img_num: 1, batch_size: 1
average latency time(ms): 1092.50, QPS: 0.915332
preprocess_time_per_im(ms): 8.80, inference_time_per_batch(ms): 1083.20, postprocess_time_per_im(ms): 0.50
------------------ Inference Time Info ----------------------
total_time(ms): 949.6, img_num: 1, batch_size: 1
average latency time(ms): 949.60, QPS: 1.053075
preprocess_time_per_im(ms): 9.30, inference_time_per_batch(ms): 939.60, postprocess_time_per_im(ms): 0.70
------------------ Inference Time Info ----------------------
total_time(ms): 996.0, img_num: 1, batch_size: 1
average latency time(ms): 996.00, QPS: 1.004016
preprocess_time_per_im(ms): 9.60, inference_time_per_batch(ms): 986.00, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 948.4, img_num: 1, batch_size: 1
average latency time(ms): 948.40, QPS: 1.054407
preprocess_time_per_im(ms): 9.60, inference_time_per_batch(ms): 938.40, postprocess_time_per_im(ms): 0.40
------------------ Inference Time Info ----------------------
total_time(ms): 820.6, img_num: 1, batch_size: 1
average latency time(ms): 820.60, QPS: 1.218621
preprocess_time_per_im(ms): 8.90, inference_time_per_batch(ms): 811.30, postprocess_time_per_im(ms): 0.40
 Run successfully with command - python3.7  ./STANET_Paddle/tutorials/infer/stanet_infer.py   --infer_dir=./STANET_Paddle/test_tipc/result/inference_model/ --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output --use_gpu=False --enable_mkldnn=False --cpu_threads=2 --enable_benchmark=True --use_tensorrt=False         > ./test_tipc/output/STANET/python_infer_cpu_usemkldnn_False_threads_2_precision_null_batchsize_False.log 2>&1 !  
aistudio@jupyter-2315405-4166493:~$ 

```
## 6.代码结构与详细说明

```
StaNet-Paddle
├── deploy               # 部署相关的文档和脚本
├── docs                 # 整个项目图片
├── output               # 输出的VDL日志
├── test_tipc            # tipc程序相关
├── paddlers  
│     ├── custom_models  # 自定义网络模型代码
│     ├── datasets       # 数据加载相关代码
│     ├── models         # 套件网络模型代码
│     ├── tasks          # 相关任务代码
│     ├── tools          # 相关脚本
│     ├── transforms     # 数据处理及增强相关代码
│     └── utils          # 各种实用程序文件
├── tools                # 用于处理遥感数据的脚本
└── tutorials
      └── train          # 模型训练
      └── eval           # 模型评估和TIPC训练
      └── infer          # 模型推理
      └── predict        # 动态模型预测

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| STANET |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 遥感图像变化检测 |

## 8.说明
感谢百度提供的算力，以及举办的本场比赛，让我增强对paddle的熟练度，加深对变化检测模型的理解，加深对模型压缩的理解！

## 9.参考
部分功能参考 [PaddleRS 手把手教你PaddleRS实现变化检测](https://aistudio.baidu.com/aistudio/projectdetail/3737991?channelType=0&channel=0)

部分功能参考 [基于Paddle复现SNUNet-CD](https://github.com/kongdebug/SNUNet-Paddle)，

## 10.竞赛过程

## 对Paddle的复现项目的修正

首先，这篇文章对我，做模型挑战的帮助很大，给了我很多思路[《产业级SOTA模型优化指南》](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/pp-series)，我按照指南一步一步走。

 esnet在imagenet的精度为ImageNet/Acc 0.7392，原模型使用的backbone为res18精度为ImageNet/Acc 0.7098，backbone的特征提取能力增强，模型的大小也大大降低，所以我选它作为backbone。我这次只用到了其中的cov1和blocks两个组件。其他组件例如maxpool，我认为可能对基于语义分割的任务意义不大，所以我去掉了（注意要将模型中用不到的部分去掉，来减少参数量）。
 
模型训练很快就出现过拟合现象，我选择使用多种数据增广方式来缓解 。例如RandomHorizontalFlip、RandomVerticalFlip、MixupImage、RandomDistort、RandomBlur、RandomSwap。

原模型使用backbone的方式，是将backbone的多尺寸输出的c统一降低到90，之后尺寸放大到相同的大小，这带来了 一定的精度损失，我修改这一点通过1*1卷积根据输出的维度，处理后仍为相同的维度。

## 待完善的内容


知识蒸馏我认为是种很好的降低模型大小的方案，原模型因为模型大小还未发挥全部的实力，增大模型的参数，后知识蒸馏可能会带来不错的效果。


