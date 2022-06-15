# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions ands
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .backbones import esnet 
from .backbones import resnet
from .backbones import mobilenet 
from .backbones import resnet_2 
from .layers import Conv1x1, Conv3x3, get_norm_layer, Identity
from .param_init import KaimingInitMixin
import paddleclas
from paddle.utils.download import get_weights_path_from_url


class STANet(nn.Layer):
    """
    The STANet implementation based on PaddlePaddle.

    The original article refers to
        H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"
        (https://www.mdpi.com/2072-4292/12/10/1662).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        att_type (str, optional): The attention module used in the model. Options are 'PAM' and 'BAM'. Default: 'BAM'.
        ds_factor (int, optional): The downsampling factor of the attention modules. When `ds_factor` is set to values 
            greater than 1, the input features will first be processed by an average pooling layer with the kernel size of 
            `ds_factor`, before being used to calculate the attention scores. Default: 1.

    Raises:
        ValueError: When `att_type` has an illeagal value (unsupported attention type).
    """

    def __init__(self, in_channels, num_classes, att_type='BAM', ds_factor=1,backbonetype="resnet18"):
        super(STANet, self).__init__()

        WIDTH = 64
        self.extract = build_feat_extractor(in_ch=in_channels, width=WIDTH,backbonetype=backbonetype)
        
        self.attend = build_sta_module(
            in_ch=WIDTH, att_type=att_type, ds=ds_factor)
        self.conv_out = nn.Sequential(
            Conv3x3(
                WIDTH, WIDTH, norm=True, act=True),
            Conv3x3(WIDTH, num_classes))

        self.init_weight()
    def forward(self, t1, t2):
        f1 = self.extract(t1)
        f2 = self.extract(t2)

        f1, f2 = self.attend(f1, f2)

        y = paddle.abs(f1 - f2)
        y = F.interpolate(
            y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        pred = self.conv_out(y)
        return [pred]

    def init_weight(self):
        # Do nothing here as the encoder and decoder weights have already been initialized.
        # Note however that currently self.attend and self.conv_out use the default initilization method.
        pass


def build_feat_extractor(in_ch, width,backbonetype):

    if  backbonetype=="MobileNetV3_small_x1_0":
        return nn.Sequential(Backbone(in_ch, backbonetype), Decoder(width,infeat=[16,24,48,96]))



    elif  backbonetype=="MobileNetV3_small_x1_25":
        return nn.Sequential(Backbone(in_ch, backbonetype), Decoder(width,infeat=[24,32,64,120]))


    elif  backbonetype=="ESNet_x1_0":
        return nn.Sequential(Backbone(in_ch, backbonetype), Decoder(width,infeat=[120,232,464]))
    else:
        return nn.Sequential(Backbone(in_ch, backbonetype), Decoder(width,infeat=[64,128,256,512]))



def build_sta_module(in_ch, att_type, ds):
    if att_type == 'BAM':
        return Attention(BAM(in_ch, ds))
    elif att_type == 'PAM':
        return Attention(PAM(in_ch, ds))
    else:
        raise ValueError


class Backbone(nn.Layer, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super(Backbone, self).__init__()
        self.basenet=0
        self.arch=arch
        if arch == 'resnet18':
            self.basenet= resnet_2.ResNet18( pretrained=pretrained, use_ssld=False ,return_stages=True)  
           
            # self.basenet = resnet.resnet18(
            #     pretrained=pretrained,
            #     strides=strides,
            #     norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.basenet = resnet.resnet34(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.basenet = resnet.resnet50(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())

        elif arch == 'MobileNetV3_small_x1_0':
            self.basenet=mobilenet.MobileNetV3_small_x1_0( pretrained=pretrained, use_ssld=False ,return_stages=True)    

        elif arch == 'MobileNetV3_small_x1_25':
            self.basenet=mobilenet.MobileNetV3_small_x1_25( pretrained=pretrained, use_ssld=False ,return_stages=True)   

        elif arch == 'ESNet_x1_0':
            self.basenet=esnet.ESNet_x1_0(pretrained=pretrained,return_stages=True)   


        else:
            raise ValueError

        self._trim_resnet()

        # if in_ch != 3:
        #     self.resnet.conv1 = nn.Conv2D(
        #         in_ch,
        #         64,
        #         kernel_size=7,
        #         stride=strides[0],
        #         padding=3,
        #         bias_attr=False)

        if not pretrained:
            self.init_weight()

    def forward(self, x):
        # if self.arch == 'resnet18':
        #     x = self.basenet.conv1(x)
        #     x = self.basenet.bn1(x)
        #     x = self.basenet.relu(x)
        #     x = self.basenet.maxpool(x)

        #     x1 = self.basenet.layer1(x)
        #     x2 = self.basenet.layer2(x1)
        #     x3 = self.basenet.layer3(x2)
        #     x4 = self.basenet.layer4(x3)
        #     return x1, x2, x3, x4  
        # # x=self.basenet.conv(x)
        # y=self.basenet(x)
        # # print(y)
        # feat=[]
        # for key in y:
        #     if key!="output":
        #         feat.append(y[key])
        # return feat

        # x=self.basenet.conv(x)

        y=self.basenet.blocks
        x = self.basenet.conv1(x)
        feat=[]
        x=y[0](x)
        x=y[1](x) 
        x=y[2](x) 
        feat.append(x) 
        x=y[3](x)
        x=y[4](x)  
        x=y[5](x)
        x=y[6](x)  
        x=y[7](x) 
        x=y[8](x)
        x=y[9](x) 
        feat.append(x)
        x=y[10](x) 
        x=y[11](x)  
        x=y[12](x) 
        feat.append(x) 
        return feat


        # y=self.basenet.blocks
        # x = self.basenet.stem(x)
        # x = self.basenet.max_pool(x)
        # feat=[]
        # x=y[0](x)
        # x=y[1](x) 
        # feat.append(x)    
        # x=y[2](x)  
        # x=y[3](x)
        # feat.append(x)   


        # x=y[4](x)  
        # x=y[5](x) 
        # feat.append(x)       
        # x=y[6](x)  
        # x=y[7](x) 
        # feat.append(x) 
        # return feat

    def _trim_resnet(self):

        self.basenet.max_pool= Identity()
        self.basenet.conv2 = Identity()
        self.basenet.avg_pool = Identity()
        self.basenet.last_conv = Identity()
        self.basenet.hardswish = Identity()
        self.basenet.dropout = Identity()
        self.basenet.flatten = Identity()
        self.basenet.fc = Identity()


class Decoder(nn.Layer, KaimingInitMixin):
    def __init__(self, f_ch,infeat=[16,24,48,96]):
        super(Decoder, self).__init__()
        # self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        # self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        # self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        # self.dr4 = Conv1x1(512, 96, norm=True, act=True)


        self.dr1 = Conv1x1(infeat[0], 120, norm=True, act=True)
        self.dr2 = Conv1x1(infeat[1], 232, norm=True, act=True)
        self.dr3 = Conv1x1(infeat[2], 464, norm=True, act=True)


        self.conv_out = nn.Sequential(
            Conv3x3(
                816, 256, norm=True, act=True),
            nn.Dropout(0.5),
            Conv1x1(
                256, f_ch, norm=True, act=True))

        self.init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])

        # f1 = F.interpolate(
        #     f1, size=[32,32], mode='bilinear', align_corners=True)
        f2 = F.interpolate(
            f2, size=[64,64], mode='bilinear', align_corners=True)
        f3 = F.interpolate(
            f3, size=[64,64], mode='bilinear', align_corners=True)


        x = paddle.concat([f1, f2, f3], axis=1)
        y = self.conv_out(x)

        return y


class BAM(nn.Layer):
    def __init__(self, in_ch, ds):
        super(BAM, self).__init__()

        self.ds = ds
        self.pool = nn.AvgPool2D(self.ds)

        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch)
        self.conv_k = Conv1x1(in_ch, self.key_ch)
        self.conv_v = Conv1x1(in_ch, self.val_ch)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        x = x.flatten(-2)
        x_rs = self.pool(x)

        b, c, h, w = paddle.shape(x_rs)
        query = self.conv_q(x_rs).reshape((b, -1, h * w)).transpose((0, 2, 1))
        key = self.conv_k(x_rs).reshape((b, -1, h * w))
        energy = paddle.matmul(query, key)
        energy = (self.key_ch**(-0.5)) * energy

        attention = self.softmax(energy)

        value = self.conv_v(x_rs).reshape((b, -1, w * h))

        out = paddle.matmul(value, attention.transpose((0, 2, 1)))
        out = out.reshape((b, c, h, w))

        out = F.interpolate(out, scale_factor=self.ds)
        out = out + x
        g1=tuple(out.shape[:-1])
        g2= tuple([out.shape[-1] // 2, 2])
        return out.reshape(g1 + g2)


class PAMBlock(nn.Layer):
    def __init__(self, in_ch, scale=1, ds=1):
        super(PAMBlock, self).__init__()

        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2D(self.ds)

        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch, norm=True)
        self.conv_k = Conv1x1(in_ch, self.key_ch, norm=True)
        self.conv_v = Conv1x1(in_ch, self.val_ch)

    def forward(self, x):
        x_rs = self.pool(x)

        # Get query, key, and value.
        query = self.conv_q(x_rs)
        key = self.conv_k(x_rs)
        value = self.conv_v(x_rs)

        # Split the whole image into subregions.
        b, c, h, w = x_rs.shape


        query = self._split_subregions(query)
        key = self._split_subregions(key)
        value = self._split_subregions(value)

        # Perform subregion-wise attention.
        out = self._attend(query, key, value)

   
        # Stack subregions to reconstruct the whole image.
        out = self._recons_whole(out, b, c, h, w)
        out = F.interpolate(out, scale_factor=self.ds)
        return out

    def _attend(self, query, key, value):
        energy = paddle.matmul(query.transpose((0, 2, 1)),
                            key)  # batch matrix multiplication
        energy = (self.key_ch**(-0.5)) * energy
        attention = F.softmax(energy, axis=-1)
        out = paddle.matmul(value, attention.transpose((0, 2, 1)))
        return out

    def _split_subregions(self, x):
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0
        x = x.reshape(
            (b, c, self.scale, h // self.scale, self.scale, w // self.scale))

        x = x.transpose((0, 2, 4, 1, 3, 5))
          
        x =x.reshape(
            (b * self.scale * self.scale, c, -1))
        return x

    def _recons_whole(self, x, b, c, h, w):
        x = x.reshape(
            (b, self.scale, self.scale, c, h // self.scale, w // self.scale))
        x = x.transpose((0, 3, 1, 4, 2, 5)).reshape((b, c, h, w))
        return x


class PAM(nn.Layer):
    def __init__(self, in_ch, ds, scales=(1, 2, 4, 8)):
        super(PAM, self).__init__()

        self.stages = nn.LayerList(
            [PAMBlock(
                in_ch, scale=s, ds=ds) for s in scales])
        self.conv_out = Conv1x1(in_ch * len(scales), in_ch, bias=False)

    def forward(self, x):
        x = x.flatten(-2)
        res = [stage(x) for stage in self.stages]


    
        out = self.conv_out(paddle.concat(res, axis=1))

        g1=tuple(out.shape[:-1])
        g2= tuple([out.shape[-1] // 2, 2])
        return out.reshape(g1 + g2)


class Attention(nn.Layer):
    def __init__(self, att):
        super(Attention, self).__init__()
        self.att = att

    def forward(self, x1, x2):
        x = paddle.stack([x1, x2], axis=-1)
        y = self.att(x)
        return y[..., 0], y[..., 1]
