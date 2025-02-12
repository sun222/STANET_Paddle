./STANET_Paddle/paddlers/models/ppgan/faceutils/face_detection/detection/blazeface/blazeface_detector.py                                                                                                                                                                                                                                                                                                                                                                                                                        ./STANET_Paddle/paddlers/models/ppgan/faceutils/face_detection/detection/blazeface/blazeface_detecto0000644 0001750 0001750 00000004771 14237127172 034266  0                                                                                                    ustar   aistudio                        aistudio                                                                                                                                                                                                               #   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
from paddle.utils.download import get_weights_path_from_url

from ..core import FaceDetector

from .net_blazeface import BlazeFace
from .detect import *

blazeface_weights = 'https://paddlegan.bj.bcebos.com/models/blazeface.pdparams'
blazeface_anchors = 'https://paddlegan.bj.bcebos.com/models/anchors.npy'


class BlazeFaceDetector(FaceDetector):
    def __init__(self,
                 path_to_detector=None,
                 path_to_anchor=None,
                 verbose=False,
                 min_score_thresh=0.5,
                 min_suppression_threshold=0.3):
        super(BlazeFaceDetector, self).__init__(verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights_path = get_weights_path_from_url(blazeface_weights)
            model_weights = paddle.load(model_weights_path)
            model_anchors = np.load(
                ge