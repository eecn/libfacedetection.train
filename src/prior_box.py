import sys
import torch
import numpy as np
from math import ceil
from itertools import product as product

# 对输入图像的多尺度特征图生成anchor
# 输出形式为cx cy w h 当输入图像比anchor中设置的最大边界小时 会存在四个数字大于1的情况
class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        #self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

        for ii in range(4):
            if(self.steps[ii] != pow(2,(ii+3))):
                print("steps must be [8,16,32,64]")
                sys.exit()

        self.feature_map_2th = [int(int((self.image_size[0] + 1) / 2) / 2),
                                int(int((self.image_size[1] + 1) / 2) / 2)]
        self.feature_map_3th = [int(self.feature_map_2th[0] / 2),
                                int(self.feature_map_2th[1] / 2)]
        self.feature_map_4th = [int(self.feature_map_3th[0] / 2),
                                int(self.feature_map_3th[1] / 2)]
        self.feature_map_5th = [int(self.feature_map_4th[0] / 2),
                                int(self.feature_map_4th[1] / 2)]
        self.feature_map_6th = [int(self.feature_map_5th[0] / 2),
                                int(self.feature_map_5th[1] / 2)]

        self.feature_maps = [self.feature_map_3th, self.feature_map_4th,
                             self.feature_map_5th, self.feature_map_6th]

    def forward(self):
        anchors = []
        # 遍历每一个特征图
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 遍历特征图上每一个像素
            for i, j in product(range(f[0]), range(f[1])):
                # 遍历每一个尺度
                for min_size in min_sizes:
                    # 最小尺寸相对原图缩放到0-1之间
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]

                    # 特征图上的像素坐标x下采样倍数-->输入图像上坐标值  / 输入图像尺寸 归一化到0-1之间
                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
if __name__ == "__main__":
    # tasks.task1.config
    cfg = {
        'name': 'YuFaceDetectNet',
        'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
        'steps': [8, 16, 32, 64],
        'variance': [0.1, 0.2],
        'clip': False,
    }


    priorBox = PriorBox(cfg,(640,480))
    #priorBox = PriorBox(cfg, (320, 240))
    priors = priorBox.forward()
    print(priors.shape)
