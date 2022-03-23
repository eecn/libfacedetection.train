# config.py

# 网络配置
'''
网络名称
四个特征图中相对于输入图像的anchor的大小
四个特征图相对于输入图像的下采样倍数
计算anchor和真实边界框的偏移量的常数
clip 指示是否针对生成的anchor进行截断
'''
cfg = {
    'name': 'YuFaceDetectNet',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
}
