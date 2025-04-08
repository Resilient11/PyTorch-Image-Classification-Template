"""
MODEL_NAME 可以从此列表中取值
MODEL_NAME = [
    'alexnet',

    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',

    'densenet121', 'densenet161', 'densenet169', 'densenet201',

    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',

    'googlenet',

    'inception_v3',

    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',

    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',

    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf',
    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',

    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',

    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',

    'squeezenet1_0', 'squeezenet1_1',

    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',

    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',

    'swin_t', 'swin_s', 'swin_b',

    'maxvit_t', 'maxvit_s', 'maxvit_b', 'maxvit_l',
]
"""

MODEL_NAME = "mobilenet_v3_large"


DATASET_PATH = "./data"

# 图片尺寸（宽, 高） 部分模型有固定输入尺寸，如 Inception (299, 299)
IMAGE_SIZE = (224, 224)

# 批大小（batch size）
BATCH_SIZE = 64

# 训练轮数
EPOCHS = 50

# 初始学习率
INIT_LR = 0.01

# 图像标准化的均值与方差（ImageNet 预训练模型通用）
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 是否使用 AMP 混合精度加速训练（由 train.py 固定开启）
USE_AMP = True

# 保存输出文件的目录
OUTPUT_DIR = f"./output/{MODEL_NAME}"
# 保存模型文件的路径
OUTPUT_MODEL = f"{OUTPUT_DIR}/{MODEL_NAME}_{EPOCHS}epochs.pth"

