# PyTorch-Image-Classification-Template Help Document

## 项目介绍
`PyTorch-Image-Classification-Template` 是一个基于 PyTorch 和 torchvision 预训练模型的通用图像分类模板，支持快速实验多种主流 CNN/ViT 模型。

## 项目总体文件结构
```
PyTorch-Image-Classification-Template/
├── config.py              # 模型配置文件
├── divide_dataset.py      # 数据集划分脚本
├── model_init.py          # 模型初始化文件
├── train.py               # 模型训练脚本
├── requirements.txt       # 项目依赖文件
```

### config.py
用于配置模型训练的参数，包括数据集路径、模型选择、训练超参数等。

### divide_dataset.py
用于将数据集划分为训练集、验证集和测试集（按照8:1:1的比例），确保数据集按类别均匀分布。

### model_init.py
用于初始化模型，从 torchvision 中加载预训练模型并进行相应的修改以适应新的分类任务。

### train.py
项目的核心训练脚本，包含了数据加载、模型训练、验证和保存的全过程。

###requirements.txt
 可以使用命令`pip install -r requirements.txt`安装项目所需依赖。

## 数据集准备及使用 divide_dataset.py 划分数据集
请将你的图像数据集按照以下结构组织：
```
dataset/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        img1.jpg
        img2.jpg
        ...
```

使用 `divide_dataset.py` 划分数据集：
```bash
python divide_dataset.py --data_dir path_to_your_dataset
```
参数说明：
- `--data_dir`：原始数据集路径

运行上述命令后，数据集将被划分为训练集、验证集和测试集，结构如下：
```
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    val/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    test/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    
```

## 模型配置（config.py）
在 `config.py` 文件中，你可以配置模型训练的参数，包括但不限于：
- 数据集路径
- 模型名称（如 resnet50, vgg16, vit_b_16 等）
- 批处理大小（BATCH_SIZE）
- 训练轮数（EPOCHS）
- 学习率（INIT_LR）

## 开始训练
在终端运行以下命令开始训练：
```bash
python train.py --config config.py
```
参数说明：
- `--config`：配置文件路径

## 控制台输出及整体输出文件
在训练过程中，控制台将输出训练的实时信息，包括训练损失、验证损失、准确率等。示例输出如下：
```
训练标签类别: ['class1', 'class2', 'class3', 'class4', 'class5']

[Train][Epoch 001] Acc: 84.52%
[Val][Epoch 001] Acc: 93.86% | P: 0.938 | R: 0.939 | F1: 0.938
最佳模型已保存（准确率: 93.86%）

[Train][Epoch 002] Acc: 94.12%
[Val][Epoch 002] Acc: 95.64% | P: 0.956 | R: 0.956 | F1: 0.956
最佳模型已保存（准确率: 95.64%）
...
```

训练结束后，模型将被保存到指定的路径，默认保存至`output`文件夹下。

## 训练结束后的项目文件结构
训练结束后，项目的文件结构可能如下所示：
```
PyTorch-Image-Classification-Template/
├── data/                          # 数据集
│   ├── test    
│   ├── train        
│   ├── val
├── output/                        # 输出目录
│   ├── mobilenet_v3_large/        # 模型一
|   │   ├── metrics.png            # 训练指标图像
|   │   ├── mobilenet_v3_large_50epochs.pth              # 模型权重文件
│   ├── shufflenet_v2_x1_0/        # 模型二
|   │   ├── metrics.png            # 训练指标图像
|   │   ├──shufflenet_v2_x1_0_50epochs.pth               # 模型权重文件
├── README.md                      # 项目介绍和基本使用说明
├── config.py                      # 模型配置文件
├── divide_dataset.py              # 数据集划分脚本
├── model_init.py                  # 模型初始化文件
├── train.py                       # 模型训练脚本
├── requirements.txt               # 项目依赖文件
├── model.pth                      # 训练好的模型文件

```

## 许可证
本项目基于 MIT 许可证开源，详情请参考 [LICENSE](LICENSE) 文件。
