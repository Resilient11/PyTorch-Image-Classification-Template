from torchvision import models
import torch.nn as nn

def get_model(name: str, num_classes: int):
    name = name.lower()

    # alexnet
    if name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # convnext
    elif name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "convnext_small":
        model = models.convnext_small(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "convnext_base":
        model = models.convnext_base(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "convnext_large":
        model = models.convnext_large(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # densenet
    elif name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "densenet169":
        model = models.densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "densenet201":
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # efficientnet
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b5":
        model = models.efficientnet_b5(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b6":
        model = models.efficientnet_b6(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # googlenet
    elif name == "googlenet":
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # inception
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # mnasnet
    elif name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mnasnet1_3":
        model = models.mnasnet1_3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # mobilenet
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # regnet
    elif name == "regnet_y_400mf":
        model = models.regnet_y_400mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "regnet_y_800mf":
        model = models.regnet_y_800mf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "regnet_y_1_6gf":
        model = models.regnet_y_1_6gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "regnet_y_3_2gf":
        model = models.regnet_y_3_2gf(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # resnet
    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet101":
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet152":
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # shufflenet
    elif name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "shufflenet_v2_x1_5":
        model = models.shufflenet_v2_x1_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "shufflenet_v2_x2_0":
        model = models.shufflenet_v2_x2_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # squeezenet
    elif name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    elif name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    # vgg
    elif name == "vgg11":
        model = models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg13":
        model = models.vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # vision transformer
    elif name == "vit_b_16":
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # swin transformer
    elif name == "swin_t":
        model = models.swin_t(pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    # maxvit
    elif name == "maxvit_t":
        model = models.maxvit_t(pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {name}")

    return model
