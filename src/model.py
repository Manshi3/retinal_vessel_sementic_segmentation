import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

def prepare_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

# def prepare_model(num_classes=2):
#     model = deeplabv3_resnet101(weights='DEFAULT')
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
#     return model

if __name__ == '__main__':
    model = prepare_model()
    print(model)