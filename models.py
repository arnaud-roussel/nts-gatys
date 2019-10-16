import torchvision.models as models
import torch.nn as nn


class VGG16(nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features[:23]

    def forward(self, x):
        results = []
        out = x
        for i in range(23):
            out = self.vgg[i](out)
            if i in [3, 8, 15, 22]:
                results.append(out)
        return results
