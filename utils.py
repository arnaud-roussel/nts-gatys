import torchvision.transforms as transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def normalize():
    return transforms.Normalize(mean=MEAN, std=STD)


def denormalize(x):
    for c in range(3):
        x[c].mul_(STD[c]).add_(MEAN[c])
    return x


def gram_matrix(x):
    N, C, H, W = x.shape
    x = x.view(N, C, H * W)
    return x.bmm(x.transpose(1, 2)) / (C * H * W)