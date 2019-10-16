import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
from utils import normalize, denormalize, gram_matrix
from models import VGG16

STYLE_SIZE = 512
LAMBDA_CONTENT = 1
LAMBDA_STYLE = 1e9
MAX_STEP = 250

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform_target = transforms.Compose([transforms.Resize(1024), transforms.ToTensor(), normalize()])
transform_style = transforms.Compose([transforms.Resize((STYLE_SIZE, STYLE_SIZE)),
                                      transforms.ToTensor(),
                                      normalize()])

x = Image.open('./target.jpg')
x = transform_target(x).unsqueeze(0).to(device)
output = x.clone()
output.requires_grad = True
vgg_model = VGG16().to(device).eval()

with torch.no_grad():
    style_image = Image.open('./candy.png')
    style_image = transform_style(style_image).unsqueeze(0).to(device)
    style_features = vgg_model(style_image)
    style_grams = [gram_matrix(f) for f in style_features]
    content_features = vgg_model(x)

optimizer = torch.optim.LBFGS([output])
loss = nn.MSELoss()

i = [0]
while i[0] < MAX_STEP:
    def closure():
        optimizer.zero_grad()
        output_features = vgg_model(output)
        output_gram = [gram_matrix(f) for f in output_features]
        loss_content = loss(output_features[1], content_features[1])
        loss_content *= LAMBDA_CONTENT
        loss_style = 0
        for k in range(4):
            loss_style += loss(output_gram[k], style_grams[k])
        loss_style *= LAMBDA_STYLE
        total_loss = loss_content + loss_style
        total_loss.backward()
        i[0] += 1
        sys.stdout.write(f'\r Iter {i[0]}: Content: {loss_content.item():.2f} '
                         f'Style: {loss_style.item():.2f} Total: {total_loss.item():.2f}')
        return total_loss
    optimizer.step(closure)

output = denormalize(output.squeeze(0)).clamp(0, 1)
output = (output.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(output).save('output.jpg')
