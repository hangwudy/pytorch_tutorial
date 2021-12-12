import torch
import torchvision
from PIL import Image
from torchvision import transforms

img = Image.open("demo.png").convert("RGB")
print(img.size)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trans = []
trans.extend(
    [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ]
)
trans = transforms.Compose(trans)

# img = trans(img)
traindir = "/media/hangwu/My Passport/dataset/Demo"
dataset = torchvision.datasets.ImageFolder(
    traindir,
    trans
)
for d in dataset:
    print(d)
