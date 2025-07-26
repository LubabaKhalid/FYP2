import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from torchvision.models import resnet18
from torch import nn

# Simple 2-class ResNet classifier
class HandwrittenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: printed, handwritten

    def forward(self, x):
        return self.model(x)

# Load a fine-tuned model (you need to train it first)
model = HandwrittenClassifier()
model.load_state_dict(torch.load("handwritten_vs_printed.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def is_handwritten(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred == 1  # 0: printed, 1: handwritten
