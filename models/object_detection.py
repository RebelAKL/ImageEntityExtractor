import torch
from torchvision import models, transforms
from PIL import Image

def detect_objects(image_path):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(image_tensor)
    

    return predictions
