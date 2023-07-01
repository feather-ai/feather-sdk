import PIL.Image as Image
import torch
import torchvision.models as models
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                ])
resnet152 = models.resnet152(pretrained=True).eval()

all_labels = open("./imagenet_classes.txt", "r").read().splitlines()

def run_model(images):
    transformed_images = [transform(image) for image in images]
    transformed_images = torch.stack(transformed_images)
    outputs = resnet152(transformed_images)
    labels = []
    for image in outputs:
        pred = torch.argmax(image).item()
        labels.append(all_labels[pred])
    return labels

if __name__ == "__main__":
    image = Image.open("./me2.jpg")
    image = [image] * 2
    print(run_model(image))
