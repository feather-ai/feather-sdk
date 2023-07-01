import feather as ftr
from torchvision.utils import save_image
from cnn_trainer import CNNLightning
import torch
from torchvision import transforms
import torchvision
from cvae_trainer import CVAELightning

def onehot(labels, num_classes=10):
    targets = torch.zeros(labels.size(0), num_classes)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def step1(image, label):
    classes = ['plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Your GROUND truth image is off a", classes[label], ". The index of this class is", label.item())
    cnn = CNNLightning.load_from_checkpoint("lightning_logs/cnn/checkpoints/epoch=6-step=9999.ckpt")
    image_prediction_logits = cnn.cnn(image)
    image_prediction = torch.argmax(image_prediction_logits, -1)
    print("Your PREDICTED class is a", classes[image_prediction], ". The index of this class is", image_prediction.item())
    print("The predicted image has a numeric index of:", image_prediction.item())
    return [image_prediction]

def step2(image_prediction):
    print("Generating an image for the class index", image_prediction.item())
    cvae = CVAELightning.load_from_checkpoint("lightning_logs/cvae/checkpoints/epoch=3-step=7499.ckpt", args={})
    gen_image = cvae.cvae.sample(1, 0, labels=onehot(image_prediction))[0][0]
    return [gen_image, image_prediction]
