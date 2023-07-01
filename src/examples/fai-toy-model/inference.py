import os
from torchvision.utils import save_image
import torch
from torchvision import transforms
import torchvision

import feather as ftr
from toy_system import step1, step2

if __name__ == "__main__":

    if not os.path.exists("samples"):
        os.mkdir("samples/")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    data_sample = torch.utils.data.DataLoader(set, batch_size=1,
                                              shuffle=True, num_workers=2)
    data = next(iter(data_sample))
    image, label = data

    # Run through the feather system
    gen_image, image_prediction = ftr.build(steps=[step1, step2], inputs=[image, label], interactive=False)

    print("Saving image...")
    save_image(gen_image.view(1, 1, 32, 32), 'samples/sample_' + str(image_prediction.item()) + '.png')
    print("Image saved!")
    print()
