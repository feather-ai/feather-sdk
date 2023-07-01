import feather as ftr
from main import run_model


def init():
    uploader = ftr.File.Upload(["images"], title="Upload files for image classification")
    return uploader


def run_image_classification(uploader):
    # [{name: "image1.jpg", data: np.array}, {name: "image2.jpg", data: np.array}]
    images = uploader.get(format="images")
    image_data = [x["data"] for x in images]
    classifications = run_model(image_data)
    output = ftr.Image.View(images, classifications)
    return output


if __name__ == "__main__":
    bundle = ftr.bundle(code_files=[__file__, "main.py", "imagenet_classes.txt"])
    ftr.build(name="Simple Image Classification", description="A simple ResNet image classification model",
              init=init, steps=[run_image_classification], file_bundle=bundle)
