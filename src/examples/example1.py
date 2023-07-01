import feather as ftr
from feather import helpers
import os
import io
import numpy as np

class ExampleCustomClass:
    def __init__(self, value):
        self.some_data = value

def Init():
    image_picker = ftr.File.Upload(types=["images"], title="my image", description="some description")
    text_view = ftr.Text.View(["hello1", "hello2"])

    textbox = ftr.Text.In(default_text=["enter text1", "enter text2"],
                          title="Ciello", description="Again?", num_inputs=2)

    my_opaque = np.arange(24).reshape(2,3,4)
    return image_picker, text_view, textbox, my_opaque

def step1(image_picker, text_view, textbox, my_opaque):
    print(my_opaque)
    # text_picker = ftr.File.Upload(types=[".txt"], title="my text", description="some description")
    textbox = ftr.Text.In(default_text=["enter text1", "enter text2"],
                          title="Ciello", description="Again?", num_inputs=2)
    txtView = ftr.Text.View(output_text="This is readonly data")
    fileDown = ftr.File.Download(title="file down", files=["This is the data I want to save"])
    # singleList = ftr.List.SelectOne(title="single list", items=["a", "b", "c"])
    # multiList = ftr.List.SelectMulti(title="my multi", items=[("a", False), ("b", True)])

    # image1 = np.zeros((4,4,3), dtype=np.uint8)
    # imageOne = ftr.Image.WithSelectOne(images=[{"name": "my image name", "data": image1}, {
    #                                    "name": "image2.jpg", "data": image1}], lists=[["cat", "dog"], ["lion", "eagle"]])

    # imageMulti = ftr.Image.WithSelectMulti(images=image1, lists=[["cat", ("Dog", True)]])

    # imageAndText = ftr.Image.WithTextIn(images=image1, default_text=["Hello"])
    # imageView = ftr.Image.View(images=image1, output_text=["Hello"])

    # docView = ftr.Document.View(
    #     documents=[{"name": "my doc name", "data": "document data"}], output_text=["Some output text"])
    # docText = ftr.Document.WithTextIn(
    #     documents=[{"name": "my doc name", "data": "document data"}], default_text=["Some output text"])
    # return (text_picker, textbox, txtView, fileDown, singleList, multiList, imageOne, imageMulti, imageAndText, imageView, docView, docText)
    return [textbox, txtView, fileDown]

@ftr.step(title="My entry step", description="my description")
def PreProcess(textbox, txtView, fileDown):
    print("Here")
    f = open("src/examples/data/file.txt", "r")
    fileData = f.read()
    f.close()

    txt = ftr.Text.View(text=fileData)
    #print("Image Picker (raw):", image_picker.get())
    #print("Image Picker: (images; only filedata)", image_picker.get("images", return_only_filedata=True))
    print("Text Picker:", text_picker.get())
    print("Text Picker: (only filedata)", text_picker.get_only_filedata())
    print("Textbox:", textbox.get())
    print("singleList:", singleList.get())
    print("singleList (with return_index):", singleList.get(return_index=True))
    print("multiList:", multiList.get())
    print("multiList (with return_all):", multiList.get(return_all=True))
    #print("imageOne:", imageOne.get())
    #print("imageOne (with return_indices):", imageOne.get(return_indices=True))
    #print("imageMulti:", imageMulti.get())
    #print("imageMulti (with return_all):", imageMulti.get(return_all=True))
    #print("imageAndText:", imageAndText.get())
    print("docText:", docText.get())

    random_text = ExampleCustomClass("Hello from my class")
    return [text_picker, textbox, txt, fileDown, singleList, multiList, imageOne, imageMulti, imageAndText, imageView, docView, docText, random_text
]

def SuperEvaluator(image_picker, text_picker, textbox, txt, fileDown, singleList, multiList, imageOne, imageMulti, imageAndText, imageView, docView, docText, random_text):
    print("Random text:", random_text.some_data, type(random_text))
    return [image_picker,
            text_picker,
            textbox,
            txt,
            fileDown,
            singleList,
            multiList,
            imageOne,
            imageMulti,
            imageAndText,
            imageView,
            docView,
            docText]


if __name__ == "__main__":
    print("Building Feather")

    # interactive -> devmode
    bundle = ftr.bundle(code_files=[__file__], model_files=["src/examples/data/file.txt"])
    ftr.build(name="3rd System",
              init=Init, steps=[step1, PreProcess, SuperEvaluator],
              file_bundle=bundle)
