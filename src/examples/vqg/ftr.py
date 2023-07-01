import os

import PIL.Image as Image
import feather as ftr
from extract_image_features import get_features_and_object_labels
import glob
from run_vqg import run_vqg
import pickle


def init():
    uploader = ftr.File.Upload(["images"], title="Upload your images")
    return [uploader]


# @ftr.step(title="Choose your images to upload!")
def step1(uploader):
    images = uploader.get("images")
    img_list = [x["data"] for x in images]
    normalised_boxes, normalised_features, full_class_labels = get_features_and_object_labels(img_list)
    full_class_labels = [lst[:6] for lst in full_class_labels]
    print(full_class_labels)
    image_select = ftr.Image.WithSelectMulti(images=images, lists=full_class_labels,
                                             title="Select objects you want a question on",
                                             description="Recommended to select only 2 objects")
    return image_select, normalised_boxes, normalised_features


@ftr.step(title="Select objects to condition a question generator on", description="Recommended to select only 2 objects")
def step2(image_select, normalised_boxes, normalised_features):
    selected_options = image_select.get()
    print(selected_options)
    images = image_select.images
    img_list = [x["data"] for x in images]
    print(images[0].items())
    print("selected_options", selected_options)
    questions, sampled_categories = run_vqg(img_list, normalised_boxes, normalised_features, selected_options)
    output_text = []
    for q, cat, selected_objects in zip(questions, sampled_categories, selected_options):
        output_text.append("""Sampled category: {}; 
        Selected Objects: {}; 
        Question: {}""".format(cat, " ".join(selected_objects), q))
    return [ftr.Image.View(images, output_text)]


if __name__ == "__main__":

    code_files, model_files = ftr.helpers.get_all_files_from_curr_dir()
    bundle = ftr.bundle(code_files=code_files, model_files=model_files)
    ftr.build(name="OD",
              description="test system",
              init=init, steps=[step1, step2],
              file_bundle=bundle)
