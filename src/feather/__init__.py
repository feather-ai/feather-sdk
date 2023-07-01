from io import BytesIO
from PIL import Image as PILImage
from feather.helpers import convert_to_JsonObject
from feather.featherlocal import webserver, systemrunner, publisher
from typing import Any, Callable, Literal, Optional, Tuple, Union, TypedDict
from typing import List as ListType
from feather import components
import os
import numpy as np


def build(name, init: Callable, steps: ListType[Callable],
          description="", file_bundle=None):
    """Build and optionally deploy a system to the feather service"""

    if os.environ.get("FEATHER_SERVICE_RUNNER") != None:
        raise RuntimeError("Calling ftr.Build from Runner")

    if init == None:
        raise ValueError("Init must be set to a valid init function")

    # Validate args
    if steps == None:
        raise ValueError("'None' steps provided")

    if len(steps) == 0:
        raise ValueError("No steps provided - you must provide at least 1 step")

    runner = systemrunner.SystemRunner(name, description, bootstrap=init, steps=steps, file_bundle=file_bundle)

    webserver.start_server(runner)


# Decorators

def step(title: str, description: str = None):
    def decorator(func):
        func._ftr_title = title
        func._ftr_description = description
        return func
    return decorator


# Public Bundle API

def bundle(code_files, model_files=[]):
    for f in code_files:
        if os.path.isfile(f) == False:
            raise ValueError("Can't find file {0} (cwd={1})".format(f, os.getcwd()))
    for f in model_files:
        if os.path.isfile(f) == False:
            raise ValueError("Can't find file {0} (cwd={1})".format(f, os.getcwd()))

    ret = publisher.Bundle(code_files=code_files, model_files=model_files)
    return ret


# Typed Dictionaries

class GenericFileType(TypedDict):
    name: str
    data: Union[str, bytes]  # b64 encoded


class ImageInputObj(TypedDict):
    name: str
    data: np.ndarray


ImageInputType = Union[ListType[ImageInputObj], ListType[np.ndarray]]


class DocumentInputObj(GenericFileType):
    name: str
    data: str


DocumentInputType = Union[ListType[DocumentInputObj], ListType[str]]


# Public Components API

class File:
    class Upload(components.FeatherComponent):
        def __init__(self, types: ListType[str],
                     min_files: int = 1,
                     max_files: int = 5,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.FileLoader(types=types, title=title, description=description,
                                                   min_files=min_files, max_files=max_files)

        def _convert_b64_to_array_(self, b64_image):
            pil_im = PILImage.open(BytesIO(b64_image))
            im_array = np.asarray(pil_im)
            return im_array

        def _decode_to_utf_(self, byte_string):
            return byte_string.decode("utf-8")

        def get(self, format="raw", return_only_filedata=False):
            if format == "raw":
                if return_only_filedata:
                    return self.get_only_filedata()
                return self.get_files()
            if format == "images":
                if return_only_filedata:
                    return self.get_image_only_filedata()
                return self.get_image_files()
            if format == "text":
                if return_only_filedata:
                    return self.get_text_only_filedata()
                return self.get_text_files()

            raise ValueError("File.Upload - get() format must be 'raw', 'text', or 'images', but got '{0}'".format(format))

        def get_text_only_filedata(self) -> ListType[str]:
            return [x.data.decode("utf-8") for x in self.component.files]

        def get_image_only_filedata(self) -> ListType[np.ndarray]:
            return [self._convert_b64_to_array_(x.data) for x in self.component.files]

        def get_only_filedata(self) -> ListType[Union[str, bytes]]:
            return [x.data for x in self.component.files]

        def get_text_files(self) -> ListType[DocumentInputObj]:
            return convert_to_JsonObject(self.component.files, self._decode_to_utf_)

        def get_image_files(self) -> ListType[ImageInputObj]:
            return convert_to_JsonObject(self.component.files, self._convert_b64_to_array_)
            # ret = []
            # for x in self.component.files:
            #     item = JsonObject()
            #     item.name = x.name
            #     item.data = self._convert_b64_to_array_(x.data)
            #     ret.append(item)
            # return ret

        def get_files(self) -> ListType[GenericFileType]:
            return self.component.files

    class Download(components.FeatherComponent):
        def __init__(self, files: ListType[Union[bytes, str, Any]],
                     output_filenames: Optional[ListType[str]] = None,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.FileDownload(
                files=files, output_filenames=output_filenames, title=title, description=description)

        @property
        def files(self):
            return self._get_files()

        def _get_files(self) -> ListType[GenericFileType]:
            return convert_to_JsonObject(self.component.files)

class Text:
    class In(components.FeatherComponent):
        def __init__(self, default_text: Optional[Union[str, ListType[str]]],
                     num_inputs: int = 1,
                     max_chars: Optional[int] = 256,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.TextBoxInput(default_text=default_text, title=title, description=description,
                                                     num_inputs=num_inputs, max_chars=max_chars)

        def get(self):
            return self.get_text()

        def get_text(self) -> ListType[str]:
            return self.component.text

    class View(components.FeatherComponent):
        def __init__(self, output_text: Union[str, ListType[str]],
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.TextLabel(text=output_text, title=title, description=description)
            
class List:
    class SelectOne(components.FeatherComponent):
        def __init__(self, items: ListType[str],
                     style: Literal["radio", "dropdown"] = "radio",
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.SingleSelectList(
                listItems=items, title=title, description=description, style=style)
            
        def get(self, return_index=False):
            if return_index:
                return self.get_selected_with_index()
            return self.get_selected()

        def get_selected_with_index(self) -> Tuple[str, int]:
            selected_string = self.component.items[self.component.selected_index]
            return (selected_string, self.component.selected_index)

        def get_selected(self) -> str:
            selected_string = self.component.items[self.component.selected_index]
            return selected_string

    class SelectMulti(components.FeatherComponent):
        def __init__(self, items: Union[ListType[str], ListType[Tuple[str, bool]]],
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.MultiSelectList(listItems=items, title=title, description=description)
            
        def get(self, return_all=False):
            if return_all:
                return self.get_all()
            return self.get_selected()

        def get_all(self) -> ListType[Tuple[str, bool]]:
            return self.component.items

        def get_selected(self) -> ListType[str]:
            selected_items = list(filter(lambda x: x[1] == True, self.component.items))
            return [item[0] for item in selected_items]


class Image:
    class WithSelectOne(components.FeatherComponent):
        def __init__(self, images: ImageInputType,
                     lists: ListType[ListType[str]],
                     style: Literal["radio", "dropdown"] = "radio",
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.ImageWithSingleSelect(
                images=images, lists=lists, title=title, description=description, style=style)
        
        @property
        def images(self):
            return self._get_images()

        def _selected_(self):
            selected = []
            for i, attribute_list in enumerate(self.component.attributes):
                selected_index = self.component.selected_indices[i]
                selected.append(attribute_list[selected_index])
            return selected

        def get(self, return_indices=False):
            if return_indices:
                return self.get_selected_with_index()
            return self.get_selected()

        def get_selected_with_index(self) -> ListType[Tuple[str, int]]:
            return list(zip(self._selected_(), self.component.selected_indices))

        def get_selected(self) -> ListType[str]:
            return self._selected_()

        def _get_images(self) -> ListType[ImageInputObj]:
            return convert_to_JsonObject(self.component.images)

    class WithSelectMulti(components.FeatherComponent):
        def __init__(self, images: ImageInputType,
                     lists: Union[ListType[ListType[str]], ListType[ListType[Tuple[str, bool]]]],
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.ImageWithMultiSelect(
                images=images, lists=lists, title=title, description=description)

        @property
        def images(self):
            return self._get_images()

        def _selected_(self):
            selected = []
            for i, attribute_list in enumerate(self.component.attributes):
                selected_items = list(filter(lambda x: x[1] == True, attribute_list))
                selected.append([item[0] for item in selected_items])
            return selected

        def get(self, return_all=False):
            if return_all:
                return self.get_all()
            return self.get_selected()

        def get_all(self) -> ListType[ListType[Tuple[str, bool]]]:
            return self.component.attributes

        def get_selected(self) -> ListType[ListType[str]]:
            return self._selected_()

        def _get_images(self) -> ListType[ImageInputObj]:
            return convert_to_JsonObject(self.component.images)

    class WithTextIn(components.FeatherComponent):
        def __init__(self, images: ImageInputType,
                     default_text: Optional[Union[str, ListType[str]]] = None,
                     max_chars: Optional[int] = 256,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.ImageWithTextIn(
                images=images, default_text=default_text, title=title, description=description, max_chars=max_chars)

        @property
        def images(self):
            return self._get_images()

        def get(self):
            return self.get_text()

        def get_text(self) -> ListType[str]:
            return self.component.text

        def _get_images(self) -> ListType[ImageInputObj]:
            return convert_to_JsonObject(self.component.images)

    class View(components.FeatherComponent):
        def __init__(self, images: ImageInputType,
                     output_text: Optional[ListType[str]] = None,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.ImageView(
                images=images, output_text=output_text, title=title, description=description)

        @property
        def images(self):
            return self._get_images()

        def _get_images(self) -> ListType[ImageInputObj]:
            return convert_to_JsonObject(self.component.images)

class Document(components.FeatherComponent):
    class WithTextIn(components.FeatherComponent):
        def __init__(self, documents: DocumentInputType,
                     default_text: Optional[Union[ListType[str], str]] = None,
                     max_chars: Optional[int] = 256,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.DocumentWithTextIn(
                documents=documents, default_text=default_text, title=title, description=description, max_chars=max_chars)

        @property
        def documents(self):
            return self._get_documents()

        def get(self):
            return self.get_text()

        def get_text(self) -> ListType[str]:
            return self.component.text

        def _get_documents(self) -> ListType[DocumentInputObj]:
            return convert_to_JsonObject(self.component.documents)


    class View(components.FeatherComponent):
        def __init__(self, documents: DocumentInputType,
                     output_text: Optional[ListType[str]] = None,
                     title: Optional[str] = None,
                     description: Optional[str] = None):
            self.component = components.DocumentView(
                documents=documents, text=output_text, title=title, description=description)

        @property
        def documents(self):
            return self._get_documents()

        def _get_documents(self) -> ListType[DocumentInputObj]:
            return convert_to_JsonObject(self.component.documents)
