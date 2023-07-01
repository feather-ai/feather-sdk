import base64
from io import BytesIO
import json
from pathlib import Path
from typing import List, Optional
import PIL.Image as Image
import numpy as np
from collections.abc import MutableMapping
import glob
import os

class JsonObject(MutableMapping):

    def __init__(self, *args, **kwargs):
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def toJSON(self, pretty=True):
        index = None if pretty == False else 4
        dumps = json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=index)
        return dumps    

    def fromJSON(self, str):
        self.__dict__ = json.loads(str)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.__dict__[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.__dict__[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key


def safe_json_get(obj, key, errMsg):
    if key in obj:
        return obj[key]
    raise ValueError(errMsg)


def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))


def isValidFileType(fileType):
    supported = ["images", "video", "audio", ".csv", ".gif", ".jpg",
                 ".json", ".mp3", ".mp4", ".mpeg", ".png", ".tsv", ".txt"]
    return fileType in supported


def CleanRelativePath(relativeRoot, path):
    absPath = Path(path).absolute().resolve()
    relPath = absPath.relative_to(relativeRoot).as_posix().replace("\\", "/")
    return relPath


def img_array_to_base64_str(image):
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def base64_str_to_img_array(b64_image):
    binary = base64.b64decode(b64_image.split(",")[1])
    pil_im = Image.open(BytesIO(binary))
    img_array = np.asarray(pil_im)
    return img_array

def convert_one_dict_to_JsonObject(dict):
    x = JsonObject()
    x.name = dict["name"]
    x.data = dict["data"]
    return x

def noop(x):
    return x

def convert_to_JsonObject(list_of_dict, data_callback=noop) -> List[JsonObject]:
    ret = []
    for x in list_of_dict:
        item = JsonObject()
        item.name = x["name"]
        item.data = data_callback(x["data"])
        ret.append(item)
    return ret

def get_all_files_from_curr_dir(include_model_extensions: Optional[List[str]]=None, exclude_model_extensions: Optional[List[str]]=None):
    model_extensions=[".ckpt", ".pt", ".pth"]
    if type(include_model_extensions) == list:
        model_extensions.extend(include_model_extensions)
    elif include_model_extensions == None:
        pass
    else:
        raise TypeError("Expected include_model_extensions to be a list of strings, or None. Got {}".format(type(include_model_extensions)))

    if type(exclude_model_extensions) == list:
        model_extensions = [ext for ext in model_extensions if ext not in exclude_model_extensions]
    elif include_model_extensions == None:
        pass
    else:
        raise TypeError("Expected exclude_model_extensions to be a list of strings, or None. Got {}".format(type(include_model_extensions)))

    all_files = [f for f in glob.iglob('./**', recursive=True) if os.path.isfile(f)]
    code_files = [f for f in all_files if not f.endswith(tuple(model_extensions))]
    model_files = [f for f in all_files if f.endswith(tuple(model_extensions))]
    return code_files, model_files