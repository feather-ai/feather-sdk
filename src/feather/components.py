import io
import uuid
import json
from typing import Callable, Dict, List
from feather import helpers
from feather.helpers import JsonObject, convert_one_dict_to_JsonObject, convert_to_JsonObject
import base64
import numpy as np
import pickle

# Given a list of outputs generated from a Step, generate the output data.
# Since a step may output components, we actually need the data from the components


def step_output_adapter(outputs):
    if outputs == None:
        return []

    ret = []
    for output in outputs:
        if isinstance(output, FeatherComponent):
            ret.append(output.component._serialize_payload())
        else:
            data = {"opaque_value": base64.b64encode(pickle.dumps(output)).decode("ascii")}
            ret.append(data)
    return ret


def step_opaque_input_data_adapter(inputIndex, payload):
    if "opaque_value" not in payload:
        raise TypeError("Input {0} has no 'opaque_value' field".format(inputIndex))
    data = payload["opaque_value"]
    data = base64.b64decode(data)
    return pickle.loads(data)

# Given a list of inputs for a Step and input payloads as sent from an external souce,
# load the external data (input payloads) into the correct input components


def step_input_adapter(step_inputs, input_payloads):
    for idx, input in enumerate(step_inputs):
        if isinstance(input, FeatherComponent):
            if input.component.wants_data():
                payload = input_payloads.pop(0)
                input.component._inject_payload(payload)  # Will throw if not valid
            else:
                input.component.generate_payload()
        else:
            payload = input_payloads.pop(0)
            step_inputs[idx] = step_opaque_input_data_adapter(idx, payload)


class FeatherComponent:
    def __init__(self):
        ""

# Base class for a Component


class Component:
    def __init__(self, version: str, typename: str, data_source: Callable):
        self.id = str(uuid.uuid4())
        self.dataSource = data_source
        self.version = version
        if typename == None or type(typename) != str or len(typename) == 0:
            raise ValueError("Invalid typename:", typename)
        self.typename = typename

    # Serialize the component into a JSON object for saving/transmitting.
    def _serialize_schema(self, metadata: JsonObject):
        if type(self.id) != str:
            raise ValueError("Component ID should be a string - did you change it?")
        metadata.type = "COMPONENT"
        metadata.component_type = self.typename
        metadata.version = self.version
        metadata.props = JsonObject()
        metadata.props = self.props

    #  Deserialize a component froma JSON string that was generated from a serialize call
    def deserialize(self, src: JsonObject):
        self.type = src.type
        self.version = src.version
        self.props = src.props

    # Wants data is true if the component needs external data to generate a result.
    def wants_data(self):
        return self.dataSource == None

    def generate_payload(self):
        if self.dataSource == None:
            raise ValueError("Component", self.typename, self.id,
                             "call to generate_payload but no data_source provided")
        self.payload = self.dataSource(self)

    # Inject payload is called when data is received from an external source for this component
    def _inject_payload(self, src: JsonObject):
        ""
        #print(self.typename, self.id, ": Injecting payload", src)
        # if src.id != str(self.id):
        #    raise ValueError("Component(",str(self.id),") inject_payload wrong ID!", src.id)


# -----------------------------------------------------------------------------
# FILE LOADER
# -----------------------------------------------------------------------------
class FileLoader(Component):
    COMPONENT_ID = "File.Upload"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, types: List[str],
                 title: str,
                 description: str,
                 min_files: int,
                 max_files: int):
        Component.__init__(self, version=FileLoader.COMPONENT_VERSION,
                           typename=FileLoader.COMPONENT_ID, data_source=None)

        if type(types) != list:
            raise TypeError(
                "File.Upload - 'types' must be a list of expected file types, but found {0}".format(type(types)))

        for ft in types:
            if helpers.isValidFileType(ft) == False:
                raise TypeError("File.Upload - unsupported file type '{0}'", ft)

        self.files = []
        self.props = JsonObject()
        self.props.types = types
        self.props.title = title
        self.props.description = description
        self.props.min_files = min_files
        self.props.max_files = max_files

    def _serialize_payload(self):
        ret = {}
        files = []
        for file in self.files:
            item = {}
            item["name"] = file.name
            item["data"] = base64.b64encode(file.data).decode("ascii")
            files.append(item)
        ret["files"] = files
        return ret

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        srcFiles = helpers.safe_json_get(src, "files", "File.Upload - missing 'files' key in data payload")

        self.files = []
        for finfo in srcFiles:
            f = JsonObject()
            f.name = helpers.safe_json_get(finfo, "name", "File.Upload - malformed payload. Missing 'name' field")
            f.data = base64.b64decode(helpers.safe_json_get(
                finfo, "data", "File.Upload - malformed payload. Missing 'data' field"))
            self.files.append(f)

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.files = []
        f = JsonObject()
        f.name = "string"
        f.data = "b64"
        o.files.append(f)
        return o.toJSON(pretty=False)

    # Generate payload is called when the component is in automatic mode
    def generate_payload(self):
        Component.generate_payload(self)
        self.files = self.dataSource(self)

# -----------------------------------------------------------------------------
# FILE DOWNLOAD
# -----------------------------------------------------------------------------


class FileDownload(Component):
    COMPONENT_ID = "File.Download"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, files,
                 output_filenames: List[str],
                 title: str,
                 description: str):
        Component.__init__(self, version=FileDownload.COMPONENT_VERSION,
                           typename=FileDownload.COMPONENT_ID, data_source=None)

        if files == None or isinstance(files, list) == False:
            raise TypeError("File.Download - 'files' must be set to a list of files, but got {0}".format(type(files)))
        self.files = files
        self.props = JsonObject()
        self.props.output_filenames = output_filenames or []
        self.props.title = title
        self.props.description = description

    def _serialize_payload(self):
        ret = {}
        files = []
        for file in self.files:
            item = {}
            if type(file) is bytes:
                item["name"] = ""
                item["data"] = base64.b64encode(file).decode("ascii")
            elif type(file) == str:
                item["name"] = ""
                item["data"] = base64.b64encode(str.encode(file)).decode("ascii")
            elif type(file) == dict or type(file) == JsonObject:
                item["name"] = file["name"]
                if type(file["data"]) is bytes:
                    item["data"] = base64.b64encode(file["data"]).decode("ascii")
                elif type(file["data"]) == str:
                    item["data"] = base64.b64encode(str.encode(file["data"])).decode("ascii")
                else:
                    raise TypeError("Expected 'data' to be of type 'str' or 'bytes'. Got {}".format(type(file["data"])))
            else:
                raise TypeError("Expected files to be either a list of 'str', 'bytes', or a dictionary-like object with 'name' and 'data' keys. Got {}".format(type(file)))
            files.append(item)
        ret["files"] = files
        return ret

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        srcFiles = helpers.safe_json_get(src, "files", "File.Download - missing 'files' key in data payload")
        self.files = []
        for finfo in srcFiles:
            f = JsonObject()
            f.name = helpers.safe_json_get(finfo, "name", "File.Download - malformed payload. Missing 'name' field")
            f.data = base64.b64decode(helpers.safe_json_get(
                finfo, "data", "File.Download - malformed payload. Missing 'data' field"))
            self.files.append(f)

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.files = []
        f = JsonObject()
        f.name = "string"
        f.data = "b64"
        o.files.append(f)
        return o.toJSON(pretty=False)

    # Generate payload is called when the component is in automatic mode
    def generate_payload(self):
        Component.generate_payload(self)
        self.files = self.dataSource(self)


# -----------------------------------------------------------------------------
# TEXT BOX INPUT
# -----------------------------------------------------------------------------
class TextBoxInput(Component):
    COMPONENT_ID = "Text.In"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, default_text,
                 title: str,
                 description: str,
                 num_inputs: int,
                 max_chars):

        Component.__init__(self, version=TextBoxInput.COMPONENT_VERSION,
                           typename=TextBoxInput.COMPONENT_ID, data_source=None)

        if num_inputs <= 0 or type(num_inputs) != int:
            raise ValueError("Text.In num_inputs must be an integer greater than 0")

        if default_text == None:
            default_text = ""
        elif not ((type(default_text) == str) or (type(default_text) == list)):
            raise ValueError("Text.In default_text must be None, string, or list")

        if type(default_text) == str:
            default_text = [default_text] * num_inputs
        elif type(default_text) == list:
            if len(default_text) != num_inputs:
                raise ValueError("Length of default text (if provided as list) must the same as num_inputs")

        self.text: List[str] = default_text
        self.props = JsonObject()
        self.props.title = title
        self.props.description = description
        self.props.num_inputs = num_inputs
        self.props.max_chars = max_chars

    # Inject payload is called when data is received from an external source for this component
    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.text = helpers.safe_json_get(src, "text", "Text.In - missing 'text' key in data payload")

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.text = []
        o.text.append("string")
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        ret["text"] = self.text
        return ret

# -----------------------------------------------------------------------------
# TEXT LABEL
# -----------------------------------------------------------------------------


class TextLabel(Component):
    COMPONENT_ID = "Text.View"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, text, title: str, description: str):

        Component.__init__(self, version=TextLabel.COMPONENT_VERSION, typename=TextLabel.COMPONENT_ID, data_source=None)
        if type(text) == str:
            text = [text]

        if type(text) != list:
            raise ValueError("Text.View - text must be str or [str]. got {0}".format(type(text)))

        for t in text:
            if type(t) != str:
                raise ValueError("Text.View - text must be str or [str]. got {0}".format(type(t)))

        self.text = text
        self.props = JsonObject()
        self.props.title = title
        self.props.description = description

    # Inject payload is called when data is received from an external source for this component
    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.text = src["text"]

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.text = "[string]"
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        ret["text"] = self.text
        return ret

# -----------------------------------------------------------------------------
# SINGLE SELECT LIST
# -----------------------------------------------------------------------------


class SingleSelectList(Component):
    COMPONENT_ID = "List.SelectOne"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, listItems: List[str],
                 title: str,
                 description: str,
                 style: str):

        Component.__init__(self, version=SingleSelectList.COMPONENT_VERSION,
                           typename=SingleSelectList.COMPONENT_ID, data_source=None)
        if style != "dropdown" and style != "radio":
            raise ValueError("List.SelectOne style must be either 'dropdown' or 'radio'")
        if listItems == None or isinstance(listItems, list) == False:
            raise TypeError(
                "List.SelectOne - 'listItems' must be set to a list of strings, but got {0}".format(type(listItems)))

        self.items = listItems
        self.selected_index = -1
        self.props = JsonObject()
        self.props.title = title
        self.props.description = description
        self.props.style = style

    # Inject payload is called when data is received from an external source for this component
    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        # Validate
        self.items = src["items"]
        self.selected_index = src["selected_index"]

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.items = []
        o.items.append("string")
        o.selected_index = "int"
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        ret["items"] = self.items
        ret["selected_index"] = self.selected_index
        return ret

# -----------------------------------------------------------------------------
# MULTI SELECT LIST
# -----------------------------------------------------------------------------


class MultiSelectList(Component):
    COMPONENT_ID = "List.SelectMulti"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, listItems,
                 title: str,
                 description: str):

        Component.__init__(self, version=MultiSelectList.COMPONENT_VERSION,
                           typename=MultiSelectList.COMPONENT_ID, data_source=None)
        if listItems == None or isinstance(listItems, list) == False:
            raise TypeError(
                "List.SelectOne - 'listItems' must be set to a list of strings or list of tuples, but got {0}".format(type(listItems)))

        self.items = []
        for i in listItems:
            if type(i) == str:
                self.items.append((i, False))
            if type(i) == tuple:
                if len(i) != 2:
                    raise TypeError("List.SelectOne - 'listItems' tuples must have 2 items [str,bool]")
                if type(i[0]) != str or type(i[1]) != bool:
                    raise TypeError("List.SelectOne - 'listItems' tuples must be [str,bool]")
                self.items.append(i)
                continue
            raise TypeError("List.SelectOne - 'listItems' must be str[] or tuple[], but got {0}".format(type(i)))

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description

    # Inject payload is called when data is received from an external source for this component
    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.items = []
        for srcItem in src["items"]:
            i = srcItem["item"]
            b = srcItem["selected"]
            self.items.append((i, b))

    # Return the JSON schema for the payload format to inject in this component
    def _get_payload_schema(self):
        o = JsonObject()
        o.items = []
        i = JsonObject()
        i.item = "string"
        i.selected = "bool"
        o.items.append(i)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        items = []
        for srcItems in self.items:
            i = srcItems[0]
            b = srcItems[1]
            items.append({"item": i, "selected": b})
        ret["items"] = items
        return ret


# -----------------------------------------------------------------------------
# IMAGE WITH SINGLE SELECT
# -----------------------------------------------------------------------------
class ImageWithSingleSelect(Component):
    COMPONENT_ID = "Image.WithSelectOne"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, images, lists,
                 title: str,
                 description: str,
                 style: str):

        Component.__init__(self, version=ImageWithSingleSelect.COMPONENT_VERSION,
                           typename=ImageWithSingleSelect.COMPONENT_ID, data_source=None)
        if isinstance(images, np.ndarray):
            images = [{"name": None, "data": images}]

        if style != "dropdown" and style != "radio":
            raise ValueError("Image.WithSelectOne style must be either 'dropdown' or 'radio'")
        if images == None:
            raise TypeError(
                "Image.WithSelectOne - 'images' must be set to a np[], or List[np[]], but got {0}".format(type(images)))
        if len(lists) != len(images):
            raise TypeError("Image.WithSelectOne - 'lists' and 'images' must have the same length")

        self.images = []
        self.attributes = []
        self.selected_indices = []
        for l in lists:
            if type(l) != list:
                raise TypeError("Image.WithSelectOne - 'lists' must be array of array List[List[str]]")
            self.attributes.append(l)
            self.selected_indices.append(-1)

        if type(images) is list:
            for obj in images:
                if type(obj) == dict:
                    if "name" in obj and "data" in obj:
                        obj = convert_one_dict_to_JsonObject(obj)
                    else:
                        raise TypeError("Image.WithSelectOne - 'images' dictionary didn't have either a 'name' or 'data' key in it")
                if not isinstance(obj, JsonObject):
                    raise TypeError("Image.WithSelectOne - 'images' must be List[{name, image}] - got {0}".format(type(obj)))
                if not hasattr(obj, "name"):
                    raise TypeError("Image.WithSelectOne - 'images' - missing 'name' attribute")
                if not hasattr(obj, "data"):
                    raise TypeError("Image.WithSelectOne - 'images' - missing 'data' attribute")

                if isinstance(obj.data, np.ndarray) == False:
                    raise TypeError("Image.WithSelectOne - 'One or more images is not a np array")
                self.images.append(obj)
        else:
            raise TypeError(
                "Image.WithSelectOne - 'images' must be set to a np[], or List[np[]], but got {0}".format(type(images)))

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description
        self.props.style = style

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.images = []
        self.attributes = []
        self.selected_indices = []

        for im in src["images"]:
            if "data" not in im:
                raise ValueError("Image.WithSelectOne - no 'data' field in 'images' list")
            if "name" not in im:
                raise ValueError("Image.WithSelectOne - no 'name' field in 'images' list")
            data = im["data"]
            name = im["name"]
            selected = im["selected_index"]
            attributes = im["attributes"]
            # TODO Validate
            self.images.append({"data": helpers.base64_str_to_img_array(data), "name": name})
            self.attributes.append(attributes)
            self.selected_indices.append(selected)

    def _get_payload_schema(self):
        o = JsonObject()
        im = JsonObject()
        im.data = "np"
        im.name = "string"
        im.attributes = ["string"]
        im.selected_index = "int"
        o.images = []
        o.images.append(im)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        images = []
        for idx in range(len(self.images)):
            image = self.images[idx]
            attributes = self.attributes[idx]
            selected = self.selected_indices[idx]
            obj = {}
            obj["data"] = helpers.img_array_to_base64_str(image["data"])
            obj["name"] = image["name"]
            obj["attributes"] = attributes
            obj["selected_index"] = selected
            images.append(obj)
        ret["images"] = images
        return ret

# -----------------------------------------------------------------------------
# IMAGE WITH MULTI SELECT
# -----------------------------------------------------------------------------


class ImageWithMultiSelect(Component):
    COMPONENT_ID = "Image.WithSelectMulti"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, images, lists,
                 title: str,
                 description: str):

        Component.__init__(self, version=ImageWithMultiSelect.COMPONENT_VERSION,
                           typename=ImageWithMultiSelect.COMPONENT_ID, data_source=None)
        if isinstance(images, np.ndarray):
            images = [{"name": None, "data": images}]

        if len(lists) != len(images):
            raise TypeError(
                "Image.WithSelectMulti - 'lists' and 'images' must have the same length {0} != {1}".format(len(lists), len(images)))

        self.images = []
        self.attributes = []
        for l in lists:
            if type(l) != list:
                raise TypeError(
                    "Image.WithSelectMulti - 'lists' must be array of array List[List[str]] or List[List[Tuple]]")
            innerList = []
            for li in l:
                if type(li) is tuple:
                    if type(li[0]) != str or type(li[1]) != bool or len(li) != 2:
                        raise TypeError("Image.WithSelectMulti - 'lists' should be a List[List[Tuple(str, bool)]]")
                    innerList.append(li)
                elif type(li) is str:
                    innerList.append((li, False))
            self.attributes.append(innerList)

        if type(images) is list:
            for obj in images:
                if type(obj) == dict:
                    if "name" in obj and "data" in obj:
                        obj = convert_one_dict_to_JsonObject(obj)
                    else:
                        raise TypeError("Image.WithSelectMulti - 'images' dictionary didn't have either a 'name' or 'data' key in it")
                if not isinstance(obj, JsonObject):
                    raise TypeError("Image.WithSelectMulti - 'images' must be List[{name, image}]- got {0}".format(type(obj)))
                if not hasattr(obj, "name"):
                    raise TypeError("Image.WithSelectMulti - 'images' - missing 'name' attribute")
                if not hasattr(obj, "data"):
                    raise TypeError("Image.WithSelectMulti - 'images' - missing 'data' attribute")

                if isinstance(obj.data, np.ndarray) == False:
                    raise TypeError("Image.WithSelectMulti - 'One or more images is not a np array")
                self.images.append(obj)
        else:
            raise TypeError(
                "Image.WithSelectMulti - 'images' must be set to a np[], or List[np[]], but got {0}".format(type(images)))

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.images = []
        self.attributes = []

        for im in src["images"]:
            if "data" not in im:
                raise ValueError("Image.WithSelectMulti - no 'data' field in 'images' list")
            if "name" not in im:
                raise ValueError("Image.WithSelectMulti - no 'name' field in 'images' list")
            data = im["data"]
            name = im["name"]
            attributes = im["attributes"]
            # TODO Validate
            self.images.append({"data": helpers.base64_str_to_img_array(data), "name": name})
            outList = []
            for attr in attributes:
                outList.append((attr["item"], attr["selected"]))
            self.attributes.append(outList)

    def _get_payload_schema(self):
        o = JsonObject()
        im = JsonObject()
        im.data = "b64"
        im.name = "string"
        attr = JsonObject()
        attr.selected = "bool"
        attr.item = "string"
        im.attributes = [attr]
        o.images = []
        o.images.append(im)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        images = []
        for idx in range(len(self.images)):
            image = self.images[idx]
            attributes = []
            for a in self.attributes[idx]:
                # Attributes are a list of a list of Tuples (we normalize in the constructor)
                innerObj = {}
                innerObj["item"] = a[0]
                innerObj["selected"] = a[1]
                attributes.append(innerObj)
            obj = {}
            obj["data"] = helpers.img_array_to_base64_str(image["data"])
            obj["name"] = image["name"]
            obj["attributes"] = attributes
            images.append(obj)
        ret["images"] = images
        return ret

# -----------------------------------------------------------------------------
# IMAGE WITH TEXT
# -----------------------------------------------------------------------------


class ImageWithTextIn(Component):
    COMPONENT_ID = "Image.WithTextIn"
    COMPONENT_VERSION = "1.0.0"

    def _validate_images(self, images):
        # If a single raw image, then promote to a list
        if isinstance(images, np.ndarray):
            images = [images]
        # If a list of raw images, then add empty names
        if type(images) != list:
            raise TypeError("Image.WithTextIn - 'images' argument in unsupported format  {0}".format(type(images)))

        ret = []
        for item in images:
            if isinstance(item, np.ndarray):
                ret.append({"name": "", "data": item})
            else:
                name = ""
                if hasattr(item, "name"):
                    name = item["name"]
                if not hasattr(item, "data"):
                    raise TypeError(
                        "Image.WithTextIn - 'images' argument in unsupported format (need 'data' key) {0}".format(type(item)))
                ret.append({"name": name, "data": item["data"]})
        return ret

    def __init__(self, images, default_text,
                 title: str,
                 description: str,
                 max_chars: int):

        Component.__init__(self, version=ImageWithTextIn.COMPONENT_VERSION,
                           typename=ImageWithTextIn.COMPONENT_ID, data_source=None)

        clean_images = self._validate_images(images)
        if default_text != None:
            if type(default_text) == str:
                new_text = []
                for d in images:
                    new_text.append(default_text)
                default_text = new_text
            else:
                if len(clean_images) != len(default_text):
                    raise TypeError("Image.WithTextIn - 'images' and 'default_text' must have the same length")
        else:
            default_text = []
            for i in images:
                default_text.append("")

        self.images = clean_images
        self.text = []
        for l in default_text:
            if type(l) != str:
                raise TypeError("Image.WithTextIn - 'default_text' must be array of array List[str]")
            self.text.append(l)

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description
        self.props.max_chars = max_chars

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.images = []
        self.text = []

        for im in src["images"]:
            data = im["data"]
            name = im["name"]
            # TODO Validate
            self.images.append({"data": helpers.base64_str_to_img_array(data), "name": name})
            self.text.append(im["text"])

    def _get_payload_schema(self):
        o = JsonObject()
        im = JsonObject()
        im.data = "np"
        im.text = "string"
        im.name = "string"
        o.images = []
        o.images.append(im)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        images = []
        for idx in range(len(self.images)):
            image = self.images[idx]
            obj = {}
            obj["data"] = helpers.img_array_to_base64_str(image["data"])
            obj["name"] = image["name"]
            obj["text"] = self.text[idx]
            images.append(obj)
        ret["images"] = images
        return ret

# -----------------------------------------------------------------------------
# IMAGE VIEW
# -----------------------------------------------------------------------------


class ImageView(Component):
    COMPONENT_ID = "Image.View"
    COMPONENT_VERSION = "1.0.0"

    def _validate_images(self, images):
        # If a single raw image, then promote to a list
        if isinstance(images, np.ndarray):
            images = [images]
        # If a list of raw images, then add empty names
        if type(images) != list:
            raise TypeError("Image.View - 'images' argument in unsupported format  {0}".format(type(images)))

        ret = []
        for item in images:
            if isinstance(item, np.ndarray):
                ret.append({"name": "", "data": item})
            else:
                name = ""
                if "name" in item:
                    name = item["name"]
                if "data" not in item:
                    raise TypeError(
                        "Image.View - 'images' argument in unsupported format (need 'data' key) {0}".format(type(item)))
                ret.append({"name": name, "data": item["data"]})
        return ret

    def __init__(self, images, output_text,
                 title: str,
                 description: str):

        Component.__init__(self, version=ImageView.COMPONENT_VERSION, typename=ImageView.COMPONENT_ID, data_source=None)

        clean_images = self._validate_images(images)

        if output_text != None and len(output_text) != len(clean_images):
            raise TypeError("Image.View - 'output_text' and 'images' must have the same length")

        self.images = clean_images
        self.text = []

        if output_text != None:
            for l in output_text:
                if type(l) != str:
                    raise TypeError("Image.View - 'output_text' must be array of array List[str]")
                self.text.append(l)

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.images = []
        self.text = []

        for im in src["images"]:
            data = im["data"]
            name = im["name"]
            # TODO Validate
            self.images.append({"data": helpers.base64_str_to_img_array(data), "name": name})
            self.text.append(im["text"])

    def _get_payload_schema(self):
        o = JsonObject()
        im = JsonObject()
        im.data = "np"
        im.text = "string"
        im.name = "string"
        o.images = []
        o.images.append(im)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        images = []
        for idx in range(len(self.images)):
            image = self.images[idx]
            obj = {}
            obj["data"] = helpers.img_array_to_base64_str(image["data"])
            obj["name"] = image["name"]
            obj["text"] = self.text[idx] if len(self.text) != 0 else ""
            images.append(obj)
        ret["images"] = images
        return ret


# -----------------------------------------------------------------------------
# DOCUMENT VIEW
# -----------------------------------------------------------------------------


class DocumentView(Component):
    COMPONENT_ID = "Document.View"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, documents, text,
                 title: str,
                 description: str):

        Component.__init__(self, version=DocumentView.COMPONENT_VERSION,
                           typename=DocumentView.COMPONENT_ID, data_source=None)

        if documents == None:
            raise TypeError("Document.View - 'documents' must be set")

        if text != None and len(documents) != len(text):
            raise TypeError("Document.View - 'documents' and 'text' must have the same length")

        self.documents = []
        self.text = []
        for d in documents:
            if type(d) == str:
                self.documents.append({"name": None, "data": d})
            else:
                self.documents.append({"name": d["name"], "data": d["data"]})

        if text is not None:
            for t in text:
                if type(t) != str:
                    raise TypeError("Document.View -  elements of 'text' must be strings - got {0}".format(type(t)))
                self.text.append(t)
        else:
            self.text = [""] * len(documents)

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.documents = []
        self.text = []

        for d in src["documents"]:
            self.documents.append({"data": d["data"], "name": d["name"]})
            self.text.append(d["text"])

    def _get_payload_schema(self):
        o = JsonObject()
        doc = JsonObject()
        doc.data = "string"
        doc.text = "string"
        doc.name = "string"
        o.documents = []
        o.documents.append(doc)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        documents = []
        for idx in range(len(self.documents)):
            obj = {}
            obj["data"] = self.documents[idx]["data"]
            obj["name"] = self.documents[idx]["name"]
            obj["text"] = self.text[idx]
            documents.append(obj)
        ret["documents"] = documents
        return ret

# -----------------------------------------------------------------------------
# DOCUMENT WITH TEXT
# -----------------------------------------------------------------------------


class DocumentWithTextIn(Component):
    COMPONENT_ID = "Document.WithTextIn"
    COMPONENT_VERSION = "1.0.0"

    def __init__(self, documents, default_text,
                 title: str,
                 description: str,
                 max_chars: int):

        Component.__init__(self, version=DocumentWithTextIn.COMPONENT_VERSION,
                           typename=DocumentWithTextIn.COMPONENT_ID, data_source=None)

        if documents == None:
            raise TypeError("Document.WithTextIn - 'documents' must be set")

        if default_text != None:
            if type(default_text) == str:
                new_text = []
                for d in documents:
                    new_text.append(default_text)
                default_text = new_text
            else:
                if len(documents) != len(default_text):
                    raise TypeError("Document.WithTextIn - 'documents' and 'default_text' must have the same length")
        else:
            default_text = []
            for d in documents:
                default_text.append("")

        self.documents = []
        self.text = []
        for d in documents:
            if type(d) == str:
                self.documents.append({"name": "", "data": d})
            else:
                self.documents.append({"name": d["name"], "data": d["data"]})

        for t in default_text:
            if type(t) != str:
                raise TypeError(
                    "Document.WithTextIn -  elements of 'default_text' must be strings - got {0}".format(type(t)))
            self.text.append(t)

        self.props = JsonObject()
        self.props.title = title
        self.props.description = description
        self.props.max_chars = max_chars

    def _inject_payload(self, src: JsonObject):
        Component._inject_payload(self, src)
        self.documents = []
        self.text = []

        for d in src["documents"]:
            self.documents.append({"data": d["data"], "name": d["name"]})
            self.text.append(d["text"])

    def _get_payload_schema(self):
        o = JsonObject()
        doc = JsonObject()
        doc.data = "string"
        doc.text = "string"
        doc.name = "string"
        o.documents = []
        o.documents.append(doc)
        return o.toJSON(pretty=False)

    def _serialize_payload(self):
        ret = {}
        documents = []
        for idx in range(len(self.documents)):
            obj = {}
            obj["data"] = self.documents[idx]["data"]
            obj["name"] = self.documents[idx]["name"]
            obj["text"] = self.text[idx]
            documents.append(obj)
        ret["documents"] = documents
        return ret
