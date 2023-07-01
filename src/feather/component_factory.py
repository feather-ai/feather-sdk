from feather import components, File, Image, List, Text, Document

# The component factory creates a concrete component from a JSON schema.
# We create the outer and inner components, BUT we DO NOT populate the props currently (TBD)
# The payloads will be injected into the components
def CreateComponent(json):
    t = json["component_type"]
    ret = None

    # File.Upload
    if t == components.FileLoader.COMPONENT_ID:
        ret = File.Upload.__new__(File.Upload)
        ret.component = components.FileLoader.__new__(components.FileLoader)
    # Text.In
    elif t == components.TextBoxInput.COMPONENT_ID:
        ret = Text.In.__new__(Text.In)
        ret.component = components.TextBoxInput.__new__(components.TextBoxInput)
    # File.Download
    elif t == components.FileDownload.COMPONENT_ID:
        ret = File.Download.__new__(File.Download)
        ret.component = components.FileDownload.__new__(components.FileDownload)
    # Text.View
    elif t == components.TextLabel.COMPONENT_ID:
        ret = Text.View.__new__(Text.View)
        ret.component = components.TextLabel.__new__(components.TextLabel)
    # List.SelectOne
    elif t == components.SingleSelectList.COMPONENT_ID:
        ret = List.SelectOne.__new__(List.SelectOne)
        ret.component = components.SingleSelectList.__new__(components.SingleSelectList)
    # List.SelectMulti
    elif t == components.MultiSelectList.COMPONENT_ID:
        ret = List.SelectMulti.__new__(List.SelectMulti)
        ret.component = components.MultiSelectList.__new__(components.MultiSelectList)
    # Image.WithSelectOne
    elif t == components.ImageWithSingleSelect.COMPONENT_ID:
        ret = Image.WithSelectOne.__new__(Image.WithSelectOne)
        ret.component = components.ImageWithSingleSelect.__new__(components.ImageWithSingleSelect)
    # Image.WithSelectMulti
    elif t == components.ImageWithMultiSelect.COMPONENT_ID:
        ret = Image.WithSelectMulti.__new__(Image.WithSelectMulti)
        ret.component = components.ImageWithMultiSelect.__new__(components.ImageWithMultiSelect)
    # Image.WithText
    elif t == components.ImageWithTextIn.COMPONENT_ID:
        ret = Image.WithTextIn.__new__(Image.WithTextIn)
        ret.component = components.ImageWithTextIn.__new__(components.ImageWithTextIn)
    # Image.View
    elif t == components.ImageView.COMPONENT_ID:
        ret = Image.View.__new__(Image.View)
        ret.component = components.ImageView.__new__(components.ImageView)
    # Document.View
    elif t == components.DocumentView.COMPONENT_ID:
        ret = Document.View.__new__(Document.View)
        ret.component = components.DocumentView.__new__(components.DocumentView)
    # Document.WithText
    elif t == components.DocumentWithTextIn.COMPONENT_ID:
        ret = Document.WithTextIn.__new__(Document.WithTextIn)
        ret.component = components.DocumentWithTextIn.__new__(components.DocumentWithTextIn)
    else:
        raise TypeError("Cannot create component - unknown type: {0}".format(t))
    return ret

    