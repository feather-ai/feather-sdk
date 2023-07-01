import base64
import pytest
import feather
import io
import numpy
from PIL import Image as PILImage

def test_FileUpload_Constructors():
    # test inputs
    title="my title"
    description="my description"
    
    component = feather.File.Upload(["images"], title=title, description=description).component
    assert component != None
    assert component.props != None
    assert component.props.title == title
    assert component.props.description == description

    schema = component._get_payload_schema()
    assert schema == '{"files": [{"data": "b64", "name": "string"}]}'

    # Pass invalid files to constructor
    with pytest.raises(TypeError):
        feather.File.Upload("images")

    with pytest.raises(TypeError):
        feather.File.Upload(("images"))

    with pytest.raises(TypeError):
        feather.File.Upload({"images":""})    

def test_FileUpload_Accessors():
    component = feather.File.Upload(["images"])
    good_payload = {"files": [ {"name": "name1", "data": base64.encodebytes(b'0123456789')}]}
    component.component._inject_payload(good_payload)

    raw_files = component.get()
    assert len(raw_files) == 1
    assert raw_files[0].name == "name1"
    assert raw_files[0].data == b'0123456789'

    raw_file_data = component.get(return_only_filedata=True)
    assert len(raw_file_data) == 1
    assert raw_file_data[0] == b'0123456789'

    # Validate the image accessors
    image = PILImage.new(mode="L", size=(4,4))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    image_payload = {"files": [ {"name": "name1", "data": base64.encodebytes(img_byte_arr)}]}
    component.component._inject_payload(image_payload)
    image_files = component.get(format="images")
    assert len(image_files) == 1
    assert image_files[0].name == "name1"
    assert (image_files[0].data == numpy.asarray(image)).all() == True

    raw_image_files = component.get(format="images", return_only_filedata=True)
    assert len(raw_image_files) == 1
    assert (raw_image_files[0] == numpy.asarray(image)).all() == True

    # Now try an invalid format
    with pytest.raises(ValueError):
        component.get(format="cakes", return_only_filedata=True)

def test_FileUpload_Payloads():
    component = feather.File.Upload(["images"]).component

    # Check payloads
    with pytest.raises(ValueError):
        missing_files_payload = {"invalid_files": []}
        component._inject_payload(missing_files_payload)

    with pytest.raises(ValueError):
        missing_name_payload = {"files": [ {"invalid_name": "", "invalid_data": ""}]}
        component._inject_payload(missing_name_payload)

    with pytest.raises(ValueError):
        missing_data_payload = {"files": [ {"name": "", "invalid_data": ""}]}
        component._inject_payload(missing_data_payload)

    good_payload = {"files": [ {"name": "name1", "data": base64.encodebytes(b'0123456789')}]}
    component._inject_payload(good_payload)
