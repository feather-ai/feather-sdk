import pytest
import feather
import numpy

def test_ImageView_Constructors():
    title="my title"
    description="my description"
    
    image1 = numpy.arange(8).reshape(1,8)
    image2 = numpy.arange(8).reshape(2,4)

    c0 = feather.Image.View(images=[image1], title=title, description=description)
    assert c0.component != None
    assert c0.component.props != None
    assert c0.component.props.title == title
    assert c0.component.props.description == description
    assert len(c0.component.images) == 1
    assert len(c0.component.text) == 0

    c1 = feather.Image.View(images=[image1, image2])
    assert len(c1.component.images) == 2
    assert len(c0.component.text) == 0

    # Pass a numpy image directly
    c2 = feather.Image.View(images=image1)
    assert len(c2.component.images) == 1
    assert len(c2.component.text) == 0

    # output text must be an array
    with pytest.raises(TypeError):
        feather.Image.View(images=[image1], output_text="Hi")

    c3 = feather.Image.View(images=[image1], output_text=["Hi"])
    assert len(c3.component.images) == 1
    assert len(c3.component.text) == 1

    # Not a numpy array
    with pytest.raises(TypeError):
        feather.Image.View(images=[[1]])

    with pytest.raises(TypeError):
        feather.Image.View(images=[image1, "hi"])
