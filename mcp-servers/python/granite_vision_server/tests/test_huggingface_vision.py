# Third-Party
from granite_vision_server.providers import huggingface_vision
import pytest


@pytest.mark.asyncio
async def test_huggingface_vision_analyze_image():

    image_data = "documents/INVOICE-3.png"

    print("Testing analyze_image for", image_data)
    res1 = await huggingface_vision.analyze_image(
        model="granite3.2-vision:latest",
        image_data=image_data,
        analysis_type="general",
        include_conf=True,
        lang="en",
    )

    assert res1 is not None
