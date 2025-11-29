# Standard
import base64

# Third-Party
from granite_vision_server.providers import ollama_vision_new
import pytest


@pytest.mark.asyncio
async def test_ollama_vision_new():

    with open("tests/invoice.png", "rb") as f:  # ✅ must be 'rb'
        image_data = base64.b64encode(f.read()).decode("utf-8")

    print("=== Testing analyze_image ===")
    res1 = await ollama_vision_new.analyze_image(
        model="granite3.2-vision:latest",
        image_data=image_data,
        analysis_type="general",
        include_conf=True,
        lang="en",
    )

    assert res1 is not None

    print("=== Testing OCR ===")
    res2 = await ollama_vision_new.ocr(
        model="granite3.2-vision:latest",
        image_data=image_data,
        languages=["en"],
        preserve_formatting=True,
        handwriting=False,
        conf_filter=False,
        bboxes=False,
        regions=None,
    )

    assert res2 is not None

    print("=== Testing VQA ===")
    res3 = await ollama_vision_new.vqa(
        model="granite3.2-vision:latest",
        image_data=image_data,
        question="What is in the picture?",
        context=None,
        answer_type="natural",
        conf=0.7,
    )

    assert res3 is not None
