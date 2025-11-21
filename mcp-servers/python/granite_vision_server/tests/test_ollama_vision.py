import asyncio
import base64, json
from granite_vision_server.providers import ollama_vision_new


async def main():

    with open("tests/invoice.png", "rb") as f:  # ✅ must be 'rb'
        image_data = base64.b64encode(f.read()).decode("utf-8")

    print("=== Testing analyze_image ===")
    res1 = await ollama_vision_new.analyze_image(
        model="granite3.2-vision:latest",
        image_data=image_data,
        analysis_type="general",
        include_conf=True,
        lang="en"
    )

    print(res1, "\n")

    print("=== Testing OCR ===")
    res2 = await ollama_vision_new.ocr(
        model="granite3.2-vision:latest",
        image_data=image_data,
        languages=["en"],
        preserve_formatting=True,
        handwriting=False,
        conf_filter=False,
        bboxes=False,
        regions=None
    )

    print(res2, "\n")

    print("=== Testing VQA ===")
    res3 = await ollama_vision_new.vqa(
        model="granite3.2-vision:latest",
        image_data=image_data,
        question="What is in the picture?",
        context=None,
        answer_type="natural",
        conf=0.7
    )

    print(res3)

if __name__ == "__main__":
    asyncio.run(main())
