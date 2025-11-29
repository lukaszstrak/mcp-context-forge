# Standard
import asyncio
import base64
import json

# Third-Party
from granite_vision_server.providers import huggingface_vision


async def main():

    image_data = "documents/INVOICE-3.png"

    print("Testing analyze_image for", image_data)
    res1 = await huggingface_vision.analyze_image(model="granite3.2-vision:latest", image_data=image_data, analysis_type="general", include_conf=True, lang="en")


if __name__ == "__main__":
    asyncio.run(main())
