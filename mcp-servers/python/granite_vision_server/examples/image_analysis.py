#!/usr/bin/env python3
"""Example: Image Analysis with Granite 3.3 Vision via Ollama

This example demonstrates how to use the Granite Vision MCP Server
to analyze images using IBM Granite 3.3 Vision model through Ollama.

Prerequisites:
1. Ollama installed and running
2. Granite 3.3 Vision model pulled: ollama pull granite3.3-vision:2b
3. Server dependencies installed: pip install -e .

Usage:
    python examples/image_analysis.py path/to/image.jpg
    python examples/image_analysis.py https://example.com/image.jpg
"""

# Standard
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Third-Party
from granite_vision_server.models.granite_vision_models import build_prompts
from granite_vision_server.providers.ollama_vision import OllamaVisionClient
from granite_vision_server.utils.format_converters import normalize_response
from granite_vision_server.utils.image_utils import ensure_base64
from granite_vision_server.utils.model_validator import validate_granite_vision_setup


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def validate_setup():
    """Validate that Granite 3.3 Vision is available."""
    print_section("Validating Setup")

    is_valid, message = validate_granite_vision_setup()

    if is_valid:
        prefix = "[OK]"
        print(f"{prefix} {message}")
        return True
    prefix = "[ERROR]"
    print(f"{prefix} {message}")
    print("\nPlease run: ollama pull granite3.3-vision:2b")
    return False


def analyze_image_file(image_path: str, analysis_type: str = "general"):
    """Analyze an image file using Granite 3.3 Vision.

    Args:
        image_path: Path to image file or URL
        analysis_type: Type of analysis (general, detailed, objects, scene, text)

    """
    print_section(f"Analyzing Image: {image_path}")

    try:
        # Load image and convert to base64
        print(">> Loading image...")
        image_b64 = ensure_base64(image_path)
        print(f"[OK] Image loaded ({len(image_b64)} bytes base64)")

        # Build prompts
        system_prompt, user_prompt = build_prompts(
            analysis_type=analysis_type,
            language="en",
            max_desc=200,
            include_confidence=True,
        )

        # Initialize Ollama client
        print("\n>> Connecting to Ollama...")
        client = OllamaVisionClient()

        # Analyze image
        print(f">> Analyzing with Granite 3.3 Vision ({analysis_type})...")
        raw_response = client.analyze_image(
            model="granite3.3-vision:2b",
            image_b64=image_b64,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        print(f"[OK] Received response ({len(raw_response)} chars)")

        # Normalize response
        print("\n>> Processing results...")
        result = normalize_response(
            raw_text=raw_response,
            analysis_type=analysis_type,
            language="en",
            include_confidence=True,
            max_desc=200,
        )

        # Print results
        print_section("Analysis Results")

        print(f"Summary:\n  {result.get('summary', 'N/A')}\n")

        if result.get("objects"):
            print("Objects Detected:")
            for obj in result["objects"]:
                print(f"  • {obj}")
            print()

        if result.get("scene"):
            print(f"Scene: {result['scene']}\n")

        if result.get("text_in_image"):
            print(f"Text in Image: {result['text_in_image']}\n")

        if result.get("confidence") is not None:
            print(f"Confidence: {result['confidence']:.2%}\n")

        # Print raw response for debugging
        print_section("Raw Model Response")
        print(raw_response)

        # Print JSON output
        print_section("JSON Output")
        print(json.dumps(result, indent=2))

        return result

    except FileNotFoundError:
        prefix = "[ERROR]"
        print(f"{prefix} Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        prefix = "[ERROR]"
        print(f"{prefix} Error: {e}")
        # Standard
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python image_analysis.py <image_path_or_url> [analysis_type]")
        print("\nExamples:")
        print("  python examples/image_analysis.py path/to/image.jpg")
        print("  python examples/image_analysis.py https://example.com/image.jpg detailed")
        print("  python examples/image_analysis.py image.png objects")
        print("\nAnalysis types: general, detailed, objects, scene, text")
        sys.exit(1)

    image_path = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 else "general"

    # Validate setup
    if not validate_setup():
        sys.exit(1)

    # Analyze image
    result = analyze_image_file(image_path, analysis_type)

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
