import os
import requests
import base64
import json
from tqdm import tqdm
import time
from PIL import Image

# --- 1. API config for image editing (set OPENAI_API_KEY if using OpenAI-compatible endpoint) ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
# For image editing (image + prompt -> image), use an image-generation/editing API URL (e.g. provider-specific).
API_URL = os.environ.get("IMAGE_EDIT_API_URL", "")
MODEL_ID = os.environ.get("IMAGE_EDIT_MODEL", "")

# --- 2. Paths (IIEBENCH_DATA_DIR for data root) ---
DATA_DIR = os.environ.get("IIEBENCH_DATA_DIR", ".")
ORIGINAL_IMAGE_DIR = os.path.join(DATA_DIR, "sampled_datasets_long")
OUTPUT_DIR = os.environ.get("IIEBENCH_OUTPUT_DIR", "")


def encode_image_to_base64(image_path: str) -> str:
    """Read image file and return base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def download_image_from_url(url: str, output_path: str):
    """Download image from URL and save to output_path."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")

def call_image_edit_api(base64_image: str, modification_prompt: str):
    """Call image editing API; returns JSON with image URL or data."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    payload = {
        'model': MODEL_ID,
        'prompt': modification_prompt,
        'image': image_data_uri,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed for prompt '{modification_prompt}': {e}")
        if e.response is not None:
            print(f"Status: {e.response.status_code}, body: {e.response.text}")
        return None

def main():
    """Run the hardcoded multi-round image edit task."""
    print(f"Starting multi-round task with model {MODEL_ID}")

    base_name = os.path.splitext(os.environ.get("IIEBENCH_BASE_IMAGE_FILENAME", ""))[0]
    task_output_dir = os.path.join(OUTPUT_DIR, os.environ.get("IIEBENCH_TASK_SUBFOLDER", ""))
    os.makedirs(task_output_dir, exist_ok=True)

    current_input_image_path = os.path.join(ORIGINAL_IMAGE_DIR, os.environ.get("IIEBENCH_TASK_SUBFOLDER", ""), os.environ.get("IIEBENCH_BASE_IMAGE_FILENAME", ""))

    print(f"Base name: {base_name}")
    print(f"Original image: {current_input_image_path}")
    print(f"Output dir: {task_output_dir}")

    if not os.path.exists(current_input_image_path):
        print(f"Error: original image not found: {current_input_image_path}")
        return

    max_rounds = len(os.environ.get("IIEBENCH_MODIFICATION_PROMPTS", ""))

    try:
        for i in range(max_rounds):
            round_num = i + 1
            prompt_text = os.environ.get("IIEBENCH_MODIFICATION_PROMPTS", "")[i]

            output_image_name = f"{base_name}_{round_num}.png"
            output_image_path = os.path.join(task_output_dir, output_image_name)

            print(f"\n--- Round {round_num}/{max_rounds} for {base_name} ---")
            print(f"Instruction: {prompt_text}")

            if os.path.exists(output_image_path):
                print(f"Output exists, skip: {output_image_name}")
                current_input_image_path = output_image_path
                continue

            if not os.path.exists(current_input_image_path):
                print(f"Error: input image not found: {current_input_image_path}")
                break

            print(f"Encoding input: {current_input_image_path}")
            base64_input_image = encode_image_to_base64(current_input_image_path)
            if not base64_input_image:
                print("Failed to encode image, abort.")
                break

            print("Calling API...")
            api_result = call_image_edit_api(base64_input_image, prompt_text)

            if api_result and "images" in api_result and len(api_result["images"]) > 0:
                image_url = api_result["images"][0].get("url")
                if image_url:
                    print("Downloading result...")
                    download_image_from_url(image_url, output_image_path)
                    print(f"Saved: {output_image_path}")
                    current_input_image_path = output_image_path
                else:
                    print("Error: no URL in API response.")
                    break
            else:
                print("Error: API failed or empty result.")
                break

            time.sleep(1)

    except Exception as e:
        print(f"Error: {e}")

    print(f"\nTask {base_name} finished.")

if __name__ == "__main__":
    main()
