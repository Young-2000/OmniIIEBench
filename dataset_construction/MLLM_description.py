import os
import requests
import base64
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# --- 1. Base directory configuration (set via env; no defaults) ---
BASE_INPUT_DIR = os.environ.get("IIEBENCH_MLLM_DESC_INPUT_DIR", "")
BASE_OUTPUT_DIR = os.environ.get("IIEBENCH_MLLM_DESC_OUTPUT_DIR", "")

def generate_dataset_configs(base_input, base_output):
    """Scan base input dir and build one config per subfolder."""
    configs = []
    if not os.path.isdir(base_input):
        print(f"Error: base input directory '{base_input}' does not exist.")
        return []
    for subdir_name in sorted(os.listdir(base_input)):
        input_path = os.path.join(base_input, subdir_name)
        if os.path.isdir(input_path):
            configs.append({
                "name": subdir_name,
                "input_dir": input_path,
                "output_dir": os.path.join(base_output, subdir_name)
            })
    return configs

DATASET_CONFIGS = generate_dataset_configs(BASE_INPUT_DIR, BASE_OUTPUT_DIR)

# --- 2. API configuration (set OPENAI_API_KEY for GPT-4o) ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_URL = os.environ.get("OPENAI_API_URL", "")
MODEL_ID = os.environ.get("OPENAI_MODEL", "")

# --- 3. Concurrency and retry ---
MAX_WORKERS = 1
MAX_RETRIES = 5
RETRY_DELAY = 5

# --- 4. System prompt for vision (image description) ---
SYSTEM_PROMPT_TEXT = """
# ROLE
You are a specialized AI Asset Extractor for Image Modification. Your sole purpose is to analyze an image and extract specific, actionable information needed to generate image modification prompts.

# TASK
Your task is to analyze the provided image and classify it as either **Single-Entity** or **Multi-Entity**. You will then generate a JSON object that cleanly separates replaceable subjects from modifiable attributes, strictly following the logic for handling complex scenes defined below. Your output is critical for two downstream edit types: High-level (Replacement) and Low-level (Attribute Edit).

# KEY PRINCIPLES
1.  **Purpose-Driven Extraction**: Only extract information directly useful for replacing a subject or modifying an attribute.
2.  **Specificity is Crucial**: `name` fields must be complete noun phrases (e.g., 'a fluffy white samoyed dog'). `attributes` must be concrete visual properties (e.g., 'fur is white').
3.  **Comprehensive Identification**: For Multi-Entity images, identify all distinct subjects central to the image's composition or action. For example, in a scene with two people flying a kite, the subjects are the two people AND the kite.

# INSTRUCTION
1.  **Analyze and Classify**: Determine if the image is `Single-Entity` or `Multi-Entity` using the specific rules below, especially for scenes.
2.  **Select Structure**: Choose the corresponding JSON structure.
3.  **Extract Assets**: Populate the chosen structure with precise details.
4.  **Output JSON Only**: Your entire output must be only the valid JSON object.

---

## Special Instructions: Handling Scenes, Landscapes, and Interiors
You MUST use the following logic to force all images into the `single` or `multi` category:

* **When to Classify as `Single-Entity`**:
    * **Dominant Element Rule**: If a scene contains one overwhelmingly dominant element, classify it as `single-entity` with that element as the subject.
        * *Example (Living Room)*: A large sofa occupies most of the frame. Classify as `single-entity`, with `replaceable_subject.name` being 'a large blue sofa'.
    * **Holistic Scene Rule**: If the a wide, general view without distinct, separable elements (e.g., a generic forest, a vast desert), classify it as `single-entity` and treat the entire scene as the subject.
        * *Example (Landscape)*: A wide shot of a forest. `replaceable_subject.name` should be 'a dense forest landscape'.

* **When to Classify as `Multi-Entity`**:
    * **Multiple Key Elements Rule**: If a scene contains several distinct, important elements of roughly equal visual weight, classify it as `multi-entity` and list them as subjects.
        * *Example (London View)*: The image clearly shows Big Ben, a bridge, and boats on the river. Classify as `multi-entity`, with `replaceable_subjects` being 'the Big Ben clock tower', 'a stone bridge', and 'tourist boats on the river'.
        * *Example (Living Room)*: A sofa, a coffee table, and a floor lamp are all clearly visible and important. Classify as `multi-entity`.

---

## 1. Single-Entity Structure
{
  "entity_type": "single",
  "replaceable_subject": {
    "name": "The full, descriptive name of the single subject. E.g., 'a ginger cat with green eyes', 'a large blue sofa', 'a dense forest landscape'."
  },
  "modifiable_attributes": {
    "subject": [ "List of concrete visual attributes of the subject. E.g., 'fur is ginger', 'fabric is blue denim', 'trees are tall and green'." ],
    "background": [ "List of concrete visual attributes of key background elements. E.g., 'sky is blue', 'wall is painted red'." ]
  }
}

---

## 2. Multi-Entity Structure
{
  "entity_type": "multi",
  "replaceable_subjects": [
    { "id": 1, "name": "Full descriptive name of the first subject. E.g., 'a man in a black jacket'." },
    { "id": 2, "name": "Full descriptive name of the second subject. E.g., 'a woman in a blue coat'." },
    { "id": 3, "name": "Full descriptive name of the third subject. E.g., 'a red and black kite'." }
  ],
  "modifiable_attributes": [
    { "id": 1, "attributes": [ "List of attributes for the first subject." ] },
    { "id": 2, "attributes": [ "List of attributes for the second subject." ] },
    { "id": 3, "attributes": [ "List of attributes for the third subject." ] }
  ]
}
"""

def encode_image_to_base64(image_path):
    """Read image file and return base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def call_vision_api(base64_image: str, file_name: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": SYSTEM_PROMPT_TEXT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
            ]}
        ],
        "max_tokens": 2048,
        "temperature": 0.2,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                delay = RETRY_DELAY * (2 ** attempt)
                tqdm.write(f"Rate limited (429), sleeping {delay}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(delay)
                continue
            else:
                tqdm.write(f"API HTTP error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            delay = RETRY_DELAY * (2 ** attempt)
            tqdm.write(f"Request error {e}, sleeping {delay}s...")
            time.sleep(delay)

    tqdm.write(f"Failed after {MAX_RETRIES} retries, skipping.")
    return None

def process_dataset_folder(config, position):
    task_name = config["name"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    tqdm.write(f"--- [Thread {position}] Processing: {task_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    except FileNotFoundError:
        return f"--- [Thread {position}] Input directory not found: '{input_dir}' ---"

    files_to_process = [f for f in all_images if not os.path.exists(os.path.join(output_dir, os.path.splitext(f)[0] + ".json"))]
    tqdm.write(f"[{task_name}] Found {len(all_images)} images, {len(all_images) - len(files_to_process)} done, processing {len(files_to_process)}.")

    if not files_to_process:
        return f"--- [Thread {position}] Task {task_name} already complete ---"

    for image_filename in tqdm(files_to_process, desc=f"Processing {task_name}", position=position):
        image_path = os.path.join(input_dir, image_filename)
        json_filename = os.path.splitext(image_filename)[0] + ".json"
        output_path = os.path.join(output_dir, json_filename)

        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            continue

        api_result = call_vision_api(base64_image, image_filename)

        if api_result:
            try:
                content_string = api_result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                if content_string.startswith("```json"):
                    content_string = content_string.strip("```json\n").strip("`\n")

                json_content = json.loads(content_string)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, ensure_ascii=False, indent=4)
            except (json.JSONDecodeError, IndexError) as e:
                tqdm.write(f"Failed to parse API response: {e}")
            except IOError as e:
                tqdm.write(f"Failed to write output: {e}")
        else:
            tqdm.write(f"Warning: failed to process {image_filename}, skipped.")

        time.sleep(0.1)

    return f"--- [Thread {position}] Task {task_name} done ---"

def main():
    if DATASET_CONFIGS:
        print("--- Configured tasks ---")
        for config in DATASET_CONFIGS:
            print(f"  - {config['name']}, input: {config['input_dir']}")
        print("-" * 50)
    else:
        print("No tasks found. Check BASE_INPUT_DIR.")
        sys.exit(1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_dataset_folder, config, i): config["name"]
            for i, config in enumerate(DATASET_CONFIGS)
        }
        print(f"Submitted {len(futures)} tasks to {MAX_WORKERS} workers.")

        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                task_name = futures[future]
                print(f"Task {task_name} failed: {e}")

    print("\n--- All tasks finished ---")

if __name__ == "__main__":
    main()
