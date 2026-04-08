import os
import requests
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# ==========================================================
# 1. Prompt template for multi-turn dialogue generation
# ==========================================================
DIALOGUE_GENERATION_PROMPT_TEMPLATE = """
# ROLE
You are a professional data generator creating high-quality multi-turn editing dialogues for an "Instruction-based Image Editing (IIE) Benchmark" focused on **editing consistency**.

# OBJECTIVE
Generate a **Stateful Editing Dialogue** of 16 rounds that simulates a real designer progressively modifying the same image.
Each round must produce a clear, standalone instruction based on the current image state.

---

## RULES
1. **Sequential Dependency:**  
   Each instruction depends on the image after the previous modification.  
   The first turn refers to the original image; later turns follow updated states.

2. **Explicit Mentions:**  
   Avoid pronouns. Always describe entities and attributes explicitly according to their latest state.

3. **Edit Type Transitions:**  
   Use a natural mix of *Attribute Modifications (AM)* and *Entity Replacements (ER)*.  
   They may alternate or repeat, as long as the progression feels coherent.

4. **Semantic Progression:**  
   The dialogue should gradually evolve from subtle *attribute modifications (AM)* to stronger *entity replacements (ER)*.

5. **State Tracking:**  
   Internally maintain all entities and their updated attributes or replacements across turns.

6. **Grounded Start:**  
   The first instruction must reference objects or attributes from the given JSON input.

7. **Language:**  
   Use concise, natural English suitable for multimodal instruction datasets.

---

## OUTPUT FORMAT
- Output only a JSON array of strings.
- Each string = one editing instruction.
- No explanations or markdown.
- The JSON must be syntactically valid and end with a closing bracket `]`.

---

# INPUT IMAGE DESCRIPTION JSON
{JSON_INPUT_HERE}

# OUTPUT JSON ARRAY
"""

# ==========================================================
# 2. Directories and API (GPT-4o via OpenAI)
# ==========================================================
BASE_INPUT_DIR = "./MLLM_description_results"
SAMPLED_IMAGE_DIR = "./sampled_datasets_3"
BASE_OUTPUT_DIR = "./LLM_modification_long_multi_results"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_URL = os.environ.get("OPENAI_API_URL", "")
MODEL_ID = os.environ.get("OPENAI_MODEL", "")

MAX_WORKERS = 6
MAX_RETRIES = 5
RETRY_DELAY = 5

# ==========================================================
# 3. Helpers
# ==========================================================
def get_target_image_names(sample_dir):
    """Return set of image base names (no extension) under sample_dir."""
    image_names = []
    for root, _, files in os.walk(sample_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_names.append(os.path.splitext(f)[0])
    return set(image_names)


def find_matching_json_files(base_input_dir, target_names):
    """Find description JSON files under base_input_dir whose base name is in target_names."""
    matched_files = []
    for root, _, files in os.walk(base_input_dir):
        for f in files:
            if f.endswith(".json"):
                name = os.path.splitext(f)[0]
                if name in target_names:
                    matched_files.append(os.path.join(root, f))
    return matched_files


def call_llm_api(input_json_path, file_name):
    """Call LLM to generate multi-turn editing dialogue."""
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            mllm_json_data = json.load(f)
        json_string_for_prompt = json.dumps(mllm_json_data, indent=4, ensure_ascii=False)
    except Exception as e:
        tqdm.write(f"Cannot read/parse JSON: {e}")
        return None

    final_prompt_string = DIALOGUE_GENERATION_PROMPT_TEMPLATE.replace("{JSON_INPUT_HERE}", json_string_for_prompt)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": final_prompt_string}],
        "max_tokens": 2048,
        "temperature": 0.5
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "[]")

            try:
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1 and end > start:
                    content_cleaned = content[start : end + 1]
                    dialogue_list = json.loads(content_cleaned)
                    return dialogue_list
                else:
                    raise json.JSONDecodeError("No '[]' found", content, 0)
            except json.JSONDecodeError:
                tqdm.write(f"Response is not valid JSON: {content[:100]}...")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(RETRY_DELAY)
                continue

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                delay = RETRY_DELAY * (2 ** attempt)
                tqdm.write(f"Rate limited (429), sleeping {delay}s...")
                time.sleep(delay)
                continue
            else:
                tqdm.write(f"HTTP error {e.response.status_code if e.response else 'N/A'}")
                return None
        except requests.exceptions.RequestException as e:
            delay = RETRY_DELAY * (2 ** attempt)
            tqdm.write(f"Request error: {e}, sleeping {delay}s...")
            time.sleep(delay)
            continue

    tqdm.write(f"Failed after {MAX_RETRIES} retries.")
    return None


# ==========================================================
# 4. Process single JSON and save to output_dir
# ==========================================================
def process_single_json(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(json_path)
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        return f"Skip (exists): {file_name}"

    dialogue = call_llm_api(json_path, file_name)
    if dialogue:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dialogue, f, ensure_ascii=False, indent=4)
        return f"OK {file_name}: {len(dialogue)} rounds"
    else:
        return f"Failed: {file_name}"


# ==========================================================
# 5. Main entry
# ==========================================================
def main():
    print("Collecting image names from sampled dir...")
    target_names = get_target_image_names(SAMPLED_IMAGE_DIR)
    print(f"Found {len(target_names)} target images")

    print(f"Matching description JSONs under {BASE_INPUT_DIR}...")
    matched_jsons = find_matching_json_files(BASE_INPUT_DIR, target_names)
    print(f"Matched {len(matched_jsons)} description files")

    if not matched_jsons:
        print("No matching JSONs found. Check paths.")
        sys.exit(1)

    print(f"Starting dialogue generation with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for json_path in matched_jsons:
            relative_path = os.path.relpath(json_path, BASE_INPUT_DIR)
            relative_dir = os.path.dirname(relative_path)
            target_output_dir = os.path.join(BASE_OUTPUT_DIR, relative_dir)
            future = executor.submit(process_single_json, json_path, target_output_dir)
            futures[future] = os.path.basename(json_path)

        progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="file")
        for future in progress_bar:
            try:
                result = future.result()
                tqdm.write(result)
            except Exception as e:
                file_name = futures[future]
                tqdm.write(f"Error {file_name}: {e}")

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()
