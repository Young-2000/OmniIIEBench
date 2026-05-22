import os
import requests
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import prompt_manager
import sys

# --- 1. Base directory configuration (set via env; no defaults) ---
BASE_INPUT_DIR = os.environ.get("IIEBENCH_LLM_MOD_INPUT_DIR", "")
BASE_OUTPUT_DIR = os.environ.get("IIEBENCH_LLM_MOD_OUTPUT_DIR", "")

def generate_dataset_configs(base_input, base_output):
    configs = []
    if not base_input or not base_output:
        return []
    if not os.path.isdir(base_input):
        print(f"Error: base input directory '{base_input}' does not exist.")
        return []
    for subdir_name in sorted(os.listdir(base_input)):
        input_path = os.path.join(base_input, subdir_name)
        if os.path.isdir(input_path):
            configs.append({
                "name": subdir_name, "input_dir": input_path,
                "output_dir": os.path.join(base_output, subdir_name)
            })
    return configs

DATASET_CONFIGS = generate_dataset_configs(BASE_INPUT_DIR, BASE_OUTPUT_DIR)

# --- 2. API configuration (set OPENAI_API_KEY for GPT-4o) ---
API_KEY = os.environ.get("OPENAI_API_KEY", "") 
API_URL = os.environ.get("OPENAI_API_URL", "")
MODEL_ID = os.environ.get("OPENAI_MODEL", "")

# --- 3. Concurrency and retry ---
MAX_WORKERS = 6
MAX_RETRIES = 5
RETRY_DELAY = 5

# --- 4. Prompts are defined in prompt_manager.py ---

# --- 5. Core: call LLM API ---
def call_llm_api(system_prompt: str, scene_description_dict: dict, file_name: str, prompt_version: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    user_prompt = f"""
# SCENE DESCRIPTION ({scene_description_dict.get("entity_type")})
{json.dumps(scene_description_dict, indent=2, ensure_ascii=False)}

# GENERATE
Now, generate the JSON response based on all the rules.
"""
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                delay = RETRY_DELAY * (2 ** attempt)
                tqdm.write(f"Rate limited (429), sleeping {delay}s...")
                time.sleep(delay)
                continue
            else:
                tqdm.write(f"API HTTP error: {e.response.status_code if e.response else 'N/A'}")
                return None
        except requests.exceptions.RequestException as e:
            tqdm.write(f"Request error: {e}")
            return None

    tqdm.write(f"Failed after {MAX_RETRIES} retries.")
    return None

def process_dataset_folder(config, position):
    task_name = config["name"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    tqdm.write(f"--- [Thread {position}] Processing: {task_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
        files_to_process = [f for f in all_files if not os.path.exists(os.path.join(output_dir, f))]
        tqdm.write(f"[{task_name}] Found {len(all_files)} files, processing {len(files_to_process)}.")
    except FileNotFoundError:
        return f"--- [Thread {position}] Input directory not found: '{input_dir}' ---"

    if not files_to_process:
        return f"--- [Thread {position}] Task {task_name} already complete ---"

    for file_name in tqdm(files_to_process, desc=f"Processing {task_name}", position=position):
        try:
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            with open(input_path, 'r', encoding='utf-8') as f:
                scene_description_dict = json.load(f)

            entity_type = scene_description_dict.get("entity_type")

            if entity_type == "single":
                prompts_to_use = prompt_manager.ALL_SINGLE_ENTITY_PROMPTS
            elif entity_type == "multi":
                prompts_to_use = prompt_manager.ALL_MULTI_ENTITY_PROMPTS
            else:
                tqdm.write(f"Skip {file_name}: missing 'entity_type'.")
                continue

            all_results = []

            for version_name, system_prompt in prompts_to_use.items():
                api_result = call_llm_api(system_prompt, scene_description_dict, file_name, version_name)

                if api_result:
                    try:
                        json_content_str = api_result['choices'][0]['message']['content']
                        final_json = json.loads(json_content_str)
                        all_results.append({
                            "prompt_version": version_name,
                            "modifications": final_json.get("modifications", [])
                        })
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        tqdm.write(f"Parse error ({version_name}): {e}")

                time.sleep(0.2)

            if all_results:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
            else:
                tqdm.write(f"Warning: {file_name}: no successful generations.")

        except json.JSONDecodeError as e:
            tqdm.write(f"JSON read/parse error: {e}")
        except Exception as e:
            tqdm.write(f"Error: {e}")

    return f"--- [Thread {position}] Task {task_name} done ---"

def main():
    if not DATASET_CONFIGS:
        print("No tasks found. Set IIEBENCH_LLM_MOD_INPUT_DIR and IIEBENCH_LLM_MOD_OUTPUT_DIR.")
        sys.exit(1)

    print("--- Configured tasks ---")
    for config in DATASET_CONFIGS:
        print(f"  - {config['name']}")
    print("-" * 30)

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
