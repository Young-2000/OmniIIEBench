import os
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# Configurations (GPT-4o via OpenAI)
# ==============================

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_URL = os.environ.get("OPENAI_API_URL", "")
MODEL_ID = os.environ.get("OPENAI_MODEL", "")

CONTEXT_DIR = Path(os.environ.get("IIEBENCH_CONTEXT_DIR", "")) 
INSTRUCTION_DIR = Path(os.environ.get("IIEBENCH_INSTRUCTION_DIR", ""))
OUTPUT_DIR = Path(os.environ.get("IIEBENCH_OUTPUT_DIR", ""))
MLLM_DESC_ROOT = CONTEXT_DIR
LLM_MOD_ROOT = INSTRUCTION_DIR
OUTPUT_ROOT = Path(os.environ.get("IIEBENCH_OUTPUT_ROOT", ""))

MAX_WORKERS = 1
MAX_RETRIES = 5
RETRY_DELAY = 5


PROMPT_TEMPLATE = """
You are an API endpoint that converts natural language image editing instructions into a structured JSON format.

# Output Rules
1. The response MUST be a single raw JSON object (no markdown).
2. `source_entity` and `target_entity` MUST ALWAYS be arrays of strings.
3. For attribute changes, the target entity should refer to the original object.
4. For multi-object modifications, list each source and target entity in corresponding order.

# Output Schema
  "source_entity": ["..."],
  "target_entity": ["..."]

---

# Examples

## Example 1: Single Entity Replacement
Source: "a red car"
Instruction: "replace the red car with a blue bus"
Output:
  "source_entity": ["a red car"],
  "target_entity": ["a blue bus"]

## Example 2: Attribute Change
Source: "a smiling dog"
Instruction: "make the dog bark"
Output:
  "source_entity": ["a smiling dog"],
  "target_entity": ["a smiling dog"]

## Example 3: Multi-part Modification
Source: "a woman with brown hair holding a book"
Instruction: "Change her hair to blonde and make her hold a red apple instead of a book."
Output:
  "source_entity": ["a woman with brown hair", "a book"],
  "target_entity": ["a woman with blonde hair", "a red apple"]

---

# Task
Process the request below using the rules above.

## Source Image Context (JSON):
{context_json_str}

## Modification Instruction:
{modification_text}

## Correct Output:
"""


# ==============================
# API Call
# ==============================

def call_llm_api(prompt: str, log_identifier: str) -> dict | None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                delay = RETRY_DELAY * (2 ** attempt)
                tqdm.write(f"[{log_identifier}] Rate limited (429). Retrying in {delay:.1f}s ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(delay)
                continue
            else:
                tqdm.write(f"[{log_identifier}] HTTP error: {e.response.status_code if e.response else 'N/A'}")
                return None

        except requests.exceptions.RequestException as e:
            tqdm.write(f"[{log_identifier}] Request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return None

    tqdm.write(f"[{log_identifier}] Failed after {MAX_RETRIES} retries.")
    return None


# ==============================
# Response Parsing
# ==============================

def parse_llm_response(response_data: dict, log_identifier: str) -> dict:
    if not response_data:
        return {"error": "API call returned no data."}

    try:
        content_str = response_data['choices'][0]['message']['content']
        return json.loads(content_str)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raw = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        tqdm.write(f"[{log_identifier}] Failed to parse JSON. Raw: '{raw}'")
        return {"error": "Invalid JSON from LLM", "details": str(e)}


# ==============================
# Progress Check
# ==============================

def is_task_completed(context_path: Path, instruction_path: Path, output_dir: Path) -> bool:
    try:
        with open(instruction_path, 'r', encoding='utf-8') as f:
            instruction_data = json.load(f)

        modifications = instruction_data.get("modifications")
        if not modifications:
            return False

        stem = context_path.stem
        for mod in modifications:
            level = mod.get("level")
            if not level:
                return False
            output_file = output_dir / f"{stem}_{level}.json"
            if not output_file.exists():
                return False

        return True

    except Exception:
        return False


# ==============================
# Single File Processing
# ==============================

def process_single_file(context_path: Path, instruction_path: Path, output_dir: Path) -> str:
    stem = context_path.stem
    log_identifier = f"{context_path.parent.name}/{context_path.name}"

    if not PROMPT_TEMPLATE.strip():
        return f"[{log_identifier}] PROMPT_TEMPLATE is empty."

    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        with open(instruction_path, 'r', encoding='utf-8') as f:
            instruction_data = json.load(f)

        for mod in instruction_data.get("modifications", []):
            level = mod.get("level")
            instruction_text = mod.get("modification_text")
            if not level or not instruction_text:
                continue

            output_path = output_dir / f"{stem}_{level}.json"
            if output_path.exists():
                continue

            final_prompt = PROMPT_TEMPLATE.format(
                context_json_str=json.dumps(context_data, indent=2),
                modification_text=instruction_text
            )

            api_response = call_llm_api(final_prompt, f"{log_identifier} [{level}]")
            parsed = parse_llm_response(api_response, f"{log_identifier} [{level}]")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=4, ensure_ascii=False)

        return f"[{log_identifier}] Done."

    except FileNotFoundError as e:
        return f"[{log_identifier}] Missing file: {e.filename}"
    except json.JSONDecodeError:
        return f"[{log_identifier}] JSON decode error"
    except Exception as e:
        return f"[{log_identifier}] Unexpected error: {e}"


# ==============================
# Main
# ==============================

def main():
    print("Starting entity extraction pipeline...")
    print(f"Model: {MODEL_ID}")
    print(f"Max workers: {MAX_WORKERS}")

    if not PROMPT_TEMPLATE.strip():
        print("ERROR: PROMPT_TEMPLATE is empty.")
        return

    potential_tasks = []
    dataset_dirs = [d for d in MLLM_DESC_ROOT.iterdir() if d.is_dir()]

    if not dataset_dirs:
        print(f"ERROR: No dataset folders found under {MLLM_DESC_ROOT}")
        return

    # Scan all pairs
    for dataset_dir in dataset_dirs:
        output_dir = OUTPUT_ROOT / dataset_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for context_path in dataset_dir.glob("*.json"):
            instruction_path = LLM_MOD_ROOT / dataset_dir.name / context_path.name
            if instruction_path.exists():
                potential_tasks.append((context_path, instruction_path, output_dir))
            else:
                print(f"Warning: Missing instruction file {instruction_path}")

    if not potential_tasks:
        print("No valid file pairs found.")
        return

    # Filter completed tasks
    tasks_to_run = []
    for c, i, o in tqdm(potential_tasks, desc="Checking progress"):
        if not is_task_completed(c, i, o):
            tasks_to_run.append((c, i, o))

    print(f"{len(tasks_to_run)} tasks remaining...")

    # Execute
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, *task): task for task in tasks_to_run}

        for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing"):
            try:
                result = future.result()
                tqdm.write(result)
            except Exception as e:
                c = futures[future][0]
                tqdm.write(f"Error during thread execution: {c.name}: {e}")

    print("All tasks completed.")


if __name__ == "__main__":
    main()
