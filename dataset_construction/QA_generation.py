import os
import json
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================================
# 1. Prompt template for QA generation
# ==========================================================
QA_GENERATION_PROMPT_TEMPLATE_V4 = """
# ROLE
You are a data generator for an IIE benchmark.

# TASK
Given one edit instruction, generate strictly between 1 and 3 (inclusive) verifiable QA pairs
to check if the edit was successful.
- The "question" asks *only* about the final state of the image.
- The "correct_answer" is the expected answer for a *perfect* edit.

# RULES
1. If the instruction is a **color/texture/style change** (e.g., "turn the car red"):
   - Q: "What is the new color of the car?"
   - A: "Red"
2. If the instruction is **keeping something** (e.g., "...while keeping the wheels black"):
   - Q: "Are the wheels still black?"
   - A: "Yes"
3. If the instruction is **replacing/removing something** (e.g., "replace the cat with a dog"):
   - Q: "Is there a dog in the image?" (Presence Check)
   - A: "Yes"
   - Q: "Is there a cat in the image?" (Absence Check - *Do not* use 'original cat')
   - A: "No"
4. Output *only* the JSON array. Do not add explanations.

# EXAMPLE INPUT
"Change the color of the smartphone from blue to red, while keeping the circular camera."

# EXAMPLE OUTPUT
[
  {"question": "What is the color of the smartphone?", "correct_answer": "Red"},
  {"question": "Does the image still show the circular camera?", "correct_answer": "Yes"}
]

---

# INPUT MODIFICATION
{MODIFICATION_TEXT}

# OUTPUT JSON ARRAY
"""

# ==========================================================
# 2. API config (GPT-4o via OPENAI_API_KEY)
# ==========================================================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_URL = os.environ.get("OPENAI_API_URL", "")
MODEL_ID = os.environ.get("OPENAI_MODEL", "")

MAX_WORKERS = 4
MAX_RETRIES = 5
RETRY_DELAY = 5


# ==========================================================
# 3. LLM API call (GPT-4o)
# ==========================================================
def call_llm_api(modification_text, log_identifier):
    """Call LLM to generate QA pairs for one edit instruction."""
    prompt_str = QA_GENERATION_PROMPT_TEMPLATE_V4.replace("{MODIFICATION_TEXT}", modification_text.strip())

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt_str}],
        "max_tokens": 512,
        "temperature": 0.3 
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "[]")

            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                qa_cleaned = content[start:end + 1]
                loaded_json = json.loads(qa_cleaned)
                if isinstance(loaded_json, list):
                    return loaded_json
                else:
                    raise json.JSONDecodeError("LLM did not return a list", content, 0)
            else:
                error_msg = f"No valid JSON array found in API response. Content: '{content}'"
                raise json.JSONDecodeError(error_msg, content, 0)

        except Exception as e:
            tqdm.write(f"API error {e}, retry {attempt+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
            continue
    return None


# ==========================================================
# 4. Process single JSON object: get modification_text, call LLM, add qa_pairs
# ==========================================================
def process_json_object(data_object):
    log_id = data_object.get("sample_id", "unknown_id")

    if "qa_pairs" in data_object:
        tqdm.write(f"Skip (QA exists): {log_id}")
        return data_object

    mod_text = data_object.get("modification_text")
    if not mod_text:
        tqdm.write(f"Skip (no modification_text): {log_id}")
        data_object["qa_pairs_error"] = "No modification_text found"
        return data_object

    qa_list = call_llm_api(mod_text, log_id)

    if qa_list is not None and len(qa_list) > 0:
        data_object["qa_pairs"] = qa_list
        tqdm.write(f"OK {log_id}: {len(qa_list)} QA pairs")
    else:
        data_object["qa_pairs_error"] = "Failed to generate QA pairs after retries"
        tqdm.write(f"Failed: {log_id}")

    return data_object


# ==========================================================
# 5. Main entry
# ==========================================================
def main():
    INPUT_JSON_FILE = "./multi_compiled_modifications_with_masks.json"
    OUTPUT_JSON_FILE = "./multi_compiled_modifications_with_masks_QA.json"

    if not API_KEY:
        print("Error: set OPENAI_API_KEY.")
        return

    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file not found: {INPUT_JSON_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {INPUT_JSON_FILE}: {e}")
        return

    if not isinstance(data_list, list):
        print(f"Error: {INPUT_JSON_FILE} root must be a list.")
        return

    print(f"Loaded {len(data_list)} objects from {INPUT_JSON_FILE}.")
    print(f"Generating QA pairs with {MAX_WORKERS} workers...")

    processed_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_json_object, obj): obj for obj in data_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="obj"):
            try:
                result_obj = future.result()
                processed_results.append(result_obj)
            except Exception as e:
                original_obj = futures[future]
                sample_id = original_obj.get("sample_id", "unknown_id")
                tqdm.write(f"Error {sample_id}: {e}")
                original_obj["qa_pairs_error"] = f"Critical Error: {e}"
                processed_results.append(original_obj)

    print(f"Saving {len(processed_results)} objects to {OUTPUT_JSON_FILE}...")

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)

    print(f"Done. Output: {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()