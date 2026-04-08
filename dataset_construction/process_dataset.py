import json
import os
from collections import defaultdict

# --- 1. Paths (override with IIEBENCH_DATA_DIR, default ./data) ---
DATA_DIR = os.environ.get("IIEBENCH_DATA_DIR", "./data")

FLAT_JSON_PATH = os.path.join(DATA_DIR, "final_multi_dataset.json")
GROUPED_JSON_PATH = os.path.join(DATA_DIR, "final_multi_dataset_multi_cleaned.json")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "final_multi_dataset_cleaned.json")

def load_paths_from_grouped_file(filepath):
    """Load grouped file (B), collect all round_X_modified_image_path into a whitelist."""
    print(f"--- Loading whitelist (file B): {filepath} ---")
    paths_whitelist = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_groups = json.load(f)
        if not isinstance(all_groups, list):
            print(f"Error: {filepath} root is not a list.")
            return paths_whitelist

        for group in all_groups:
            if not isinstance(group, dict):
                continue
            try:
                total_rounds_int = int(group.get("total_rounds"))
                for i in range(1, total_rounds_int + 1):
                    key = f"round_{i}_modified_image_path"
                    path = group.get(key)
                    if path:
                        paths_whitelist.add(path)
            except (ValueError, TypeError, AttributeError):
                print(f"  Warning: invalid total_rounds for {group.get('sample_id', 'Unknown')}, skip.")

        print(f"Loaded {len(paths_whitelist)} unique modified_image_path into whitelist.")
        return paths_whitelist

    except FileNotFoundError:
        print(f"Error: file not found: {filepath}")
        return paths_whitelist
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in {filepath}")
        return paths_whitelist

def filter_flat_file(filepath, valid_paths_set):
    """Load flat file (A), keep only items whose modified_image_path is in whitelist."""
    print(f"\n--- Filtering flat file (A): {filepath} ---")
    kept_items = []
    total_count = 0
    discarded_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_items = json.load(f)
        if not isinstance(all_items, list):
            print(f"Error: {filepath} root is not a list.")
            return kept_items

        for item in all_items:
            total_count += 1
            if isinstance(item, dict):
                path_to_check = item.get("modified_image_path")
                if path_to_check in valid_paths_set:
                    kept_items.append(item)
                else:
                    discarded_count += 1

        print("\n--- Filter report ---")
        print(f"  Total: {total_count}, Kept: {len(kept_items)}, Discarded: {discarded_count}")
        return kept_items

    except FileNotFoundError:
        print(f"Error: file not found: {filepath}")
        return kept_items
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in {filepath}")
        return kept_items

def main():
    if (FLAT_JSON_PATH == "/path/to/your/flat_rounds_list.json" or
        GROUPED_JSON_PATH == "/path/to/your/grouped_by_sample_output.json" or
        OUTPUT_JSON_PATH == "/path/to/your/flat_list_filtered.json"):
        print("Error: set FLAT_JSON_PATH, GROUPED_JSON_PATH, OUTPUT_JSON_PATH at top of script.")
        return

    valid_paths = load_paths_from_grouped_file(GROUPED_JSON_PATH)
    if not valid_paths:
        print("Error: whitelist (file B) is empty.")
        return

    filtered_list = filter_flat_file(FLAT_JSON_PATH, valid_paths)

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f_out:
            json.dump(filtered_list, f_out, indent=4, ensure_ascii=False)
        print(f"\nSaved {len(filtered_list)} items to: {OUTPUT_JSON_PATH}")
    except IOError as e:
        print(f"Error writing output: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
