"""
Grounding DINO: fill missing source_boxes / target_boxes in JSON from entity text and images.
"""
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Configuration
INPUT_JSON_FILE = Path(os.environ.get("IIEBENCH_GROUNDING_JSON", "./long_multi_compiled_modifications_with_masks.json"))
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(HF_CACHE_DIR, exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

print(f"Loading GroundingDINO '{MODEL_ID}' on {DEVICE}...")
print(f"Cache: {HF_CACHE_DIR}")
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR).to(DEVICE)
print("Model loaded.")


def run_grounding_dino(image_path: str, text_prompts: list[str]) -> list[list[float]]:
    """Run Grounding DINO on one image and text prompts; return list of boxes."""
    if not text_prompts or not all(text_prompts):
        return []
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        tqdm.write(f"  Warning: image not found {image_path}, skip.")
        return []
    formatted_prompts = [text_prompts]
    inputs = processor(images=image, text=formatted_prompts, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)]
    )
    if "boxes" in results[0]:
        return [[round(x, 2) for x in box.tolist()] for box in results[0]["boxes"]]
    return []


def main():
    """Load JSON, fill missing source_boxes and target_boxes, save in place."""
    print(f"Processing: {INPUT_JSON_FILE}")
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {INPUT_JSON_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: invalid JSON: {INPUT_JSON_FILE}")
        return

    print(f"Found {len(data_list)} samples. Filling missing boxes...")
    samples_updated = 0
    source_boxes_added = 0
    target_boxes_added = 0

    for item in tqdm(data_list, desc="Processing"):
        try:
            entities = item.get("entities", {})
            if not entities:
                tqdm.write(f"  Warning: sample {item.get('sample_id', 'N/A')} missing 'entities', skip.")
                continue
            original_image_path = item.get("original_image_path")
            modified_image_path = item.get("modified_image_path")
            item_was_updated = False

            if not entities.get("source_boxes") and entities.get("source_entity") and original_image_path:
                tqdm.write(f"  Generating source_boxes for {item.get('sample_id')}...")
                new_source_boxes = run_grounding_dino(original_image_path, entities["source_entity"])
                item["entities"]["source_boxes"] = new_source_boxes
                source_boxes_added += 1
                item_was_updated = True
            elif not entities.get("source_boxes") and (not entities.get("source_entity") or not original_image_path):
                tqdm.write(f"  Warning: sample {item.get('sample_id')} missing source_entity or original_image_path.")

            if not entities.get("target_boxes") and entities.get("target_entity") and modified_image_path:
                tqdm.write(f"  Generating target_boxes for {item.get('sample_id')}...")
                new_target_boxes = run_grounding_dino(modified_image_path, entities["target_entity"])
                item["entities"]["target_boxes"] = new_target_boxes
                target_boxes_added += 1
                item_was_updated = True
            elif not entities.get("target_boxes") and (not entities.get("target_entity") or not modified_image_path):
                tqdm.write(f"  Warning: sample {item.get('sample_id')} missing target_entity or modified_image_path.")

            if item_was_updated:
                samples_updated += 1
        except Exception as e:
            tqdm.write(f"Error processing {item.get('sample_id', 'N/A')}: {e}")
            continue

    print("Saving...")
    try:
        with open(INPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing {INPUT_JSON_FILE}: {e}")
        return
    print(f"Done. Updated {samples_updated} samples. source_boxes: {source_boxes_added}, target_boxes: {target_boxes_added}. Saved to {INPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
