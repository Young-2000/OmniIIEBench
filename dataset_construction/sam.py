"""
SAM: generate source/target mask images from JSON (boxes + images). Fills missing mask files.
"""
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from transformers import SamModel, SamProcessor

INPUT_JSON_FILE = Path(os.environ.get("IIEBENCH_SAM_JSON", "./long_multi_compiled_modifications_with_masks.json"))
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(HF_CACHE_DIR, exist_ok=True)

MODEL_ID = "facebook/sam-vit-huge"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM '{MODEL_ID}' on {DEVICE}...")
print(f"Cache: {HF_CACHE_DIR}")
model = SamModel.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR).to(DEVICE)
processor = SamProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
print("Model loaded.")


def generate_combined_mask(image_path: str, boxes: list[list[float]]) -> np.ndarray | None:
    """Run SAM on one image and boxes; merge all masks into one binary mask. Returns (H,W) uint8 or None."""
    if not boxes:
        tqdm.write(f"  Warning: no boxes for {image_path}, skip.")
        return None
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        tqdm.write(f"  Warning: image not found {image_path}, skip.")
        return None
    except Exception as e:
        tqdm.write(f"  Warning: open image {image_path} failed: {e}, skip.")
        return None
    input_boxes = [boxes]
    inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    iou_scores = outputs.iou_scores.cpu()
    image_height, image_width = masks[0].shape[-2:]
    combined_mask_np = np.zeros((image_height, image_width), dtype=np.uint8)
    for i in range(len(boxes)):
        best_mask_idx = torch.argmax(iou_scores.squeeze(0)[i]).item()
        segmentation_mask = masks[0][i][best_mask_idx]
        combined_mask_np = np.logical_or(combined_mask_np, segmentation_mask.numpy())
    return combined_mask_np.astype(np.uint8)


def main():
    """Load JSON, generate missing source/target mask files from boxes and images."""
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

    print(f"Found {len(data_list)} samples. Generating missing masks...")
    generated_source_masks = 0
    generated_target_masks = 0
    samples_processed = 0

    for item in tqdm(data_list, desc="Masks"):
        try:
            item_was_processed = False
            source_mask_path_str = item.get("source_mask_path")
            if source_mask_path_str:
                output_source_mask_path = Path(source_mask_path_str)
                if not output_source_mask_path.exists():
                    tqdm.write(f"  Generating (Source): {output_source_mask_path.name}")
                    output_source_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    original_image_path = item.get("original_image_path")
                    source_boxes = item.get("entities", {}).get("source_boxes", [])
                    source_mask_np = generate_combined_mask(original_image_path, source_boxes)
                    if source_mask_np is not None:
                        Image.fromarray(source_mask_np * 255).save(output_source_mask_path)
                        generated_source_masks += 1
                        item_was_processed = True
            else:
                tqdm.write(f"  Warning: sample {item.get('sample_id', 'N/A')} missing source_mask_path.")
            target_mask_path_str = item.get("target_mask_path")
            if target_mask_path_str:
                output_target_mask_path = Path(target_mask_path_str)
                if not output_target_mask_path.exists():
                    tqdm.write(f"  Generating (Target {item.get('level')}): {output_target_mask_path.name}")
                    output_target_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    modified_image_path = item.get("modified_image_path")
                    target_boxes = item.get("entities", {}).get("target_boxes", [])
                    target_mask_np = generate_combined_mask(modified_image_path, target_boxes)
                    if target_mask_np is not None:
                        Image.fromarray(target_mask_np * 255).save(output_target_mask_path)
                        generated_target_masks += 1
                        item_was_processed = True
            else:
                tqdm.write(f"  Warning: sample {item.get('sample_id', 'N/A')} missing target_mask_path.")
            if item_was_processed:
                samples_processed += 1
        except Exception as e:
            tqdm.write(f"Error processing {item.get('sample_id', 'N/A')}: {e}")
            continue

    print("Done.")
    print(f"Total samples: {len(data_list)}, processed: {samples_processed}, source masks: {generated_source_masks}, target masks: {generated_target_masks}")


if __name__ == "__main__":
    main()
