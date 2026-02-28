"""
evaluate.py -- Evaluation script (JSON-based).
1. Load input JSON (GT), gen_dir, and model name from CLI.
2. Match GT paths and generated image paths per sample.
3. Compute metrics via metrics_utils (LPIPS, PSNR, SSIM, FG_LPIPS, FG_CLIP_I, BG_LPIPS, BG_CLIP_I).
4. Aggregate and save results to CSV.
"""

import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import json
import metrics_utils as mu

STANDARD_SIZE = (768, 768)


# ===================================================================
# Helpers
# ===================================================================
def _resize_image(img: Image.Image) -> Image.Image:
    if img.size == STANDARD_SIZE:
        return img
    try:
        return img.resize(STANDARD_SIZE, Image.Resampling.LANCZOS)
    except AttributeError:
        return img.resize(STANDARD_SIZE, Image.LANCZOS)

def _resize_mask(mask: Image.Image) -> Image.Image:
    if mask.size == STANDARD_SIZE:
        return mask
    try:
        return mask.resize(STANDARD_SIZE, Image.Resampling.NEAREST)
    except AttributeError:
        return mask.resize(STANDARD_SIZE, Image.NEAREST)

# ===================================================================
# Load samples from JSON and match generated images
# ===================================================================
def load_samples_from_json(args, skipped_json_path):
    """
    Load sample list from JSON and match generated images.
    If --gen_dir basename ends with '_multi', choose subdir
    generated_images_multi_50 or generated_images_multi_long from temp_path.
    """
    matched = []
    skipped_samples = [] 
    
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if not isinstance(json_data, list):
            print("Error: JSON root must be a list.")
            return []
    except Exception as e:
        print(f"Error loading {args.input_json}: {e}")
        return []

    print(f"Loaded {len(json_data)} samples from JSON. Matching against --gen_dir...")

    for item in tqdm(json_data, desc="Matching files"):
        try:
            sample_id = item['sample_id']
            level = item['level']
            suffix = f"_{level}" # e.g., "_high" or "_1"
            
            # GT paths from JSON
            paths = {
                'img_target': item['modified_image_path'],
                'mask_target': item['target_mask_path'],
            }

            temp_path = item['modified_image_path']

            temp_rel_dir = os.path.basename(os.path.dirname(temp_path))
            if temp_rel_dir == "sharegpt4v(llava)":
                rel_dir = "sharegpt4v"
            elif temp_rel_dir == "sharegpt4v":
                rel_dir = "sharegpt4v"
            else:
                rel_dir = temp_rel_dir

            base_gen_dir = args.gen_dir
            base_dir_name = os.path.basename(args.gen_dir.rstrip('/\\'))
            is_multi_eval = base_dir_name.endswith('_multi')

            if is_multi_eval:
                if 'synthesized_images_50' in temp_path or 'sampled_datasets_50' in temp_path:
                    base_gen_dir = os.path.join(args.gen_dir, 'generated_images_multi_50')
                elif 'synthesized_images_long' in temp_path or 'sampled_datasets_long' in temp_path:
                    base_gen_dir = os.path.join(args.gen_dir, 'generated_images_multi_long')
                else:
                    print(f"\nWarning: multi-eval but path matched neither '50' nor 'long': {temp_path}")

            gen_filename_base = f"{sample_id}{suffix}"
            
            img_gen_path_base = os.path.join(base_gen_dir, rel_dir, gen_filename_base)
            
            search_pattern = f"{img_gen_path_base}.*" 
            found_files = glob.glob(search_pattern)

            if not found_files:
                paths['img_gen'] = None
                item['expected_gen_path'] = search_pattern 
            else:
                if len(found_files) > 1:
                    print(f"Warning: Found multiple matches for {img_gen_path_base}. Using first one: {found_files[0]}")
                paths['img_gen'] = found_files[0] 

            
            missing = [k for k, v in paths.items() if not (v and os.path.exists(v))]
            if not missing:
                matched.append({
                    'sample_id': os.path.join(rel_dir, sample_id).replace('\\', '/'),
                    'sample_suffix': suffix,
                    'qa_data': None,
                    **paths
                })
            else:
                print(search_pattern)
                item['skip_reason'] = f"Missing files: {missing}"
                skipped_samples.append(item)
                # (!!!) END: (!!!)
                
        except Exception as e:
            print(f"Error parsing JSON item for {item.get('sample_id')}: {e}")
            item['skip_reason'] = f"Error: {e}"
            skipped_samples.append(item)

    print(f"Successfully matched {len(matched)} complete samples.")
    
    if skipped_samples:
        print(f"Saving {len(skipped_samples)} skipped samples to {skipped_json_path}...")
        try:
            with open(skipped_json_path, 'w', encoding='utf-8') as f_skip:
                json.dump(skipped_samples, f_skip, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving skipped samples: {e}")
    else:
        print("No samples were skipped.")

    return matched

def process_sample(paths):
    try:
        img_target  = _resize_image(Image.open(paths['img_target']).convert("RGB"))
        img_gen     = _resize_image(Image.open(paths['img_gen']).convert("RGB"))
        mask_target = _resize_mask(Image.open(paths['mask_target']).convert("L"))

        results = {
            'LPIPS': mu.calculate_lpips(img_gen, img_target),
            'PSNR': mu.calculate_psnr(img_gen, img_target),
            'SSIM': mu.calculate_ssim(img_gen, img_target),
            'CLIP_I': mu.calculate_clip_score(img_gen, img_target),
            'FG_CLIP_I': mu.calculate_clip_i_score(img_gen, mask_target, img_target, mask_target),
        }
        bg_lpips, bg_weight = mu.calculate_background_lpips(img_gen, img_target, mask_target)
        results['BG_LPIPS'] = bg_lpips
        results['_BG_Weight'] = bg_weight
        bg_clip, _ = mu.calculate_background_clip_score(img_gen, img_target, mask_target)
        results['BG_CLIP_I'] = bg_clip
        fg_lpips, fg_weight = mu.calculate_foreground_lpips(img_gen, img_target, mask_target)
        results['FG_LPIPS'] = fg_lpips
        results['_FG_Weight'] = fg_weight


        return results

    except Exception as e:
        print(f"Error processing {paths['sample_id']}: {e}")
        return None


# ===================================================================
# (!!!) 新: 单轮评测函数 (原 'main') (!!!)
# ===================================================================
def run_single_turn_evaluation(args):
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    csv_dir = os.path.dirname(args.output_csv)
    base_name = args.model_name 
    skipped_json_path = os.path.join(csv_dir, f"{base_name}_skipped.json")
    samples = load_samples_from_json(args, skipped_json_path)
    if not samples:
        print("No complete samples found.")
        return

    # --- 2. 计算所有样本 ---
    all_results = []
    for s in tqdm(samples, desc="Calculating metrics"):
        m = process_sample(s)
        if m:
            all_results.append({
                'model_name': args.model_name, 
                'sample_id': s['sample_id'], 
                'suffix': s['sample_suffix'], 
                **m
            })

    if not all_results:
        print("No valid results.")
        return

    df = pd.DataFrame(all_results)

    # --- 3. 泛化的加权平均辅助函数 ---
    def weighted_average(df_sub, metric_col: str, weight_col: str):
        valid = df_sub.dropna(subset=[metric_col, weight_col])
        if len(valid) == 0 or valid[weight_col].sum() == 0:
            return np.nan
        return np.average(valid[metric_col], weights=valid[weight_col])

    # Group averages
    numeric_cols = ['LPIPS', 'PSNR', 'SSIM', 'CLIP_I', 'FG_LPIPS', 'FG_CLIP_I', 'BG_LPIPS', 'BG_CLIP_I']
    
    avg_all  = df[numeric_cols].mean(skipna=True)
    avg_low  = df[df['suffix'] == '_low'][numeric_cols].mean(skipna=True)
    avg_high = df[df['suffix'] == '_high'][numeric_cols].mean(skipna=True)

    # BG metrics
    avg_all['BG_LPIPS']  = weighted_average(df, 'BG_LPIPS', '_BG_Weight')
    avg_low['BG_LPIPS']  = weighted_average(df[df['suffix'] == '_low'], 'BG_LPIPS', '_BG_Weight')
    avg_high['BG_LPIPS'] = weighted_average(df[df['suffix'] == '_high'], 'BG_LPIPS', '_BG_Weight')

    avg_all['BG_CLIP_I']  = weighted_average(df, 'BG_CLIP_I', '_BG_Weight')
    avg_low['BG_CLIP_I']  = weighted_average(df[df['suffix'] == '_low'], 'BG_CLIP_I', '_BG_Weight')
    avg_high['BG_CLIP_I'] = weighted_average(df[df['suffix'] == '_high'], 'BG_CLIP_I', '_BG_Weight')

    # FG metrics
    avg_all['FG_LPIPS']  = weighted_average(df, 'FG_LPIPS', '_FG_Weight')
    avg_low['FG_LPIPS']  = weighted_average(df[df['suffix'] == '_low'], 'FG_LPIPS', '_FG_Weight')
    avg_high['FG_LPIPS'] = weighted_average(df[df['suffix'] == '_high'], 'FG_LPIPS', '_FG_Weight')
    
    # Print results
    print("\n--- Average Scores ---")
    print("All:\n", avg_all)
    print("Low:\n", avg_low)
    print("High:\n", avg_high)

    # Save to CSV
    df = df.drop(columns=['_BG_Weight', '_FG_Weight'], errors='ignore')
    
    summary = pd.concat([
        avg_all.to_frame('score').assign(group='All'),
        avg_low.to_frame('score').assign(group='Low'),
        avg_high.to_frame('score').assign(group='High')
    ])

    summary.index.name = 'metric'
    summary = summary.reset_index().pivot(index='group', columns='metric', values='score').reset_index()
    summary['model_name'] = args.model_name
    
    # Row/column order
    group_order = ['All', 'Low', 'High']
    summary['group'] = pd.Categorical(summary['group'], categories=group_order, ordered=True)
    summary = summary.sort_values('group')
        
    metric_order = ['LPIPS', 'PSNR', 'SSIM', 'CLIP_I', 'FG_LPIPS', 'FG_CLIP_I', 'BG_LPIPS', 'BG_CLIP_I']
    final_cols = ['model_name', 'group'] + metric_order
    cols_to_save = [col for col in final_cols if col in summary.columns]
    summary = summary[cols_to_save]
        
    # (!!!) 修复: 6位小数 -> 4位小数 (!!!)
    summary.to_csv(args.output_csv, index=False, float_format='%.4f', mode='a',
                   header=not os.path.isfile(args.output_csv))
    print(f"✅ Saved results to {args.output_csv}")


# ===================================================================
# Multi-turn evaluation
# ===================================================================
def run_multi_turn_evaluation(args):

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    csv_dir = os.path.dirname(args.output_csv)
    base_name = args.model_name 
    skipped_json_path = os.path.join(csv_dir, f"{base_name}_skipped.json")
    
    # 1. 加载并匹配文件
    samples = load_samples_from_json(args, skipped_json_path) 
    if not samples:
        print("No complete samples found.")
        return

    # --- 2. 计算所有样本 ---
    all_results = []
    for s in tqdm(samples, desc="Calculating metrics"):
        m = process_sample(s)
        if m:
            all_results.append({
                'model_name': args.model_name, 
                'sample_id': s['sample_id'], 
                'suffix': s['sample_suffix'], 
                **m
            })

    if not all_results:
        print("No valid results.")
        return

    df = pd.DataFrame(all_results)

    # --- 3. 泛化的加权平均辅助函数 ---
    def weighted_average(df_sub, metric_col: str, weight_col: str):
        valid = df_sub.dropna(subset=[metric_col, weight_col])
        if len(valid) == 0 or valid[weight_col].sum() == 0:
            return np.nan
        return np.average(valid[metric_col], weights=valid[weight_col])

    # Overall average
    numeric_cols = ['LPIPS', 'PSNR', 'SSIM', 'CLIP_I', 'FG_LPIPS', 'FG_CLIP_I', 'BG_LPIPS', 'BG_CLIP_I']
    
    avg_all  = df[numeric_cols].mean(skipna=True)

    avg_all['BG_LPIPS']  = weighted_average(df, 'BG_LPIPS', '_BG_Weight')
    avg_all['BG_CLIP_I']  = weighted_average(df, 'BG_CLIP_I', '_BG_Weight')
    avg_all['FG_LPIPS']  = weighted_average(df, 'FG_LPIPS', '_FG_Weight')
    
    # Print results
    print("\n--- Average Scores ---")
    print("All:\n", avg_all)

    # Save to CSV
    df = df.drop(columns=['_BG_Weight', '_FG_Weight'], errors='ignore')
    
    avg_scores = avg_all.copy()
    avg_scores['model_name'] = args.model_name
    
    summary = avg_scores.to_frame().T 
    
    # Add 'group' column in multi mode
    summary['group'] = 'All' 

    metric_order = ['LPIPS', 'PSNR', 'SSIM', 'CLIP_I', 'FG_LPIPS', 'FG_CLIP_I', 'BG_LPIPS', 'BG_CLIP_I']
    final_cols = ['model_name', 'group'] + metric_order
        
    cols_to_save = [col for col in final_cols if col in summary.columns]
    summary = summary[cols_to_save]
        
    # (!!!) 修复: 6位小数 -> 4位小数 (!!!)
    summary.to_csv(args.output_csv, index=False, float_format='%.4f', mode='a',
                   header=not os.path.isfile(args.output_csv))
    print(f"✅ Saved results to {args.output_csv}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="IIE-Bench evaluation (single/multi turn).")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to JSON with GT sample info.")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Root directory of generated images.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    
    args = parser.parse_args()
    base_dir_name = os.path.basename(args.gen_dir.rstrip('/\\'))
    is_multi_eval = base_dir_name.endswith('_multi')
    if is_multi_eval:
        print("--- Running in MULTI-TURN evaluation mode ---")
        run_multi_turn_evaluation(args)
    else:
        print("--- Running in SINGLE-TURN evaluation mode (All, Low, High) ---")
        run_single_turn_evaluation(args)