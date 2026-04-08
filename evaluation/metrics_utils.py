# metrics_utils.py
# Image editing metrics: LPIPS, CLIP-I, IoU, BG-LPIPS, FG-LPIPS, VQA + LLM Judge QA-Score.
# Lazy-loaded models, unified interface.

import os
import torch
import lpips
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import json

# --- Global config ---
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_DEVICE = None
_LPIPS_MODEL = None
_CLIP_MODEL_HF = None
_CLIP_PROCESSOR_HF = None
_VQA_PIPE = None
_LLM_PIPE = None


def _initialize_device():
    """Initialize global device."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Full-image metrics: CLIP-I, LPIPS, FG-CLIP, IoU, PSNR, SSIM
# ---------------------------------------------------------------------------

def calculate_clip_score(img_gen_pil: Image.Image, img_target_pil: Image.Image) -> float:
    """Full-image CLIP similarity between generated and target image."""
    global _DEVICE, _CLIP_MODEL_HF, _CLIP_PROCESSOR_HF
    MODEL_ID = "openai/clip-vit-base-patch32"
    _initialize_device()

    if _CLIP_MODEL_HF is None:
        print(f"Initializing Hugging Face CLIP model ({MODEL_ID})...")
        _CLIP_MODEL_HF = CLIPModel.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        ).to(_DEVICE)
        _CLIP_PROCESSOR_HF = CLIPProcessor.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        )

    img_gen = img_gen_pil.convert("RGB")
    img_target = img_target_pil.convert("RGB")

    inputs = _CLIP_PROCESSOR_HF(
        images=[img_gen, img_target], 
        return_tensors="pt"
    ).to(_DEVICE)

    with torch.no_grad():
        image_features = _CLIP_MODEL_HF.get_image_features(**inputs)
        features_gen = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        features_target = image_features[1] / image_features[1].norm(dim=-1, keepdim=True)
        similarity = torch.dot(features_gen, features_target).item()
        
    return similarity

def calculate_lpips(img_gen_pil: Image.Image, img_target_pil: Image.Image) -> float:
    """Full-image LPIPS distance between generated and target image (lower is better)."""
    global _DEVICE, _LPIPS_MODEL
    _initialize_device()
    if _LPIPS_MODEL is None:
        print("Initializing LPIPS model (VGG)...")
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(_DEVICE)

    def _preprocess(img: Image.Image) -> torch.Tensor:
        # LPIPS expects [-1, 1]
        return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(_DEVICE) / 127.5 - 1

    tensor_gen = _preprocess(img_gen_pil)
    tensor_target = _preprocess(img_target_pil)
    with torch.no_grad():
        distance = _LPIPS_MODEL(tensor_gen, tensor_target).item()
    return distance


def calculate_clip_i_score(
    img_gen_pil: Image.Image, 
    mask_gen_pil: Image.Image, 
    img_target_pil: Image.Image, 
    mask_target_pil: Image.Image
) -> float:
    """Foreground CLIP similarity (masked region only)."""
    global _DEVICE, _CLIP_MODEL_HF, _CLIP_PROCESSOR_HF
    MODEL_ID = "openai/clip-vit-base-patch32"
    _initialize_device()
    if _CLIP_MODEL_HF is None:
        print(f"Initializing Hugging Face CLIP model ({MODEL_ID})...")
        _CLIP_MODEL_HF = CLIPModel.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        ).to(_DEVICE)
        _CLIP_PROCESSOR_HF = CLIPProcessor.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        )

    def _extract_foreground(image: Image.Image, mask: Image.Image) -> Image.Image:
        """Extract foreground by zeroing background pixels."""
        image_np = np.array(image.convert("RGB")).astype(np.uint8)
        mask_np_bool = np.array(mask.convert("L")) > 128
        fg_np = image_np.copy()
        fg_np[~mask_np_bool] = 0  

        return Image.fromarray(fg_np, 'RGB')

    fg_gen = _extract_foreground(img_gen_pil, mask_gen_pil)
    fg_target = _extract_foreground(img_target_pil, mask_target_pil)
    
    inputs = _CLIP_PROCESSOR_HF(
        images=[fg_gen, fg_target], 
        return_tensors="pt"
    ).to(_DEVICE)

    with torch.no_grad():
        image_features = _CLIP_MODEL_HF.get_image_features(**inputs)
        features_gen = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        features_target = image_features[1] / image_features[1].norm(dim=-1, keepdim=True)
        similarity = torch.dot(features_gen, features_target).item()
        
    return similarity


def calculate_iou(mask1_pil: Image.Image, mask2_pil: Image.Image) -> float:
    """IoU between two binary masks."""
    mask1_bool = np.array(mask1_pil.convert('L')) > 128
    mask2_bool = np.array(mask2_pil.convert('L')) > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    if np.sum(union) == 0: return 1.0 if np.sum(intersection) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)

def calculate_psnr(img_gen_pil: Image.Image, img_target_pil: Image.Image) -> float:
    """PSNR between generated and target image (higher is better). Images must be same size."""
    img_target_np = np.array(img_target_pil)
    img_gen_np = np.array(img_gen_pil)
    psnr_score = peak_signal_noise_ratio(img_target_np, img_gen_np, data_range=255)
    return float(psnr_score)


def calculate_ssim(img_gen_pil: Image.Image, img_target_pil: Image.Image) -> float:
    """SSIM between generated and target image (higher is better). Images must be same size."""
    img_target_np = np.array(img_target_pil)
    img_gen_np = np.array(img_gen_pil)
    ssim_score = structural_similarity(
        img_target_np, 
        img_gen_np, 
        channel_axis=-1, 
        data_range=255
    )
    return float(ssim_score)

# ---------------------------------------------------------------------------
# Region-based LPIPS (background / foreground)
# ---------------------------------------------------------------------------

def calculate_background_lpips(img_gen_pil: Image.Image, img_ref_pil: Image.Image, mask_ref_pil: Image.Image) -> tuple[float, float]:
    """LPIPS on background region only. Returns (lpips_score, background_proportion)."""
    global _DEVICE, _LPIPS_MODEL
    _initialize_device()
    if _LPIPS_MODEL is None:
        print("Initializing LPIPS model (VGG)...")
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(_DEVICE)

    mask_ref_np = np.array(mask_ref_pil.convert('L')) > 128
    bg_mask_np = ~mask_ref_np
    if bg_mask_np.size == 0:
        bg_proportion = 0.0
    else:
        bg_proportion = np.sum(bg_mask_np) / bg_mask_np.size

    if bg_proportion < 1e-5:
        return 0.0, 0.0

    def _preprocess_lpips(img: Image.Image) -> torch.Tensor:
        return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(_DEVICE) / 127.5 - 1

    tensor_gen = _preprocess_lpips(img_gen_pil)
    tensor_ref = _preprocess_lpips(img_ref_pil)
    bg_mask_float = torch.from_numpy(bg_mask_np).float().to(_DEVICE)
    bg_mask_tensor = bg_mask_float.unsqueeze(0).unsqueeze(0)
    tensor_gen_bg = tensor_gen * bg_mask_tensor
    tensor_ref_bg = tensor_ref * bg_mask_tensor

    with torch.no_grad():
        distance = _LPIPS_MODEL(tensor_gen_bg, tensor_ref_bg, retPerLayer=False, normalize=True).item()
        
    return distance, bg_proportion


def calculate_foreground_lpips(img_gen_pil: Image.Image, img_ref_pil: Image.Image, mask_ref_pil: Image.Image) -> tuple[float, float]:
    """LPIPS on foreground region only. Returns (lpips_score, foreground_proportion)."""
    global _DEVICE, _LPIPS_MODEL
    _initialize_device()
    if _LPIPS_MODEL is None:
        print("Initializing LPIPS model (VGG)...")
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(_DEVICE)

    # --- 1. 计算前景权重 (Foreground Proportion) ---
    mask_ref_np = np.array(mask_ref_pil.convert('L')) > 128
    fg_mask_np = mask_ref_np
    if fg_mask_np.size == 0:
        fg_proportion = 0.0
    else:
        fg_proportion = np.sum(fg_mask_np) / fg_mask_np.size
    if fg_proportion < 1e-5:
        return 0.0, 0.0
    def _preprocess_lpips(img: Image.Image) -> torch.Tensor:
        return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(_DEVICE) / 127.5 - 1

    tensor_gen = _preprocess_lpips(img_gen_pil)
    tensor_ref = _preprocess_lpips(img_ref_pil)

    fg_mask_float = torch.from_numpy(fg_mask_np).float().to(_DEVICE)
    fg_mask_tensor = fg_mask_float.unsqueeze(0).unsqueeze(0)
    tensor_gen_fg = tensor_gen * fg_mask_tensor
    tensor_ref_fg = tensor_ref * fg_mask_tensor
    with torch.no_grad():
        distance = _LPIPS_MODEL(tensor_gen_fg, tensor_ref_fg, retPerLayer=False, normalize=True).item()
        
    return distance, fg_proportion

# ===================================================================
# ---------------------------------------------------------------------------
# Region-based CLIP (background)
# ---------------------------------------------------------------------------
# ===================================================================

def calculate_background_clip_score(
    img_gen_pil: Image.Image, 
    img_ref_pil: Image.Image, 
    mask_ref_pil: Image.Image
) -> tuple[float, float]:
    """CLIP similarity on background region only. Returns (similarity, bg_proportion)."""
    global _DEVICE, _CLIP_MODEL_HF, _CLIP_PROCESSOR_HF
    MODEL_ID = "openai/clip-vit-base-patch32"
    _initialize_device()
    
    if _CLIP_MODEL_HF is None:
        print(f"Initializing Hugging Face CLIP model ({MODEL_ID})...")
        _CLIP_MODEL_HF = CLIPModel.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        ).to(_DEVICE)
        _CLIP_PROCESSOR_HF = CLIPProcessor.from_pretrained(
            MODEL_ID, 
            cache_dir=HF_CACHE_DIR
        )

    # --- 2. 计算背景权重 (Background Proportion) ---
    mask_ref_np = np.array(mask_ref_pil.convert('L')) > 128
    bg_mask_np = ~mask_ref_np
    if bg_mask_np.size == 0:
        bg_proportion = 0.0
    else:
        bg_proportion = np.sum(bg_mask_np) / bg_mask_np.size
    if bg_proportion < 1e-5:
        return 0.0, 0.0

    def _extract_background(image: Image.Image, mask_bool: np.ndarray) -> Image.Image:
        image_np = np.array(image.convert("RGB"))
        rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_np
        rgba[:, :, 3] = mask_bool * 255
        return Image.fromarray(rgba, 'RGBA')

    bg_gen = _extract_background(img_gen_pil, bg_mask_np)
    bg_ref = _extract_background(img_ref_pil, bg_mask_np)
    inputs = _CLIP_PROCESSOR_HF(
        images=[bg_gen, bg_ref], 
        return_tensors="pt"
    ).to(_DEVICE)

    with torch.no_grad():
        image_features = _CLIP_MODEL_HF.get_image_features(**inputs)
        features_gen = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        features_target = image_features[1] / image_features[1].norm(dim=-1, keepdim=True)
        similarity = torch.dot(features_gen, features_target).item()
        
    return similarity, bg_proportion


# ---------------------------------------------------------------------------
# VQA + LLM Judge (QA-Score)
# ---------------------------------------------------------------------------

def run_generative_vqa(img_pil: Image.Image, question: str) -> str:
    """Run VQA model to answer question about image."""
    global _VQA_PIPE, _DEVICE
    _initialize_device()
    
    if _VQA_PIPE is None:
        print(f"Initializing VQA model (Qwen/Qwen2.5-VL-7B-Instruct) on device: cuda:4...")
        try:
            _VQA_PIPE = pipeline(
                "image-text-to-text", 
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                device="cuda:4",
                torch_dtype="auto",
                model_kwargs={"cache_dir": HF_CACHE_DIR} 
            )
        except Exception as e:
            print(f"VQA pipeline init failed: {e}. Ensure transformers, torch, accelerate are installed.")
            raise e

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil.convert("RGB")}, 
                {"type": "text", "text": question}
            ]
        },
    ]

    try:
        response = _VQA_PIPE(text=messages, max_new_tokens=50)
        
        if (response and isinstance(response, list) and 
            'generated_text' in response[0] and 
            isinstance(response[0]['generated_text'], list) and
            response[0]['generated_text']): 

            chat_history = response[0]['generated_text']
            last_message = chat_history[-1]
            
            if last_message.get('role') == 'assistant' and 'content' in last_message:
                model_answer = last_message['content'].strip()
                return model_answer
            else:
                 print(f"Warning: VQA response format unexpected (no assistant content): {response}")
                 return ""
        else:
            print(f"Warning: VQA response format unexpected (generated_text not list): {response}")
            return ""
    except Exception as e:
        print(f"VQA run failed: {e}")
        return ""


def run_llm_judge(question: str, model_answer: str, correct_answer: str) -> bool:
    """[LLM - Stage 2] Run LLM Judge to check VQA answers."""
    global _LLM_PIPE, _DEVICE
    _initialize_device()

    if _LLM_PIPE is None:
        print(f"Initializing LLM Judge model (Qwen/Qwen2.5-7B-Instruct) on device: {_DEVICE}...")
        try:
            _LLM_PIPE = pipeline(
                "text-generation", 
                model="Qwen/Qwen2.5-7B-Instruct",
                device=_DEVICE,
                torch_dtype="auto",
                model_kwargs={"cache_dir": HF_CACHE_DIR}
            )
        except Exception as e:
            print(f"LLM Judge pipeline init failed: {e}. Ensure transformers, torch, accelerate are installed.")
            raise e

    prompt = f"""[Task]: Evaluate if an AI model's answer is semantically equivalent to the ground truth answer.
[Question]: {question}
[AI Model's Answer]: {model_answer}
[Ground Truth Answer]: {correct_answer}

[Judgement]: Is the AI model's answer semantically equivalent to the ground truth answer?

[Your Response] (Must respond with only "True" or "False"):
"""

    messages = [
        {"role": "system", "content": "You are a precise evaluator. You must only respond with the single word 'True' or the single word 'False'."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = _LLM_PIPE(messages, max_new_tokens=5, pad_token_id=_LLM_PIPE.tokenizer.eos_token_id)
        
        if response and isinstance(response, list) and 'generated_text' in response[0]:
            llm_response_text = response[0]['generated_text'][-1]['content'].strip()
            
            if "true" in llm_response_text.lower():
                return True
            else:
                return False
        else:
            print(f"Warning: LLM Judge response format unexpected: {response}")
            return False

    except Exception as e:
        print(f"LLM Judge run failed: {e}")
        return False


def calculate_qa_score(img_gen_pil: Image.Image, qa_data: list) -> float:
    """
    QA-Score (VQA + LLM Judge): for each QA pair, get VQA answer then LLM Judge.
    Returns 1.0 only if all QA pairs are correct, else 0.0.
    """
    _initialize_device()
    
    if not qa_data:
        return np.nan

    try:
        for qa_pair in qa_data:
            question = qa_pair.get("question")
            correct_answer = qa_pair.get("correct_answer")

            if not question or not correct_answer:
                print(f"Warning: Skipping invalid QA pair: {qa_pair}")
                continue

            # --- Stage 1: VQA ---
            model_answer = run_generative_vqa(img_gen_pil, question)
            
            if model_answer == "":
                print(f"VQA for question '{question}' failed. Marking as incorrect.")
                return 0.0

            # --- Stage 2: LLM Judge ---
            is_correct = run_llm_judge(question, model_answer, correct_answer)

            if not is_correct:
                return 0.0
        return 1.0
    except Exception as e:
        print(f"\nQA evaluation error: {e}")
        return np.nan