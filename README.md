# IIE-Bench

Benchmark and toolkit for **Instruction-based Image Editing (IIE)**. This repo provides evaluation code and dataset construction pipelines for single-turn, short multi-turn, and long multi-turn settings.

---

## Repository structure

```
iie_bench/
├── README.md
│
├── evaluation/
│   ├── evaluate.py           # Main evaluation script (LPIPS, PSNR, SSIM, CLIP-I, FG/BG)
│   ├── metrics_utils.py      # Metric implementations
│   └── run_evaluation.sh     # Batch multi-turn evaluation
│
└── dataset_construction/
    ├── MLLM_description.py       # MLLM image description
    ├── LLM_modification.py       # Single-turn edit instruction generation
    ├── Multi_LLM_Modification.py  # Multi-turn edit dialogue generation
    ├── prompt_manager.py         # Prompt templates
    ├── extract_entity.py         # Entity extraction
    ├── GroundingDINO.py          # Object detection (HuggingFace)
    ├── sam.py                    # SAM segmentation (HuggingFace)
    ├── image_synthesize.py       # Edit image synthesis (GT)
    ├── process_dataset.py        # Multi-turn JSON filtering
    └── QA_generation.py          # QA pair generation (optional)
```


## Data

The benchmark uses three JSON datasets (not shipped in this repo; distribute separately or via Hugging Face):

| Setting            | File                                              | Description                    |
|--------------------|---------------------------------------------------|--------------------------------|
| Single-turn        | `final_dataset.json`                              | 1725 samples, high/low level  |
| Multi-turn (short) | `final_multi_dataset_cleaned.json`                | 1131 round-level records      |
| Multi-turn (long)   | `long_multi_compiled_modifications_with_masks.json` | 1226 round-level records    |

Put these JSON files and the corresponding images/masks in a directory and set `IIEBENCH_DATA_DIR` to that path.

---

## Evaluation

### Environment variables

| Variable                    | Description                                      | Default       |
|----------------------------|--------------------------------------------------|---------------|
| `IIEBENCH_DATA_DIR`        | Directory with benchmark JSONs and GT paths      | `./data`      |
| `IIEBENCH_RESULTS_DIR`     | Single-turn result CSVs                         | `./results`   |
| `IIEBENCH_RESULTS_MULTI_DIR` | Multi-turn result CSVs                        | `./results_multi` |
| `HF_HOME`                  | Hugging Face cache (CLIP, LPIPS, etc.)           | `~/.cache/huggingface` |

### Single-turn

1. Place `final_dataset.json` in `$IIEBENCH_DATA_DIR`.
2. Organize model outputs so that for each `sample_id` and level (e.g. `_high`, `_low`) the evaluator can find the generated image under your model root (see `evaluate.py` for path logic).
3. Run:

```bash
cd evaluation
export IIEBENCH_DATA_DIR=/path/to/your/data
python evaluate.py \
  --input_json "$IIEBENCH_DATA_DIR/final_dataset.json" \
  --gen_dir /path/to/model/outputs \
  --model_name MyModel \
  --output_csv "$IIEBENCH_RESULTS_DIR/MyModel_scores.csv"
```

### Multi-turn

1. Place `final_multi_dataset_cleaned.json` (or the long JSON) in `$IIEBENCH_DATA_DIR`.
2. Model output directory names should end with `_multi`; the script looks under `generated_images_multi_50` or `generated_images_multi_long` as in the JSON paths.
3. Edit `evaluation/run_evaluation.sh`: set `MODEL_MULTI_PATHS` and `MODEL_MULTI_NAMES`, then:

```bash
cd evaluation
export IIEBENCH_DATA_DIR=/path/to/your/data
bash run_evaluation.sh
```

---

## Dataset construction

High-level pipeline: **MLLM description** → **LLM modification** (single or multi-turn) → **entity extraction** → **Grounding DINO + SAM** (boxes & masks) → **image synthesis** → **process_dataset.py** for multi-turn filtering.

### LLM API (required for modification/description scripts)

No default URLs or model names; set in env or `.env`:

| Variable           | Description |
|--------------------|-------------|
| `OPENAI_API_KEY`   | Your API key |
| `OPENAI_API_URL`   | Chat completion endpoint (e.g. `https://api.openai.com/v1/chat/completions`) |
| `OPENAI_MODEL`     | Model name (e.g. `gpt-4o`) |

For **image synthesis** (`image_synthesize.py`): set `IMAGE_EDIT_API_URL` and `IMAGE_EDIT_MODEL` (no defaults).

Example:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_URL="https://api.openai.com/v1/chat/completions"
export OPENAI_MODEL="gpt-4o"

# LLM_modification.py: set base dirs for input (MLLM description JSONs) and output (modification JSONs)
export IIEBENCH_LLM_MOD_INPUT_DIR=./MLLM_description_results
export IIEBENCH_LLM_MOD_OUTPUT_DIR=./LLM_modification_results
python dataset_construction/LLM_modification.py
```

### Other paths

- **`IIEBENCH_DATA_DIR`**: Used by `process_dataset.py` for JSON and output paths.
- **`HF_HOME`**: Hugging Face cache for Grounding DINO, SAM, etc.

---

## Citation

If you use IIE-Bench in your work, please cite:

```bibtex

}
```

(Replace with your paper details.)
