#!/bin/bash
#
# Multi-turn batch evaluation. Set: IIEBENCH_DATA_DIR, MODEL_MULTI_PATHS, MODEL_MULTI_NAMES, OUTPUT_BASE_MULTI_DIR.
#

set -e

# --- 1. Config (edit for your environment) ---

export IIEBENCH_DATA_DIR="${IIEBENCH_DATA_DIR:-./data}"
# Multi-turn GT JSON (HF dataset: multi_turn/multi_turn.json)
INPUT_MULTI_JSON_PATH="${IIEBENCH_DATA_DIR}/multi_turn/multi_turn.json"

# Model output dirs (names ending with _multi; evaluate.py looks under generated_images_multi_50)
MODEL_MULTI_PATHS=(
    "${IIEBENCH_DATA_DIR}/model_outputs/hive_multi"
    # Add more paths here...
)

MODEL_MULTI_NAMES=(
    "hive"
    # Add more names here...
)

OUTPUT_BASE_MULTI_DIR="${IIEBENCH_RESULTS_MULTI_DIR:-./results_multi}"

# --- 2. Run evaluation ---
echo "Starting multi-turn batch evaluation..."

mkdir -p "$OUTPUT_BASE_MULTI_DIR"

if [ ${#MODEL_MULTI_PATHS[@]} -ne ${#MODEL_MULTI_NAMES[@]} ]; then
    echo "Error: MODEL_MULTI_PATHS and MODEL_MULTI_NAMES must have the same length."
    exit 1
fi

if [ ! -f "$INPUT_MULTI_JSON_PATH" ]; then
    echo "Error: JSON not found: $INPUT_MULTI_JSON_PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for (( i=0; i<${#MODEL_MULTI_NAMES[@]}; i++ )); do
    NAME=${MODEL_MULTI_NAMES[$i]}
    GEN_DIR=${MODEL_MULTI_PATHS[$i]}
    OUTPUT_CSV="$OUTPUT_BASE_MULTI_DIR/${NAME}_scores.csv"

    echo "-----------------------------------------------------"
    echo "Evaluating: $NAME"
    echo "  JSON:  $INPUT_MULTI_JSON_PATH"
    echo "  Gen:   $GEN_DIR"
    echo "  Out:   $OUTPUT_CSV"
    echo "-----------------------------------------------------"

    python evaluate.py \
        --input_json "$INPUT_MULTI_JSON_PATH" \
        --gen_dir "$GEN_DIR" \
        --model_name "$NAME" \
        --output_csv "$OUTPUT_CSV"

    echo "Done: $NAME"
    echo ""
done

echo "All evaluations complete."
