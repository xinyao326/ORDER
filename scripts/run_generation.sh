#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SEED=0

CLI_METHODS=""; CLI_DATASETS=""; CLI_TASKS=""; CLI_DEVICE=""
CLI_PRIOR_EPOCHS=""; CLI_DECODER_EPOCHS_COMPOSITE="1500"; CLI_DECODER_EPOCHS_FIBER="30"
CLI_SPLIT="train test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)                  CLI_METHODS="$2";                  shift 2 ;;
        --datasets)                 CLI_DATASETS="$2";                 shift 2 ;;
        --tasks)                    CLI_TASKS="$2";                    shift 2 ;;
        --device)                   CLI_DEVICE="$2";                   shift 2 ;;
        --prior_epochs)             CLI_PRIOR_EPOCHS="$2";             shift 2 ;;
        --decoder_epochs_composite) CLI_DECODER_EPOCHS_COMPOSITE="$2"; shift 2 ;;
        --decoder_epochs_fiber)     CLI_DECODER_EPOCHS_FIBER="$2";     shift 2 ;;
        --split)                    CLI_SPLIT="$2";                    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

NON_INTERACTIVE=false
[[ -n "$CLI_METHODS" && -n "$CLI_DATASETS" && -n "$CLI_TASKS" ]] && NON_INTERACTIVE=true

BOLD=$'\e[1m'; RESET=$'\e[0m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'
header() { echo -e "\n${BOLD}${CYAN}$*${RESET}"; }
ok()     { echo -e "${GREEN}✓ $*${RESET}"; }
warn()   { echo -e "${YELLOW}⚠ $*${RESET}"; }

multi_select() {
    local result_var="$1" prompt="$2" options=($3) default="${4:-}"
    echo -e "\n${BOLD}${prompt}${RESET}"
    local i=1
    for opt in "${options[@]}"; do printf "  %2d) %s\n" "$i" "$opt"; ((i++)); done
    if [[ -n "$default" ]]; then
        printf "  Select (space-separated indices or 'a' for all) [default: %s]: " "$default"
    else
        printf "  Select (space-separated indices or 'a' for all): "
    fi
    read -r input
    [[ -z "$input" && -n "$default" ]] && input="$default"
    local selected=()
    if [[ "$input" == "a" || "$input" == "all" ]]; then
        selected=("${options[@]}")
    else
        for idx in $input; do
            local n=$((idx - 1))
            if (( n >= 0 && n < ${#options[@]} )); then selected+=("${options[$n]}")
            else warn "Index $idx out of range — skipped."; fi
        done
    fi
    printf -v "$result_var" '%s' "${selected[*]}"
}

if [[ -z "$CLI_DEVICE" ]]; then
    header "GPU / Device"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null \
        | awk -F', ' '{printf "  GPU %s: %s | used %s | free %s\n",$1,$2,$3,$4}' || true
    printf "\n  Enter device [default: cuda:0]: "
    read -r DEVICE; DEVICE="${DEVICE:-cuda:0}"
else
    DEVICE="$CLI_DEVICE"
fi
echo "  Using device: ${BOLD}${DEVICE}${RESET}"

ALL_TASKS="train_prior train_decoder generate eval_generate physics_eval"
if $NON_INTERACTIVE; then
    SELECTED_TASKS="$CLI_TASKS"
else
    multi_select SELECTED_TASKS "Tasks:" "$ALL_TASKS" "a"
fi
[[ -z "$SELECTED_TASKS" ]] && { warn "No tasks selected. Exiting."; exit 1; }
echo "  Tasks: ${BOLD}${SELECTED_TASKS}${RESET}"

ALL_METHODS=("order_dyn"
             "order_alpha:0.0"
             "order_alpha:0.2"
             "order_alpha:0.5"
             "order_alpha:0.9"
             "order_alpha:0.95"
             "order_dyn_surr"
             "order_alpha_surr:0.2"
             "order_alpha_surr:0.5"
             "order_alpha_surr:0.9"
             "order_dyn_vit16"
             "order_alpha_vit16:0.0"
             "order_alpha_vit16:0.2"
             "order_alpha_vit16:0.5"
             "order_alpha_vit16:0.9"
             "order_alpha_vit16:0.95"
             "order_dyn_surr_vit16"
             "order_alpha_surr_vit16:0.2"
             "order_alpha_surr_vit16:0.5"
             "order_alpha_surr_vit16:0.9")

if $NON_INTERACTIVE; then
    SELECTED_METHODS="$CLI_METHODS"
else
    multi_select SELECTED_METHODS "Methods:" "${ALL_METHODS[*]}" "1"
fi
[[ -z "$SELECTED_METHODS" ]] && { warn "No methods selected. Exiting."; exit 1; }
echo "  Methods: ${BOLD}${SELECTED_METHODS}${RESET}"

if $NON_INTERACTIVE; then
    SELECTED_DATASETS="$CLI_DATASETS"
else
    multi_select SELECTED_DATASETS "Datasets:" "composite fiber" "a"
fi
[[ -z "$SELECTED_DATASETS" ]] && { warn "No datasets selected. Exiting."; exit 1; }
echo "  Datasets: ${BOLD}${SELECTED_DATASETS}${RESET}"

header "Epoch settings"
echo "  prior        n_epochs : ${CLI_PRIOR_EPOCHS:-default (500)}"
echo "  decoder composite    : ${CLI_DECODER_EPOCHS_COMPOSITE} epochs"
echo "  decoder fiber        : ${CLI_DECODER_EPOCHS_FIBER} epochs"
echo "  generate split       : ${CLI_SPLIT}"

header "Summary"
echo "  Device:   $DEVICE"
echo "  Tasks:    $SELECTED_TASKS"
echo "  Methods:  $SELECTED_METHODS"
echo "  Datasets: $SELECTED_DATASETS"

if ! $NON_INTERACTIVE; then
    printf "\n  Proceed? [Y/n]: "
    read -r confirm
    [[ "$confirm" =~ ^[Nn] ]] && { echo "Aborted."; exit 0; }
fi

echo ""
PASS=0; FAIL=0; SKIP=0
FAILED_JOBS=()

run_job() {
    local label="$1"; shift
    echo -e "\n${BOLD}▶ $label${RESET}"
    if "$PYTHON" -u "$@"; then
        ok "Done: $label"
        PASS=$((PASS + 1))
    else
        warn "FAILED: $label"
        FAILED_JOBS+=("$label")
        FAIL=$((FAIL + 1))
    fi
}

for METHOD in $SELECTED_METHODS; do
    PREFIX="${METHOD%%:*}"
    ALPHA="${METHOD##*:}"
    [[ "$ALPHA" == "$PREFIX" ]] && ALPHA=""

    IS_VIT16=false
    IS_SURR=false
    ACTUAL_PREFIX="$PREFIX"

    if [[ "$ACTUAL_PREFIX" == *_vit16 ]]; then
        IS_VIT16=true
        ACTUAL_PREFIX="${ACTUAL_PREFIX%_vit16}"
    fi
    if [[ "$ACTUAL_PREFIX" == *_surr ]]; then
        IS_SURR=true
        ACTUAL_PREFIX="${ACTUAL_PREFIX%_surr}"
    fi

    BACKBONE="CLIP_ViT-B/16"
    $IS_VIT16 && BACKBONE="ViT-B/16"

    SPLIT_SUFFIX="_clean"
    $IS_SURR && SPLIT_SUFFIX="_surr"

    for DATASET in $SELECTED_DATASETS; do

        LABEL_SUFFIX="$ACTUAL_PREFIX"
        [[ -n "$ALPHA" ]] && LABEL_SUFFIX="${ACTUAL_PREFIX}(α=${ALPHA})"
        $IS_SURR  && LABEL_SUFFIX="${LABEL_SUFFIX}[surr]"
        $IS_VIT16 && LABEL_SUFFIX="${LABEL_SUFFIX}[vit16]"
        LABEL_BASE="${LABEL_SUFFIX} / ${DATASET}"

        if [[ -n "$ALPHA" ]]; then
            SETTING="seed${SEED}_Alpha${ALPHA}${SPLIT_SUFFIX}_${DATASET}"
            SAVEPTH="save/${ACTUAL_PREFIX}/${BACKBONE/\//_}/${SETTING}"
        else
            SETTING="seed${SEED}${SPLIT_SUFFIX}_${DATASET}"
            SAVEPTH="save/${ACTUAL_PREFIX}/${BACKBONE/\//_}/${SETTING}"
        fi

        CKPT="${SAVEPTH}/weight-final.pth"
        if [[ ! -f "$CKPT" ]]; then
            warn "Pretrained checkpoint not found: $CKPT — skipping $LABEL_BASE"
            SKIP=$((SKIP + 1))
            continue
        fi

        COMMON_ARGS=(
            --prefix "$ACTUAL_PREFIX"
            --dataset "$DATASET"
            --device "$DEVICE"
            --seed "$SEED"
            --backbone "$BACKBONE"
            --split_suffix "$SPLIT_SUFFIX"
        )
        [[ -n "$ALPHA" ]] && COMMON_ARGS+=(--alpha "$ALPHA")

        for TASK in $SELECTED_TASKS; do

            if [[ "$TASK" == "train_prior" ]]; then
                LOG="${SAVEPTH}/log-prior.log"
                PRIOR_ARGS=("${COMMON_ARGS[@]}")
                [[ -n "$CLI_PRIOR_EPOCHS" ]] && PRIOR_ARGS+=(--n_epochs "$CLI_PRIOR_EPOCHS")
                run_job "train_prior $LABEL_BASE" \
                    train_prior.py "${PRIOR_ARGS[@]}" \
                    > "$LOG" 2>&1

            elif [[ "$TASK" == "train_decoder" ]]; then
                LOG="${SAVEPTH}/log-decoder.log"
                if [[ "$DATASET" == "composite" ]]; then
                    DECODER_EPOCHS="$CLI_DECODER_EPOCHS_COMPOSITE"
                else
                    DECODER_EPOCHS="$CLI_DECODER_EPOCHS_FIBER"
                fi
                run_job "train_decoder $LABEL_BASE" \
                    train_decoder.py "${COMMON_ARGS[@]}" --n_epochs "$DECODER_EPOCHS" \
                    > "$LOG" 2>&1

            elif [[ "$TASK" == "generate" ]]; then
                PRIOR_CKPT="${SAVEPTH}/prior.pth"
                DECODER_CKPT="${SAVEPTH}/decoder.pth"
                if [[ ! -f "$PRIOR_CKPT" ]]; then
                    warn "Prior checkpoint not found: $PRIOR_CKPT — skipping generate"
                    SKIP=$((SKIP + 1)); continue
                fi
                if [[ ! -f "$DECODER_CKPT" ]]; then
                    warn "Decoder checkpoint not found: $DECODER_CKPT — skipping generate"
                    SKIP=$((SKIP + 1)); continue
                fi
                for SPLIT in $CLI_SPLIT; do
                    LOG="${SAVEPTH}/log-generate-${SPLIT}.log"
                    run_job "generate $LABEL_BASE (split=$SPLIT)" \
                        generate.py "${COMMON_ARGS[@]}" --split "$SPLIT" \
                        > "$LOG" 2>&1
                done

            elif [[ "$TASK" == "eval_generate" ]]; then
                for SPLIT in $CLI_SPLIT; do
                    GEN_DIR="${SAVEPTH}/gen-${SPLIT}"
                    if [[ ! -d "$GEN_DIR" ]]; then
                        warn "Generated images not found: $GEN_DIR — run generate first"
                        SKIP=$((SKIP + 1)); continue
                    fi
                    LOG="${SAVEPTH}/log-evalgen-${SPLIT}.log"
                    run_job "eval_generate $LABEL_BASE (split=$SPLIT)" \
                        eval_generate.py "${COMMON_ARGS[@]}" --split "$SPLIT" \
                        > "$LOG" 2>&1
                done

            elif [[ "$TASK" == "physics_eval" ]]; then
                for SPLIT in $CLI_SPLIT; do
                    GEN_DIR="${SAVEPTH}/gen-${SPLIT}"
                    if [[ ! -d "$GEN_DIR" ]]; then
                        warn "Generated images not found: $GEN_DIR — run generate first"
                        SKIP=$((SKIP + 1)); continue
                    fi
                    LOG="${SAVEPTH}/log-physicseval-${SPLIT}.log"
                    if [[ "$DATASET" == "composite" ]]; then
                        CSV_PATH="../datasets_composite/${SPLIT}${SPLIT_SUFFIX}.csv"
                        run_job "physics_eval $LABEL_BASE (split=$SPLIT)" \
                            demo_physics_metrics.py --mode eval \
                            --gen_dir "$GEN_DIR" --csv "$CSV_PATH" \
                            > "$LOG" 2>&1
                    else
                        warn "physics_eval: unsupported dataset '$DATASET' — skipping"
                        SKIP=$((SKIP + 1)); continue
                    fi
                done

            fi
        done
    done
done

header "Finished"
echo "  Passed: ${GREEN}${PASS}${RESET}  Failed: ${YELLOW}${FAIL}${RESET}  Skipped: ${SKIP}"
if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo "  Failed jobs:"
    for j in "${FAILED_JOBS[@]}"; do echo "    • $j"; done
fi
