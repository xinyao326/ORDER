#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
ORDER_SCRIPTS="$(pwd)"

CLI_METHODS=""; CLI_DATASETS=""; CLI_TASKS=""; CLI_DEVICE=""; CLI_SPLIT="_clean"; CLI_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)    CLI_METHODS="$2";    shift 2 ;;
        --datasets)   CLI_DATASETS="$2";   shift 2 ;;
        --tasks)      CLI_TASKS="$2";      shift 2 ;;
        --device)     CLI_DEVICE="$2";     shift 2 ;;
        --split)      CLI_SPLIT="$2";      shift 2 ;;
        --seed)       CLI_SEED="$2";       shift 2 ;;
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

GPU_IDX="${DEVICE##*:}"

if [[ -z "$CLI_SEED" ]]; then
    header "Seed"
    printf "  Enter seed [default: 0]: "
    read -r SEED; SEED="${SEED:-0}"
else
    SEED="$CLI_SEED"
fi
echo "  Using seed: ${BOLD}${SEED}${RESET}"

if $NON_INTERACTIVE; then
    SELECTED_TASKS="$CLI_TASKS"
else
    multi_select SELECTED_TASKS "Tasks:" "pretrain retrieval" "a"
fi
[[ -z "$SELECTED_TASKS" ]] && { warn "No tasks selected. Exiting."; exit 1; }
echo "  Tasks: ${BOLD}${SELECTED_TASKS}${RESET}"

ALL_METHODS=("order_dyn"
             "order_alpha:0.0"
             "order_alpha:0.2"
             "order_alpha:0.5"
             "order_alpha:0.9"
             "order_dyn_surr"
             "order_alpha_surr:0.2"
             "order_alpha_surr:0.5"
             "order_alpha_surr:0.9"
             "order_dyn_vit16"
             "order_alpha_vit16:0.0"
             "order_alpha_vit16:0.2"
             "order_alpha_vit16:0.5"
             "order_alpha_vit16:0.9"
             "order_dyn_surr_vit16"
             "order_alpha_surr_vit16:0.2"
             "order_alpha_surr_vit16:0.5"
             "order_alpha_surr_vit16:0.9")

if $NON_INTERACTIVE; then
    SELECTED_METHODS="$CLI_METHODS"
else
    multi_select SELECTED_METHODS "Methods:" "${ALL_METHODS[*]}" "1 3 5 6 7 9 10 12 14 15 17 18"
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

header "Summary"
echo "  Device:   $DEVICE"
echo "  Tasks:    $SELECTED_TASKS"
echo "  Methods:  $SELECTED_METHODS"
echo "  Datasets: $SELECTED_DATASETS"
echo "  Split:    $CLI_SPLIT"

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
    local workdir="$1"; shift
    echo -e "\n${BOLD}▶ $label${RESET}"
    pushd "$workdir" > /dev/null
    if "$PYTHON" -u "$@"; then
        ok "Done: $label"
        PASS=$((PASS + 1))
    else
        warn "FAILED: $label"
        FAILED_JOBS+=("$label")
        FAIL=$((FAIL + 1))
    fi
    popd > /dev/null
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
    SPLIT_SUFFIX="$CLI_SPLIT"
    $IS_SURR && SPLIT_SUFFIX="_surr"
    BACKBONE_ARG=()
    $IS_VIT16 && BACKBONE_ARG=(--backbone "ViT-B/16")

    for DATASET in $SELECTED_DATASETS; do

        LABEL_DISP="$ACTUAL_PREFIX"
        $IS_VIT16 && LABEL_DISP="${LABEL_DISP}[vit16]"
        $IS_SURR  && LABEL_DISP="${LABEL_DISP}[surr]"
        if [[ -n "$ALPHA" ]]; then
            LABEL_BASE="${LABEL_DISP}(α=${ALPHA}) / ${DATASET}"
        else
            LABEL_BASE="${LABEL_DISP} / ${DATASET}"
        fi

        LOG_TAG="${ACTUAL_PREFIX}"
        $IS_VIT16 && LOG_TAG="${LOG_TAG}_vit16"
        $IS_SURR  && LOG_TAG="${LOG_TAG}_surr"

        for TASK in $SELECTED_TASKS; do

            if [[ "$TASK" == "pretrain" ]]; then

                if [[ "$ACTUAL_PREFIX" == "order_dyn" ]]; then
                    $IS_SURR && TRAIN_SCRIPT="train_order_dyn_surr.py" || TRAIN_SCRIPT="train_order_dyn.py"
                    LOGFILE="logs/pretrain/pretrain_${LOG_TAG}_${DATASET}.log"
                    if $IS_SURR; then
                        run_job "pretrain $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode train --seed "$SEED" \
                            --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    else
                        run_job "pretrain $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode train --seed "$SEED" \
                            --split_suffix "$SPLIT_SUFFIX" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    fi

                elif [[ "$ACTUAL_PREFIX" == "order_alpha" ]]; then
                    $IS_SURR && TRAIN_SCRIPT="train_order_alpha_surr.py" || TRAIN_SCRIPT="train_order_alpha.py"
                    LOGFILE="logs/pretrain/pretrain_${LOG_TAG}${ALPHA}_${DATASET}.log"
                    if $IS_SURR; then
                        run_job "pretrain $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode train --seed "$SEED" \
                            --alpha "$ALPHA" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    else
                        run_job "pretrain $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode train --seed "$SEED" \
                            --alpha "$ALPHA" --split_suffix "$SPLIT_SUFFIX" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    fi
                fi

            elif [[ "$TASK" == "retrieval" ]]; then

                if [[ "$ACTUAL_PREFIX" == "order_dyn" ]]; then
                    $IS_SURR && TRAIN_SCRIPT="train_order_dyn_surr.py" || TRAIN_SCRIPT="train_order_dyn.py"
                    LOGFILE="logs/retrieval/retrieval_${LOG_TAG}_${DATASET}.log"
                    if $IS_SURR; then
                        run_job "retrieval $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode test --seed "$SEED" \
                            --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    else
                        run_job "retrieval $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode test --seed "$SEED" \
                            --split_suffix "$SPLIT_SUFFIX" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    fi

                elif [[ "$ACTUAL_PREFIX" == "order_alpha" ]]; then
                    $IS_SURR && TRAIN_SCRIPT="train_order_alpha_surr.py" || TRAIN_SCRIPT="train_order_alpha.py"
                    LOGFILE="logs/retrieval/retrieval_${LOG_TAG}${ALPHA}_${DATASET}.log"
                    if $IS_SURR; then
                        run_job "retrieval $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode test --seed "$SEED" \
                            --alpha "$ALPHA" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    else
                        run_job "retrieval $LABEL_BASE" "$ORDER_SCRIPTS" \
                            "$TRAIN_SCRIPT" \
                            --dataset "$DATASET" --mode test --seed "$SEED" \
                            --alpha "$ALPHA" --split_suffix "$SPLIT_SUFFIX" --device "$DEVICE" "${BACKBONE_ARG[@]}" \
                            > "$LOGFILE" 2>&1
                    fi
                fi

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
