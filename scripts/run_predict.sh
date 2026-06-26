#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

[[ -f predict_hparams.sh ]] && source predict_hparams.sh

CLI_METHODS=""; CLI_DATASETS=""; CLI_MODALITIES=""; CLI_DEVICE=""; CLI_SEED=""
CLI_N_EPOCHS=""; CLI_LR=""; CLI_DROPOUT=""; CLI_WEIGHT_DECAY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)       CLI_METHODS="$2";       shift 2 ;;
        --datasets)      CLI_DATASETS="$2";      shift 2 ;;
        --modalities)    CLI_MODALITIES="$2";    shift 2 ;;
        --device)        CLI_DEVICE="$2";        shift 2 ;;
        --seed)          CLI_SEED="$2";          shift 2 ;;
        --n_epochs)      CLI_N_EPOCHS="$2";      shift 2 ;;
        --lr)            CLI_LR="$2";            shift 2 ;;
        --dropout)       CLI_DROPOUT="$2";       shift 2 ;;
        --weight_decay)  CLI_WEIGHT_DECAY="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

NON_INTERACTIVE=false
[[ -n "$CLI_METHODS" && -n "$CLI_DATASETS" && -n "$CLI_MODALITIES" ]] && NON_INTERACTIVE=true

SEED="${CLI_SEED:-}"

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

get_hp() {
    local param="${1^^}" ds="${2^^}" modal="${3^^}"
    local task_key="HP_${param}_${ds}_${modal}"
    local ds_key="HP_${param}_${ds}"
    local modal_key="HP_${param}_${modal}"
    local global_key="HP_${param}"
    echo "${!task_key:-${!ds_key:-${!modal_key:-${!global_key:-}}}}"
}

declare -A RT_HPARAMS=()

set_rt_hp() { RT_HPARAMS["${1}__${2}__${3}"]="$4"; }
get_rt_hp() {
    local key="${1}__${2}__${3}"
    echo "${RT_HPARAMS[$key]:-}"
}

resolve_hp() {
    local param="$1" ds="$2" modal="$3"
    local rt; rt="$(get_rt_hp "$param" "$ds" "$modal")"
    if   [[ -n "$rt" ]];                                                then echo "$rt"
    elif [[ "$param" == "N_EPOCHS"     && -n "$CLI_N_EPOCHS"     ]];   then echo "$CLI_N_EPOCHS"
    elif [[ "$param" == "LR"           && -n "$CLI_LR"           ]];   then echo "$CLI_LR"
    elif [[ "$param" == "DROPOUT"      && -n "$CLI_DROPOUT"      ]];   then echo "$CLI_DROPOUT"
    elif [[ "$param" == "WEIGHT_DECAY" && -n "$CLI_WEIGHT_DECAY" ]];   then echo "$CLI_WEIGHT_DECAY"
    else get_hp "$param" "$ds" "$modal"
    fi
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

if [[ -z "$CLI_SEED" ]]; then
    header "Seed"
    printf "  Enter pretrained-checkpoint seed [default: 0]: "
    read -r SEED; SEED="${SEED:-0}"
else
    SEED="$CLI_SEED"
fi
echo "  Using seed: ${BOLD}${SEED}${RESET}"

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
             "order_dyn_surr_vit16"
             "order_alpha_surr_vit16:0.2"
             "order_alpha_surr_vit16:0.5"
             "order_alpha_surr_vit16:0.9")

if $NON_INTERACTIVE; then
    SELECTED_METHODS="$CLI_METHODS"
else
    multi_select SELECTED_METHODS "Methods:" "${ALL_METHODS[*]}" "1 3 5 7 8 10 11 13 15 16 18 19"
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

if $NON_INTERACTIVE; then
    SELECTED_MODALITIES="$CLI_MODALITIES"
else
    multi_select SELECTED_MODALITIES "Modalities:" "tab image fusion" "a"
fi
[[ -z "$SELECTED_MODALITIES" ]] && { warn "No modalities selected. Exiting."; exit 1; }
echo "  Modalities: ${BOLD}${SELECTED_MODALITIES}${RESET}"

PARAMS=(N_EPOCHS LR WEIGHT_DECAY DROPOUT)

print_hp_table() {
    printf "\n  %-12s %-8s  %-10s %-10s %-13s %-10s\n" \
        "Dataset" "Modal" "n_epochs" "lr" "weight_decay" "dropout"
    printf "  %s\n" "$(printf '─%.0s' {1..65})"
    for ds in $SELECTED_DATASETS; do
        for modal in $SELECTED_MODALITIES; do
            printf "  %-12s %-8s  %-10s %-10s %-13s %-10s\n" \
                "$ds" "$modal" \
                "$(resolve_hp N_EPOCHS     "$ds" "$modal")" \
                "$(resolve_hp LR           "$ds" "$modal")" \
                "$(resolve_hp WEIGHT_DECAY "$ds" "$modal")" \
                "$(resolve_hp DROPOUT      "$ds" "$modal")"
        done
    done
}

header "Hyperparameters"
print_hp_table

if ! $NON_INTERACTIVE; then
    printf "\n  Edit hyperparameters? [y/N]: "
    read -r edit_hp
    if [[ "$edit_hp" =~ ^[Yy] ]]; then
        echo -e "  (Press ${BOLD}Enter${RESET} to keep current value)"
        for ds in $SELECTED_DATASETS; do
            for modal in $SELECTED_MODALITIES; do
                echo -e "\n  ${BOLD}${ds} / ${modal}${RESET}"
                for param in "${PARAMS[@]}"; do
                    cur="$(resolve_hp "$param" "$ds" "$modal")"
                    printf "    %-14s [%s]: " "${param,,}" "$cur"
                    read -r val
                    [[ -n "$val" ]] && set_rt_hp "$param" "$ds" "$modal" "$val"
                done
            done
        done
        echo ""
        header "Updated hyperparameters"
        print_hp_table
    fi
fi

header "Summary"
echo "  Device:     $DEVICE"
echo "  Methods:    $SELECTED_METHODS"
echo "  Datasets:   $SELECTED_DATASETS"
echo "  Modalities: $SELECTED_MODALITIES"

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
    local script="$1"; shift
    echo -e "\n${BOLD}▶ $label${RESET}"
    if "$PYTHON" "$script" "$@"; then
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

    for DATASET in $SELECTED_DATASETS; do

        LABEL_SUFFIX="$ACTUAL_PREFIX"
        [[ -n "$ALPHA" ]] && LABEL_SUFFIX="${ACTUAL_PREFIX}(α=${ALPHA})"
        $IS_SURR  && LABEL_SUFFIX="${LABEL_SUFFIX}[surr]"
        $IS_VIT16 && LABEL_SUFFIX="${LABEL_SUFFIX}[vit16]"

        BASE_ARGS=(--prefix "$ACTUAL_PREFIX" --dataset "$DATASET" --device "$DEVICE" --seed "$SEED")
        [[ -n "$ALPHA" ]] && BASE_ARGS+=(--alpha "$ALPHA")
        $IS_SURR && BASE_ARGS+=(--split_suffix _surr)
        $IS_VIT16 && BASE_ARGS+=(--backbone "ViT-B/16")

        for MODAL in $SELECTED_MODALITIES; do
            LABEL="${LABEL_SUFFIX} / ${DATASET} / ${MODAL}"

            HP_ARGS=(
                --n_epochs     "$(resolve_hp N_EPOCHS     "$DATASET" "$MODAL")"
                --lr           "$(resolve_hp LR           "$DATASET" "$MODAL")"
                --weight_decay "$(resolve_hp WEIGHT_DECAY "$DATASET" "$MODAL")"
                --dropout      "$(resolve_hp DROPOUT      "$DATASET" "$MODAL")"
            )

            if [[ "$MODAL" == "fusion" ]]; then
                run_job "$LABEL" fusion_predict.py "${BASE_ARGS[@]}" "${HP_ARGS[@]}"
            else
                run_job "$LABEL" predict.py "${BASE_ARGS[@]}" --modal "$MODAL" "${HP_ARGS[@]}"
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
