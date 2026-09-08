# _ckpt_lib.sh — sourced by exp2/exp3. Resolves a confirmed checkpoint path from
# analysis/model_under_test.md so nothing needs to be typed by hand -- exp1_pin_checkpoint.sh (a
# single pre-existing checkpoint, CKPT_PATH=) or pin_trained_variant.sh (one of the three trained
# variants, CKPT_PATH_<VARIANT>=) write these lines there automatically.
resolve_ckpt_path () {  # resolve_ckpt_path <override-or-empty>
  local OVERRIDE="${1:-}"
  if [[ -n "$OVERRIDE" ]]; then echo "$OVERRIDE"; return 0; fi
  local LINE
  LINE=$(grep '^CKPT_PATH=' analysis/model_under_test.md 2>/dev/null | tail -1)
  if [[ -z "$LINE" ]]; then
    echo "!! no CKPT_PATH in analysis/model_under_test.md yet. Run:" >&2
    echo "!!   bash scripts/exp1_pin_checkpoint.sh" >&2
    return 1
  fi
  echo "${LINE#CKPT_PATH=}"
}

resolve_ckpt_path_for_variant () {  # resolve_ckpt_path_for_variant <baseline|reanchor|ss>
  local VARIANT="$1" VAR_UPPER
  VAR_UPPER=$(echo "$VARIANT" | tr '[:lower:]' '[:upper:]')
  local LINE
  LINE=$(grep "^CKPT_PATH_${VAR_UPPER}=" analysis/model_under_test.md 2>/dev/null | tail -1)
  if [[ -z "$LINE" ]]; then
    echo "!! no CKPT_PATH_${VAR_UPPER} in analysis/model_under_test.md yet. Run:" >&2
    echo "!!   bash scripts/train_variant.sh $VARIANT   (then pin_trained_variant.sh $VARIANT)" >&2
    return 1
  fi
  echo "${LINE#CKPT_PATH_${VAR_UPPER}=}"
}

# reanchor_for_variant <baseline|reanchor|ss> -- the price-reanchoring flag state a checkpoint was
# TRAINED with; simulating with a different state than training would evaluate the model
# out-of-distribution on its own conditioning, so this is not an independent free choice per run.
reanchor_for_variant () {
  case "$1" in
    reanchor) echo "on" ;;
    *)        echo "off" ;;
  esac
}
