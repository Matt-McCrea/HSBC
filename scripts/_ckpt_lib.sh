# _ckpt_lib.sh — sourced by exp2/exp3. Resolves the confirmed checkpoint path from
# analysis/model_under_test.md so nothing needs to be typed by hand -- exp1_pin_checkpoint.sh
# writes the CKPT_PATH= line there automatically once it finds an unambiguous match.
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
