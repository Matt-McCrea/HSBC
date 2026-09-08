#!/bin/bash
# pin_trained_variant.sh — after scripts/train_variant.sh <variant> finishes, picks the checkpoint
# to use (highest epoch available -- "epoch 1, at most epoch 2" means prefer epoch=1 if it saved,
# else epoch=0) and writes CKPT_PATH_<VARIANT>=... into analysis/model_under_test.md
# automatically. Nothing to type or paste -- exp2/exp3 read it from there via --variant.
#
# ONE COMMAND: bash scripts/pin_trained_variant.sh <baseline|reanchor|ss>
set -uo pipefail
VARIANT="${1:?usage: pin_trained_variant.sh <baseline|reanchor|ss>}"
DEST_DIR="data/checkpoints/TRADES_${VARIANT}"

# portable (no mapfile/readarray -- avoids depending on bash >= 4)
N=0
for f in "$DEST_DIR"/*.ckpt; do [[ -f "$f" ]] && N=$((N + 1)); done
[[ "$N" -gt 0 ]] || { echo "!! no checkpoints in $DEST_DIR -- did training finish? check the train_${VARIANT}_*.log"; exit 1; }

# prefer the highest epoch number present (epoch=1 over epoch=0, i.e. "epoch two" over "epoch one")
BEST=""; BEST_EPOCH=-1
echo "== $VARIANT: $N checkpoint(s) found, picking highest epoch =="
for f in "$DEST_DIR"/*.ckpt; do
  [[ -f "$f" ]] || continue
  echo "  $f"
  EP=$(basename "$f" | sed -nE 's/.*_epoch=([0-9]+)_.*/\1/p')
  [[ -n "$EP" && "$EP" -gt "$BEST_EPOCH" ]] && { BEST_EPOCH="$EP"; BEST="$f"; }
done
[[ -n "$BEST" ]] || { echo "!! found files but couldn't parse an epoch number from any of them"; exit 1; }
echo "  -> selected: $BEST  (epoch=$BEST_EPOCH)"

VAR_UPPER=$(echo "$VARIANT" | tr '[:lower:]' '[:upper:]')
python3 - "$VAR_UPPER" "$BEST" <<'PY'
import re, sys
var, path, doc = sys.argv[1], sys.argv[2], "analysis/model_under_test.md"
key = f"CKPT_PATH_{var}"
text = open(doc).read()
line = f"{key}={path}"

# Scoped edits only -- this file has content (a "Superseded" section) AFTER the "Confirmed
# checkpoints" heading, so anything using re.DOTALL or "append to end of text" would silently
# eat everything below it. Three cases, most-specific first:
own_line = re.compile(rf"^{re.escape(key)}=.*$", re.M)
placeholder = "*(empty until training finishes on the box that's actually running it)*"
existing_block = re.compile(r"(?:^CKPT_PATH_\w+=.*$\n?)+", re.M)

if own_line.search(text):
    text = own_line.sub(line, text)
elif placeholder in text:
    text = text.replace(placeholder, line, 1)
elif existing_block.search(text):
    m = existing_block.search(text)
    text = text[:m.end()] + line + "\n" + text[m.end():]
elif "## Confirmed checkpoints" in text:
    text = text.replace("## Confirmed checkpoints\n", "## Confirmed checkpoints\n\n" + line + "\n", 1)
else:
    text = text.rstrip("\n") + "\n\n## Confirmed checkpoints\n\n" + line + "\n"
open(doc, "w").write(text)
print(f"  wrote {key}={path} into {doc}")
PY

echo ""
echo "Next: bash scripts/exp2_survival_sweep.sh --smoke --variant $VARIANT"
