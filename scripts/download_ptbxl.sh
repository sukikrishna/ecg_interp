#!/usr/bin/env bash
# Downloads PTB-XL 1.0.3 from PhysioNet into data/raw/ptb-xl/.
# Fully open access — no PhysioNet account or agreement needed.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw/ptb-xl"
mkdir -p "$DEST"

wget -r -N -c -np -nH --cut-dirs=3 -P "$DEST" \
  https://physionet.org/files/ptb-xl/1.0.3/

echo "PTB-XL downloaded to $DEST"
