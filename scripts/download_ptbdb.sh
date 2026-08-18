#!/usr/bin/env bash
# Downloads the PTB Diagnostic ECG Database 1.0.0 from PhysioNet into data/raw/ptbdb/.
# Fully open access — no PhysioNet account or agreement needed.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw/ptbdb"
mkdir -p "$DEST"

wget -r -N -c -np -nH --cut-dirs=3 -P "$DEST" \
  https://physionet.org/files/ptbdb/1.0.0/

echo "PTB Diagnostic ECG Database downloaded to $DEST"
