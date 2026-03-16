#!/bin/bash
# Copy external datasets to EFS for multi-node training access.
# Run this once on the server that has NVMe access, then all nodes can use the data.
#
# Usage:
#   bash scripts/copy_external_data.sh           # copy all
#   bash scripts/copy_external_data.sh annutacon  # copy only annutacon
#   bash scripts/copy_external_data.sh fisher     # retry fisher download
#
# Logs: /tmp/copy_external_*.log
# Safe to re-run — rsync skips already-copied files.

set -e
cd "$(dirname "$0")/.."

TARGET="data/external"
mkdir -p "$TARGET"

copy_annutacon() {
    echo "=== Copying annutacon (NVMe → EFS) ==="
    SRC="/mnt/nvme/advait/annutacon/data"
    DST="$TARGET/annutacon"
    mkdir -p "$DST"

    if [ ! -d "$SRC" ]; then
        echo "[ERROR] Source not found: $SRC"
        return 1
    fi

    # Copy dataset.jsonl
    cp -n "$SRC/dataset.jsonl" "$DST/" 2>/dev/null || true

    # Copy WAVs and alignment JSONs (skip .transcript.json — not needed)
    echo "Copying data_stereo/ (WAV + JSON)..."
    echo "This is ~483GB, will take a while. Log: /tmp/copy_external_annutacon.log"
    rsync -a \
        --include='*.wav' \
        --include='*.json' \
        --exclude='*.transcript.json' \
        "$SRC/data_stereo/" "$DST/data_stereo/" \
        >> /tmp/copy_external_annutacon.log 2>&1

    COPIED=$(ls "$DST/data_stereo/"*.wav 2>/dev/null | wc -l)
    echo "Done. $COPIED WAV files in $DST/data_stereo/"
}

download_fisher() {
    echo "=== Downloading Fisher (Backblaze B2 → EFS) ==="
    DST="$TARGET/fisher"
    mkdir -p "$DST"

    if [ -f "$DST/fisher_wav.tar" ]; then
        SIZE=$(stat -c%s "$DST/fisher_wav.tar" 2>/dev/null || echo 0)
        # Full tar is ~53GB
        if [ "$SIZE" -gt 50000000000 ]; then
            echo "fisher_wav.tar already downloaded ($(du -h "$DST/fisher_wav.tar" | cut -f1))"
        else
            echo "Partial download detected ($(du -h "$DST/fisher_wav.tar" | cut -f1)). Re-downloading..."
            rm -f "$DST/fisher_wav.tar"
        fi
    fi

    if [ ! -f "$DST/fisher_wav.tar" ] || [ "$(stat -c%s "$DST/fisher_wav.tar" 2>/dev/null || echo 0)" -lt 50000000000 ]; then
        echo "Downloading fisher_wav.tar (~53GB)..."
        echo "Log: /tmp/copy_external_fisher.log"
        AWS_ACCESS_KEY_ID=0054ea8aad895950000000002 \
        AWS_SECRET_ACCESS_KEY=K005SOf982O0F/Kg7flqdHECHwcMz6c \
        aws s3 cp s3://audio-datasets/fisher_wav.tar "$DST/fisher_wav.tar" \
            --endpoint-url=https://s3.us-east-005.backblazeb2.com \
            >> /tmp/copy_external_fisher.log 2>&1
    fi

    # Extract if not already extracted
    if [ ! -d "$DST/fisher_wav" ] || [ "$(ls "$DST/fisher_wav/"*.wav 2>/dev/null | wc -l)" -lt 10 ]; then
        echo "Extracting fisher_wav.tar..."
        cd "$DST"
        tar xf fisher_wav.tar
        cd -
    fi

    EXTRACTED=$(ls "$DST/fisher_wav/"*.wav 2>/dev/null | wc -l)
    echo "Done. $EXTRACTED WAV files in $DST/fisher_wav/"
}

# Main
WHAT="${1:-all}"

case "$WHAT" in
    annutacon)
        copy_annutacon
        ;;
    fisher)
        download_fisher
        ;;
    all)
        copy_annutacon &
        ANNU_PID=$!
        download_fisher &
        FISH_PID=$!
        echo ""
        echo "Running in parallel:"
        echo "  annutacon PID: $ANNU_PID  (log: /tmp/copy_external_annutacon.log)"
        echo "  fisher PID:    $FISH_PID  (log: /tmp/copy_external_fisher.log)"
        echo ""
        echo "Wait for both..."
        wait $ANNU_PID
        echo "annutacon: done"
        wait $FISH_PID
        echo "fisher: done"
        ;;
    *)
        echo "Usage: $0 [annutacon|fisher|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Summary ==="
echo "Annutacon: $(ls "$TARGET/annutacon/data_stereo/"*.wav 2>/dev/null | wc -l) WAV files"
echo "Fisher:    $(ls "$TARGET/fisher/fisher_wav/"*.wav 2>/dev/null | wc -l) WAV files"
echo ""
echo "Next steps:"
echo "  python -m pipeline.import_external --dataset annutacon"
echo "  python -m pipeline.import_external --dataset fisher"
echo "  python -m pipeline.merge_manifests --config configs/full_training.yaml --stats_only"
