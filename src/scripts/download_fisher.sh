#!/bin/bash
# Download Fisher dataset from Backblaze B2 using rclone.
# rclone handles B2 throttling with chunked transfers + auto-retry.
#
# Usage:
#   nohup bash scripts/download_fisher.sh &
#   tail -f /tmp/fisher_download.log

set -e
cd "$(dirname "$0")/.."

DST="data/external/fisher"
TAR="$DST/fisher_wav.tar"
LOG="/tmp/fisher_download.log"

mkdir -p "$DST"

echo "$(date): Starting Fisher download via rclone" | tee "$LOG"

# rclone with inline B2 config — no config file needed
rclone copyto \
    :s3,provider=Other,access_key_id=0054ea8aad895950000000002,secret_access_key=K005SOf982O0F/Kg7flqdHECHwcMz6c,endpoint=s3.us-east-005.backblazeb2.com:audio-datasets/fisher_wav.tar \
    "$TAR" \
    --progress \
    --retries 30 \
    --retries-sleep 30s \
    --low-level-retries 10 \
    --s3-chunk-size 64M \
    --s3-upload-concurrency 1 \
    --transfers 1 \
    --multi-thread-streams 4 \
    2>&1 | tee -a "$LOG"

echo "$(date): Download complete. Size: $(du -h "$TAR" | cut -f1)" | tee -a "$LOG"

# Extract
echo "$(date): Extracting..." | tee -a "$LOG"
cd "$DST"
tar xf fisher_wav.tar
cd -
echo "$(date): Done. $(ls "$DST/fisher_wav/"*.wav 2>/dev/null | wc -l) WAV files" | tee -a "$LOG"
