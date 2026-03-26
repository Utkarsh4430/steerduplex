#!/bin/bash
set -euo pipefail
FISHER_DIR="data/external/fisher"
mkdir -p "$FISHER_DIR"

export AWS_ACCESS_KEY_ID_B2="0054ea8aad895950000000002"
export AWS_SECRET_ACCESS_KEY_B2="K005SOf982O0F/Kg7flqdHECHwcMz6c"
B2_EP="--endpoint-url=https://s3.us-east-005.backblazeb2.com"

download_resumable() {
    local s3_key="$1" dst="$2"
    # Skip if valid tar exists
    if [ -f "$dst" ] && tar tf "$dst" > /dev/null 2>&1; then
        echo "$(basename $dst): valid tar exists. Skipping."
        return 0
    fi
    for attempt in $(seq 1 20); do
        echo "[Attempt $attempt] $(basename $dst) — generating presigned URL..."
        local url
        url=$(AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_B2 AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_B2 \
            aws s3 presign "s3://audio-datasets/$s3_key" $B2_EP --expires-in 86400)
        [ -f "$dst" ] && echo "  Resuming from $(numfmt --to=iec $(stat -c%s "$dst"))"
        if curl -C - -L -o "$dst" --retry 3 --retry-delay 15 \
            --speed-limit 50000 --speed-time 120 --progress-bar "$url"; then
            echo "SUCCESS: $(ls -lh "$dst" | awk '{print $5}')"
            return 0
        fi
        echo "  Partial: $(numfmt --to=iec $(stat -c%s "$dst" 2>/dev/null || echo 0)). Waiting 60s..."
        sleep 60
    done
    return 1
}

echo "=== Fisher P1 (resumable) ==="
download_resumable "steer_duplex/training_datasets/fisher_p1_wav.tar" "$FISHER_DIR/fisher_p1_wav.tar"

echo "Cooling down 30s..."
sleep 30

echo "=== Fisher P2 (resumable) ==="
download_resumable "steer_duplex/training_datasets/fisher_p2_wav.tar" "$FISHER_DIR/fisher_p2_wav.tar"

echo ""
echo "=== Extracting ==="
cd "$FISHER_DIR"
tar xf fisher_p1_wav.tar && echo "P1 extracted"
tar xf fisher_p2_wav.tar && echo "P2 extracted"

echo ""
echo "=== Copying to scale-ml (no rate limits for future use) ==="
AWS_PROFILE=production-developer aws s3 cp fisher_p1_wav.tar s3://scale-ml/genai/utkarsh/steerduplex/fisher_p1_wav.tar &
AWS_PROFILE=production-developer aws s3 cp fisher_p2_wav.tar s3://scale-ml/genai/utkarsh/steerduplex/fisher_p2_wav.tar &
wait
echo "Copied to scale-ml"

echo ""
echo "Next:"
echo "  python -m pipeline.import_external --dataset fisher"
echo "  python -m pipeline.add_prompts_external --dataset fisher"
