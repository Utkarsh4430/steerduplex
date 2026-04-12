#!/bin/bash

cd /fs/gamma-projects/audio/raman/steerd/steerduplex/src/eval/fdb_v2

# bash run_dataset.sh prompts_staged_200.json --base-out ./outputs --adapter-b adapters/gptRealtime_adapter.js --examiner-mode fast --limit 2

bash run_dataset.sh prompts_staged_200.json --base-out ./moshi_outputs_fast_examiner --adapter-b adapters/moshi_adapter.js --examiner-mode fast