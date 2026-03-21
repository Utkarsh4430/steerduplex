# Voice Pools

The following voice datasets are available for use in this project.

| Dataset | Location | Download |
|---|---|---|
| Fisher | Backblaze B2 (`s3://audio-datasets/steer_duplex/voices/audios/fisher_audios.tar`) | See below |
| VoxPopuli | Backblaze B2 (`s3://audio-datasets/steer_duplex/voices/audios/voxpopuli_audios.tar`) | See below |
| Common Voice | Google Drive | See below |
| LibriSpeech | Google Drive | See below |

## Download Instructions

**Fisher / VoxPopuli** (Backblaze B2):
```bash
aws s3 cp s3://audio-datasets/steer_duplex/voices/audios/fisher_audios.tar . --endpoint-url=https://s3.us-east-005.backblazeb2.com
aws s3 cp s3://audio-datasets/steer_duplex/voices/audios/voxpopuli_audios.tar . --endpoint-url=https://s3.us-east-005.backblazeb2.com
```

**Common Voice / LibriSpeech** (Google Drive via `gdown`):
```bash
pip install gdown  # if not already installed

gdown https://drive.google.com/uc?id=1ZI25NHSZxHRubi-d-_qKKazMT9nyg4_D  # Common Voice
gdown https://drive.google.com/uc?id=1EfvOItGfQICtR-vAO71tw-OQzevxkRUq  # LibriSpeech
```