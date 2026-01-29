# RIR-Mega-Speech Builder (HF-ready)

This repo lets you **recreate the full RIR-Mega-Speech reverberant corpus** (≈53k WAVs) from:
- `data/clean_speech/` (LibriSpeech WAV/FLAC audio)
- `data/rirmega_rirs/` (RIR-Mega RIR WAVs)

It produces:
- Sharded audio folders under `outputs/rir-mega-speech/audio/train/shard_XXX/` (keeps <10,000 files per folder for Hugging Face)
- `outputs/rir-mega-speech/metadata/metadata.csv` with schema:
  `audio, rt60, drr, c50, lufs, duration_s, clean_id, rir_id, split`

Then you can upload `outputs/rir-mega-speech/` to Hugging Face using Git LFS.

## Setup (Windows / PyCharm)

```powershell
python -m pip install -r requirements.txt
```

## Expected input layout

```
data/
├── clean_speech/
└── rirmega_rirs/
```

## Build

```powershell
python scripts/build_rir_mega_speech.py `
  --clean_root data/clean_speech `
  --rir_root data/rirmega_rirs `
  --out_root outputs/rir-mega-speech `
  --total_outputs 53200 `
  --max_variants_per_clean 10 `
  --seed 42
```

Dry run:

```powershell
python scripts/build_rir_mega_speech.py --clean_root data/clean_speech --rir_root data/rirmega_rirs --out_root outputs/rir-mega-speech --dry_run
```

Validate:

```powershell
python scripts/validate_outputs.py --out_root outputs/rir-mega-speech
```

## Output structure

```
outputs/rir-mega-speech/
├── audio/train/shard_000/*.wav
├── metadata/metadata.csv
└── stats/*.json
```

## Upload to Hugging Face

From `outputs/rir-mega-speech/`:

```powershell
huggingface-cli login
git lfs install
git lfs track "*.wav"
git init
git remote add origin https://huggingface.co/datasets/mandipgoswami/rir-mega-speech
git add .
git commit -m "Add audio + metadata"
git push -u origin main
```


