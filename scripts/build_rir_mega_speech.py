from __future__ import annotations
import argparse
from pathlib import Path
import random
import json
import pandas as pd
from tqdm import tqdm

from scripts.audio_utils import load_audio_mono, write_wav_int16, convolve_time
from scripts.metrics import rt60_schroeder, drr_direct_window, c50_from_rir

def discover_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".flac"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def infer_clean_id(path: Path) -> str:
    return path.stem

def infer_rir_id(path: Path) -> str:
    return path.stem

def assign_split(clean_id: str) -> str:
    import hashlib
    h = hashlib.md5(clean_id.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    if u < 0.82:
        return "train"
    elif u < 0.907:
        return "dev"
    else:
        return "test"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", type=str, required=True)
    ap.add_argument("--rir_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--total_outputs", type=int, default=53234)
    ap.add_argument("--max_variants_per_clean", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    rir_root = Path(args.rir_root)
    out_root = Path(args.out_root)

    clean_files = discover_audio_files(clean_root)
    rir_files = discover_audio_files(rir_root)

    if not clean_files:
        raise SystemExit(f"No clean speech files found under: {clean_root}")
    if not rir_files:
        raise SystemExit(f"No RIR files found under: {rir_root}")

    random.seed(args.seed)
    clean_files = sorted(clean_files)
    rir_files = sorted(rir_files)
    random.shuffle(clean_files)
    random.shuffle(rir_files)

    print(f"Computing metrics for {len(rir_files)} RIRs...")
    rir_metrics = []
    for rp in tqdm(rir_files):
        h, sr = load_audio_mono(rp, target_sr=args.sr)
        rt60 = rt60_schroeder(h, sr)
        drr = drr_direct_window(h, sr, window_ms=2.5)
        c50 = c50_from_rir(h, sr)
        rir_metrics.append((rp, rt60, drr, c50))

    out_audio_base = out_root / "audio" / "train"
    meta_root = out_root / "metadata"
    stats_root = out_root / "stats"
    meta_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    rows = []
    produced = 0
    shard_idx = 0
    in_shard = 0
    rir_ptr = 0

    pbar = tqdm(total=args.total_outputs, desc="Generating")
    for cp in clean_files:
        if produced >= args.total_outputs:
            break
        clean_id = infer_clean_id(cp)
        split = assign_split(clean_id)

        for _ in range(args.max_variants_per_clean):
            if produced >= args.total_outputs:
                break

            rp, rt60, drr, c50 = rir_metrics[rir_ptr % len(rir_metrics)]
            rir_ptr += 1
            rir_id = infer_rir_id(rp)

            if in_shard >= args.shard_size:
                shard_idx += 1
                in_shard = 0

            shard_name = f"shard_{shard_idx:03d}"
            out_name = f"{clean_id}__{rir_id}__{produced:06d}.wav"
            rel_audio = Path("audio") / "train" / shard_name / out_name
            out_path = out_root / rel_audio

            if not args.dry_run:
                x, sr = load_audio_mono(cp, target_sr=args.sr)
                h, _ = load_audio_mono(rp, target_sr=args.sr)
                y = convolve_time(x, h)

                duration_s = float(len(y) / sr)

                try:
                    import pyloudnorm as pyln
                    meter = pyln.Meter(sr)
                    lufs = float(meter.integrated_loudness(y))
                except Exception:
                    lufs = float("nan")

                write_wav_int16(out_path, y, sr)
            else:
                duration_s = float("nan")
                lufs = float("nan")

            rows.append({
                "audio": str(rel_audio).replace("\\", "/"),
                "rt60": float(rt60),
                "drr": float(drr),
                "c50": float(c50),
                "lufs": float(lufs),
                "duration_s": float(duration_s),
                "clean_id": clean_id,
                "rir_id": rir_id,
                "split": split,
            })

            produced += 1
            in_shard += 1
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(meta_root / "metadata.csv", index=False)
    for s in ["train","dev","test"]:
        df[df["split"] == s].to_csv(meta_root / f"{s}.csv", index=False)

    summary = {
        "seed": args.seed,
        "sr": args.sr,
        "total_outputs": produced,
        "max_variants_per_clean": args.max_variants_per_clean,
        "shard_size": args.shard_size,
        "num_clean_files_found": len(clean_files),
        "num_rir_files_found": len(rir_files),
        "dry_run": args.dry_run,
    }
    (stats_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Outputs at: {out_root}")

if __name__ == "__main__":
    main()
