from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    meta = out_root / "metadata" / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"Missing: {meta}")

    df = pd.read_csv(meta)
    required = ["audio","rt60","drr","c50","lufs","duration_s","clean_id","rir_id","split"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}")

    # spot-check file existence
    missing = 0
    for rel in df["audio"].head(2000):
        if not (out_root / rel).exists():
            missing += 1
    print(f"Checked 2000 audio paths. Missing: {missing}")
    print(df["split"].value_counts())

if __name__ == "__main__":
    main()
