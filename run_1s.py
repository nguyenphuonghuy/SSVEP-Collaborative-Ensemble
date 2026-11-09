"""
run_1s.py — Chay pipeline o CUA SO 1.0 GIAY VAT LY that (Benchmark).

Muc dich: co con so so sanh CONG BANG voi SSVEPformer (cross-subject @1s = 80.4%).

DA SUA LOI WINDOW (xem data/benchmark_loader.py va run_full.py): t_start/t_end
gio la giay tinh TU ONSET KICH THICH THUC (truoc day tinh tu dau buffer 6s,
lech mat ~0.14s). Voi quy uoc moi:
    - 1.0s vat ly = 250 mau  =>  t_start=0.14, t_end=1.14
    - (2.0s: t_end=2.14; 4.0s: t_end=4.14; 5.0s/main: t_end=5.14 — xem run_2s.py,
      run_4s.py, run_full.py)

Script chay 3 buoc tuan tu: baseline -> naive ensemble -> collaborative.

Chay:  python run_1s.py
       python run_1s.py --skip-baseline     (neu da train baseline roi)
"""
import subprocess
import sys
import argparse
from pathlib import Path

# ── THAM SO CUA SO 1.0s VAT LY (DA SUA WINDOW BUG) ──
T_START = "0.14"
T_END = "1.14"          # tinh tu onset thuc -> 250 mau @ 250Hz = 1.0s vat ly

# ── THU MUC KET QUA (suffix _fixedwindow_round de khong de len ket qua cu co bug) ──
BASE_DIR = "results/baselines_1s_fixedwindow_round"
ENS_DIR = "results/ensemble_1s_fixedwindow_round"

EXTRACTORS = ["temporal_cnn", "freq_domain_cnn", "spatial_temporal_cnn"]

# ── Python interpreter dang chay (dam bao dung venv) ──
PY = sys.executable


def run_step(name, cmd):
    """Chay 1 buoc, in ro rang, dung neu loi."""
    print("\n" + "=" * 64)
    print(f"  {name}")
    print("=" * 64)
    print("  Lenh:", " ".join(cmd))
    print("-" * 64)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] LOI o buoc: {name} (return code {result.returncode})")
        print("    Dung lai. Kiem tra loi o tren.")
        sys.exit(result.returncode)
    print(f"\n  [OK] Xong: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Bo qua buoc train baseline (neu da co results/baselines_1s)")
    ap.add_argument("--skip-naive", action="store_true",
                    help="Bo qua buoc naive ensemble")
    args = ap.parse_args()

    print("=" * 64)
    print("  CHAY CUA SO 1.0s VAT LY")
    print(f"  t_start={T_START}, t_end={T_END} (tinh tu onset thuc -> 250 mau @ 250Hz = 1.0s)")
    print(f"  Python: {PY}")
    print(f"  Baseline dir: {BASE_DIR}")
    print(f"  Ensemble dir: {ENS_DIR}")
    print("=" * 64)

    # ── BUOC 1: Train 3 baseline extractors ──
    if not args.skip_baseline:
        for ext in EXTRACTORS:
            run_step(
                f"[1/3] Train baseline extractor: {ext} (1s)",
                [PY, "training/train_extractors.py",
                 "--extractor", ext,
                 "--data_path", "data/raw/benchmark",
                 "--n_subjects", "35",
                 "--epochs", "100",
                 "--batch_size", "16",
                 "--t_start", T_START,
                 "--t_end", T_END,
                 "--save_dir", BASE_DIR]
            )
    else:
        print("\n[Bo qua] Buoc 1 (baseline) theo yeu cau --skip-baseline")

    # ── BUOC 2: Naive ensemble ──
    if not args.skip_naive:
        run_step(
            "[2/3] Naive ensemble (1s)",
            [PY, "evaluation/naive_ensemble.py",
             "--data_path", "data/raw/benchmark",
             "--checkpoint_dir", BASE_DIR,
             "--n_subjects", "35",
             "--t_start", T_START,
             "--t_end", T_END,
             "--save_dir", ENS_DIR]
        )
    else:
        print("\n[Bo qua] Buoc 2 (naive) theo yeu cau --skip-naive")

    # ── BUOC 3: Collaborative distillation ──
    run_step(
        "[3/3] Collaborative distillation (1s)",
        [PY, "training/collaborative_ensemble.py",
         "--data_path", "data/raw/benchmark",
         "--init_checkpoint_dir", BASE_DIR,
         "--n_subjects", "35",
         "--epochs", "50",
         "--batch_size", "16",
         "--alpha", "0.5",
         "--temperature", "4.0",
         "--t_start", T_START,
         "--t_end", T_END,
         "--save_dir", ENS_DIR]
    )

    # ── Tong ket ──
    print("\n" + "=" * 64)
    print("  HOAN TAT CUA SO 1.0s")
    print("=" * 64)
    print(f"  Ket qua trong:")
    print(f"    - {BASE_DIR}/  (3 baseline)")
    print(f"    - {ENS_DIR}/   (naive + collaborative)")
    print()
    print("  BUOC TIEP: python compute_metrics_1s.py")
    print("  (ITR @1s dung T = 1.0 + 0.5 gaze = 1.5s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
