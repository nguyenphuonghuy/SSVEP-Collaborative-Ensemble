"""
run_2s.py — Chay pipeline o CUA SO 2.0 GIAY VAT LY (Benchmark).

Diem con thieu trong bo quet cua so cua Bang 5 (main.tex): 1.0 / 2.0 / 4.0 / 5.44s.
1.0s co run_1s.py, 4.0s co run_4s.py, 5.44s tai dung ket qua Bang 2/3 (baselines_full).
Cua so 2.0s truoc day tung bi goi nham la "0.5s" trong code cu — file nay dat ten
ro rang theo dung cua so vat ly.

DA SUA LOI WINDOW (xem data/benchmark_loader.py va run_full.py): t_start/t_end
gio la giay tinh TU ONSET KICH THICH THUC.
    - 2.0s vat ly = 500 mau  =>  t_start=0.14, t_end=2.14
      (sample_start=onset+35=160, sample_end=onset+535=660, n_samples=500)

Chi train 3 nhanh chinh thuc (temporal_cnn, freq_domain_cnn, spatial_temporal_cnn) —
KHONG train shallow_conv_net/filter_bank_cnn (khong can cho Bang 5, tranh phi compute).

Batch size = 16 giu nguyen theo dung Bang sieu tham so trong bai bao (khong doi de
tan dung H100 — xem quyet dinh 2026-08-04: uu tien khop chinh xac voi so da cong bo
hon la toc do).

Ba buoc: baseline -> naive ensemble -> collaborative.

Chay:  python run_2s.py
       python run_2s.py --skip-baseline     (neu da train baseline roi)
"""
import subprocess
import sys
import argparse
from pathlib import Path

# ── THAM SO CUA SO 2.0s VAT LY (DA SUA WINDOW BUG) ──
T_START = "0.14"
T_END = "2.14"          # tinh tu onset thuc -> 500 mau @ 250Hz = 2.0s vat ly

BASE_DIR = "results/baselines_2s_fixedwindow_round"
ENS_DIR = "results/ensemble_2s_fixedwindow_round"

EXTRACTORS = ["temporal_cnn", "freq_domain_cnn", "spatial_temporal_cnn"]

PY = sys.executable


def run_step(name, cmd):
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
                    help="Bo qua buoc train baseline (neu da co results/baselines_2s)")
    ap.add_argument("--skip-naive", action="store_true",
                    help="Bo qua buoc naive ensemble")
    args = ap.parse_args()

    print("=" * 64)
    print("  CHAY CUA SO 2.0s VAT LY")
    print(f"  t_start={T_START}, t_end={T_END} (500 mau @ 250Hz)")
    print(f"  Python: {PY}")
    print(f"  Baseline dir: {BASE_DIR}")
    print(f"  Ensemble dir: {ENS_DIR}")
    print("=" * 64)

    # ── BUOC 1: Train 3 baseline extractors ──
    if not args.skip_baseline:
        for ext in EXTRACTORS:
            run_step(
                f"[1/3] Train baseline extractor: {ext} (2s)",
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
            "[2/3] Naive ensemble (2s)",
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
        "[3/3] Collaborative distillation (2s)",
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

    print("\n" + "=" * 64)
    print("  HOAN TAT CUA SO 2.0s")
    print("=" * 64)
    print(f"  Ket qua trong:")
    print(f"    - {BASE_DIR}/  (3 baseline)")
    print(f"    - {ENS_DIR}/   (naive + collaborative)")
    print()
    print("  (ITR @2s dung T = 2.0 + 0.5 gaze = 2.5s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
