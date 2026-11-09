"""
run_4s.py — Chay pipeline o CUA SO 4.0 GIAY VAT LY TRON (Benchmark).

Muc dich: co mot moc cua so TRON (chuan linh vuc)
lam ket qua chinh / so sanh SOTA, thay vi con so le 5.44s.

DA SUA LOI WINDOW (xem data/benchmark_loader.py va run_full.py): t_start/t_end
gio la giay tinh TU ONSET KICH THICH THUC (truoc day tinh tu dau buffer 6s).
    - 4.0s vat ly = 1000 mau  =>  t_start=0.14, t_end=4.14
      (sample_start=onset+35=160, sample_end=onset+1035=1160, n_samples=1000)

    So sanh cac moc (tat ca da sua window bug):
    - 1.0s = 250 mau  (t_end=1.14)  — xem run_1s.py
    - 2.0s = 500 mau  (t_end=2.14)  — xem run_2s.py
    - 4.0s = 1000 mau (t_end=4.14)  [CHAY BAY GIO]
    - 5.0s = 1250 mau (t_end=5.14, cau hinh chinh, thay cho "5.44s" cu co bug) — xem run_full.py

Ba buoc: baseline -> naive ensemble -> collaborative.

Chay:  python run_4s.py
"""
import subprocess
import sys
import argparse
from pathlib import Path

# ── THAM SO CUA SO 4.0s VAT LY TRON (DA SUA WINDOW BUG) ──
T_START = "0.14"
T_END = "4.14"          # tinh tu onset thuc -> 1000 mau @ 250Hz = 4.0s vat ly

BASE_DIR = "results/baselines_4s_fixedwindow_round"
ENS_DIR = "results/ensemble_4s_fixedwindow_round"

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
        sys.exit(result.returncode)
    print(f"\n  [OK] Xong: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-naive", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("  CHAY CUA SO 4.0s VAT LY TRON")
    print(f"  t_start={T_START}, t_end={T_END} (1000 mau @ 250Hz)")
    print(f"  Baseline dir: {BASE_DIR}")
    print(f"  Ensemble dir: {ENS_DIR}")
    print("=" * 64)

    if not args.skip_baseline:
        for ext in EXTRACTORS:
            run_step(
                f"[1/3] Train baseline extractor: {ext} (4.0s)",
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
        print("\n[Bo qua] Buoc 1 (baseline)")

    if not args.skip_naive:
        run_step(
            "[2/3] Naive ensemble (4.0s)",
            [PY, "evaluation/naive_ensemble.py",
             "--data_path", "data/raw/benchmark",
             "--checkpoint_dir", BASE_DIR,
             "--n_subjects", "35",
             "--t_start", T_START,
             "--t_end", T_END,
             "--save_dir", ENS_DIR]
        )
    else:
        print("\n[Bo qua] Buoc 2 (naive)")

    run_step(
        "[3/3] Collaborative distillation (4.0s)",
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
    print("  HOAN TAT CUA SO 4.0s")
    print("=" * 64)
    print(f"  Ket qua trong: {BASE_DIR}/ va {ENS_DIR}/")
    print()
    print("  BUOC TIEP: python compute_metrics_4s.py")
    print("  (ITR @4s dung T = 4.0 + 0.5 = 4.5s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
