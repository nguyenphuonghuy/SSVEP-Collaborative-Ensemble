"""
run_full.py — Chay pipeline o CUA SO CHINH 5.0s VAT LY (Benchmark), DA SUA LOI
WINDOW (xem data/benchmark_loader.py).

BUG DA PHAT HIEN VA SUA (khong con dung nua): BenchmarkDataset truoc day cat
mau bang self.sfreq noi bo mac dinh =1000, KHONG phai sfreq that cua du lieu
(250Hz), VA khong cong offset onset thuc (125 mau = 0.5s cue truoc kich thich).
Voi t_start=0.14, t_end=4.14 (cu), ket qua la sample_start=140 (~0.06s SAU onset
thuc, khong phai 0.14s nhu mo ta), sample_end bi cap cung o 1500 (het buffer)
=> n_samples=1360 => "5.44s" — con so nay la SAN PHAM NGAU NHIEN cua bug (cap o
cuoi buffer), KHONG phai gia tri duoc chon co chu dich, va cua so nay AN het
0.5s hau-kich-thich (vuot qua moc tat kich thich thuc 0.5s).

DA SUA (data/benchmark_loader.py): sfreq mac dinh 1000->250 (dung thuc te), va
sample_start/sample_end gio tinh tu ONSET KICH THICH THUC (them offset 125 mau
= 0.5s cue). t_start/t_end tu day la giay tinh TU ONSET THUC, dung quy uoc
chuan trong van lieu SSVEP (dich ca dau/cuoi cua so theo cung do tre thi giac).

Cua so CHINH MOI = 5.0s (t_start=0.14, t_end=5.14 => sample_start=160,
sample_end=1410 => n_samples=1250 => 1250/250=5.0s dung). Cua so nay lech dung
0.14s qua moc tat kich thich thuc — day la quy uoc chuan (dich dong bo ca hai
dau theo do tre 0.14s), KHONG phai loi.

Day la cua so tao ra con so chinh cua bai bao — Bang 2, 3, 8 (cot Benchmark),
9, va duoc trich dan trong Tom tat/Dong gop/Ket luan. Ket qua CU (5.44s, co
bug) van con trong results/baselines_full va results/ensemble_full — job nay
ghi vao thu muc MOI (co suffix _fixedwindow_round) de khong ghi de, tien so sanh.

Chi train 3 nhanh chinh thuc (temporal_cnn, freq_domain_cnn, spatial_temporal_cnn).
shallow_conv_net (nhanh thu 4, cho Bang 4/7 ablation) duoc train RIENG o giai
doan sau, khong can trong job nay.

Batch size = 16 dung theo Bang sieu tham so da cong bo trong bai (quyet dinh
2026-08-04: khong tang batch cho H100, uu tien khop chinh xac voi so da viet).

Ba buoc: baseline (3 x 100 epoch, LOSO 35 fold) -> naive ensemble -> collaborative
distillation (50 epoch, LOSO 35 fold).

Chay:  python run_full.py
       python run_full.py --skip-baseline     (neu da train baseline roi)
"""
import subprocess
import sys
import argparse
from pathlib import Path

# ── THAM SO CUA SO CHINH 5.0s VAT LY (DA SUA, xem docstring o tren) ──
T_START = "0.14"
T_END = "5.14"          # dung tu onset thuc => sample_end=1410 => 1250 mau => 5.0s

BASE_DIR = "results/baselines_full_fixedwindow_round"
ENS_DIR = "results/ensemble_full_fixedwindow_round"

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
                    help="Bo qua buoc train baseline (neu da co results/baselines_full)")
    ap.add_argument("--skip-naive", action="store_true",
                    help="Bo qua buoc naive ensemble")
    args = ap.parse_args()

    print("=" * 64)
    print("  CHAY CUA SO CHINH 5.0s VAT LY (DA SUA WINDOW BUG)")
    print(f"  t_start={T_START}, t_end={T_END} (tinh tu onset thuc -> 1250 mau @ 250Hz = 5.0s)")
    print(f"  Python: {PY}")
    print(f"  Baseline dir: {BASE_DIR}")
    print(f"  Ensemble dir: {ENS_DIR}")
    print("=" * 64)

    # ── BUOC 1: Train 3 baseline extractors ──
    if not args.skip_baseline:
        for ext in EXTRACTORS:
            run_step(
                f"[1/3] Train baseline extractor: {ext} (5.0s, fixed window)",
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
            "[2/3] Naive ensemble (5.0s, fixed window)",
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
        "[3/3] Collaborative distillation (5.0s, fixed window)",
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
    print("  HOAN TAT CUA SO CHINH 5.0s (fixed window)")
    print("=" * 64)
    print(f"  Ket qua trong:")
    print(f"    - {BASE_DIR}/  (3 baseline)")
    print(f"    - {ENS_DIR}/   (naive + collaborative)")
    print()
    print("  (ITR @5.0s dung T = 5.0 + 0.5 gaze = 5.5s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
