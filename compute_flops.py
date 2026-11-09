"""
compute_flops.py — Tinh FLOPs va params cho ensemble (khong train, chay vai giay).

Tinh cho tung nhanh + tong ensemble, o cua so chinh 5.44s (1360 mau).
Dung thu vien ptflops (hoac thop) - tu dong cai neu chua co.

Chay:  python compute_flops.py
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
from models.extractors.extractors import build_extractor

# ── CAU HINH (cua so chinh 5.44s) ──
N_CH = 9
N_SAMP = 1250          # 5.0s @ 250Hz (cua so chinh, da sua tu 5.44s loi)
N_CLS = 40
EXTRACTORS = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def try_ptflops(model, n_ch, n_samp):
    """Dung ptflops neu co, tra ve (macs, params)."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("  (dang cai ptflops...)")
        subprocess.run([sys.executable, "-m", "pip", "install", "ptflops", "-q"])
        from ptflops import get_model_complexity_info

    # ptflops can input_res dang tuple (khong gom batch)
    # Model nhan (B, C, T) hoac (B, 1, C, T) tuy kien truc -> thu ca hai
    for shape in [(n_ch, n_samp), (1, n_ch, n_samp)]:
        try:
            macs, params = get_model_complexity_info(
                model, shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            return macs, params, shape
        except Exception:
            continue
    return None, None, None


def main():
    device = torch.device('cpu')  # FLOPs khong can GPU
    print("=" * 64)
    print(f"  TINH FLOPs & PARAMS (cua so 5.44s, {N_SAMP} mau)")
    print("=" * 64)
    print(f"  Input: C={N_CH}, T={N_SAMP}, classes={N_CLS}")
    print()

    total_params = 0
    total_macs = 0
    results = []

    for ext in EXTRACTORS:
        model = build_extractor(ext, N_CH, N_SAMP, N_CLS).to(device).eval()
        params = count_params(model)
        macs, p_check, shape = try_ptflops(model, N_CH, N_SAMP)

        total_params += params
        if macs is not None:
            total_macs += macs
            flops = 2 * macs  # 1 MAC = 2 FLOP (nhan + cong)
            print(f"  {ext:24s}: params={params:,} | MACs={macs/1e6:.3f}M | FLOPs={flops/1e6:.3f}M")
            results.append((ext, params, macs))
        else:
            print(f"  {ext:24s}: params={params:,} | [!] khong tinh duoc MACs (thu thop)")
            results.append((ext, params, None))

    print("\n" + "=" * 64)
    print("  TONG ENSEMBLE (3 nhanh)")
    print("=" * 64)
    print(f"  Tong params : {total_params:,}  (~{total_params/1000:.1f}K)")
    if total_macs > 0:
        total_flops = 2 * total_macs
        print(f"  Tong MACs   : {total_macs/1e6:.3f}M")
        print(f"  Tong FLOPs  : {total_flops/1e6:.3f}M  (~{total_flops/1e6:.1f} MFLOPs)")
        print()
        print(f"  >>> DUNG CHO BAI:")
        print(f"      Params : {total_params/1000:.1f}K")
        print(f"      FLOPs  : {total_flops/1e6:.1f} MFLOPs (1 lan suy luan)")
        print(f"      Luu y: ensemble can 3 lan forward -> tong ~{3*total_flops/1e6:.1f} MFLOPs")
    print("=" * 64)


if __name__ == "__main__":
    main()
