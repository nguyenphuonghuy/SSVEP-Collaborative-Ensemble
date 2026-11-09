"""
evaluation/compute_effect_sizes.py
====================================
Tinh effect size (paired Cohen's d), trung vi chenh lech, CI 95%, va co
hieu chinh Bonferroni cho cac so sanh collaborative vs naive ensemble
(phan hoi gop y ve thong ke: "Van de thong ke", muc 6.1 va 6.2).

Doc truc tiep tu cac file collaborative_loso_*.json (moi nhat) trong
tung thu muc ket qua da chi ra.

Usage:
    python evaluation/compute_effect_sizes.py
"""

import json
import glob
import numpy as np
from scipy import stats

N_COMPARISONS = 5  # 4 cua so Benchmark + 1 BETA -> Bonferroni alpha = 0.05/5
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / N_COMPARISONS


def paired_cohens_d(diff: np.ndarray) -> float:
    """Cohen's d cho du lieu bat cap: mean(diff) / std(diff, ddof=1)."""
    return float(diff.mean() / diff.std(ddof=1))


def analyze(name: str, json_glob: str):
    files = sorted(glob.glob(json_glob))
    if not files:
        print(f"{name}: KHONG TIM THAY file khop '{json_glob}'")
        return None
    f = files[-1]
    d = json.load(open(f))
    folds = d['folds']
    naive = np.array([x['baseline_acc'] for x in folds])
    collab = np.array([x['best_acc'] for x in folds])
    diff = collab - naive
    n = len(diff)

    t, p_t = stats.ttest_rel(collab, naive)
    w, p_w = stats.wilcoxon(collab, naive)

    d_cohen = paired_cohens_d(diff)
    median_diff = float(np.median(diff))
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean_diff - t_crit * se, mean_diff + t_crit * se

    sig_corrected = p_t < ALPHA_CORRECTED

    print(f"=== {name} ({f}) ===")
    print(f"  n folds        : {n}")
    print(f"  Mean diff      : {mean_diff*100:+.2f} pp")
    print(f"  Median diff    : {median_diff*100:+.2f} pp")
    print(f"  95% CI mean    : [{ci_low*100:+.2f}, {ci_high*100:+.2f}] pp")
    print(f"  Cohen's d      : {d_cohen:.3f}  ({cohens_d_label(d_cohen)})")
    print(f"  paired t-test  : t={t:.3f}, p={p_t:.3e}")
    print(f"  Wilcoxon       : W={w:.3f}, p={p_w:.3e}")
    print(f"  p < alpha_corr ({ALPHA_CORRECTED:.3f}, Bonferroni/{N_COMPARISONS})"
          f" : {'CO Y NGHIA' if sig_corrected else 'KHONG con y nghia'}")
    print()

    return {
        'name': name, 'file': f, 'n': n,
        'mean_diff_pp': mean_diff * 100, 'median_diff_pp': median_diff * 100,
        'ci95_pp': [ci_low * 100, ci_high * 100],
        'cohens_d': d_cohen, 't': float(t), 'p_t': float(p_t),
        'wilcoxon_p': float(p_w),
        'significant_after_bonferroni': bool(sig_corrected),
    }


def cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "rat nho"
    if ad < 0.5: return "nho"
    if ad < 0.8: return "trung binh"
    return "lon"


if __name__ == "__main__":
    configs = [
        ("Benchmark 1,0s",  "results/ensemble_1s/collaborative_loso_*.json"),
        ("Benchmark 2,0s",  "results/ensemble_2s/collaborative_loso_*.json"),
        ("Benchmark 4,0s",  "results/ensemble_4s_real/collaborative_loso_*.json"),
        ("Benchmark 5,44s", "results/ensemble_full/collaborative_loso_*.json"),
        ("BETA 2,0s",       "results/ensemble_beta/collaborative_loso_*.json"),
    ]
    results = [analyze(name, g) for name, g in configs]
    results = [r for r in results if r is not None]

    print("=" * 70)
    print(f"TOM TAT — {len(results)}/{N_COMPARISONS} so sanh, "
          f"Bonferroni alpha={ALPHA_CORRECTED:.3f}")
    print("=" * 70)
    for r in results:
        flag = "PASS" if r['significant_after_bonferroni'] else "FAIL"
        print(f"  [{flag}] {r['name']:<16} d={r['cohens_d']:.3f}  "
              f"mean_diff={r['mean_diff_pp']:+.2f}pp  p={r['p_t']:.2e}")

    with open("results/effect_sizes_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved: results/effect_sizes_summary.json")
