"""
evaluation/compute_effect_sizes_fixedwindow.py
================================================
Recompute paired statistics (t-test, Wilcoxon, paired Cohen's d, 95% CI,
win/tie/loss) for every Benchmark comparison after the window-definition
bug fix (data/benchmark_loader.py, data/beta_loader.py) and the resulting
full re-run of training (results/*_fixedwindow/).

Two comparison shapes:
  (a) "naive vs X" where both accuracies live in the SAME per-fold JSON
      (baseline_acc = naive ensemble, best_acc = X), e.g. the 4 Benchmark
      windows' main collaborative run, and the CE-only training-fairness
      check.
  (b) "A vs B" where A and B are two DIFFERENT collaborative_ensemble.py
      runs (e.g. avg vs pairwise, avg vs kdcl_naive) — matched by subject
      id, not by field name within one file.

Ties are defined as an exact difference of 0 (Benchmark accuracy has a
resolution of 1/240 per fold, so an exact 0 reliably means "no change",
not floating-point noise from a different computation path). Wilcoxon
uses zero_method='pratt' (keeps zero-difference pairs, ranks them at 0)
so the reported n matches the full fold count; 'wilcox' (drops zeros) is
also reported for comparison, to disclose the
zero-handling method.

Usage:
    python evaluation/compute_effect_sizes_fixedwindow.py
"""

import json
import glob
import numpy as np
from scipy import stats


def paired_cohens_d(diff: np.ndarray) -> float:
    return float(diff.mean() / diff.std(ddof=1))


def cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "rat nho"
    if ad < 0.5:
        return "nho"
    if ad < 0.8:
        return "trung binh"
    return "lon"


def _latest(json_glob: str):
    files = sorted(glob.glob(json_glob))
    if not files:
        return None
    return files[-1]


def _paired_stats(name: str, a: np.ndarray, b: np.ndarray, subjects=None):
    """a = the 'better/new' arm, b = the 'baseline' arm. diff = a - b."""
    diff = a - b
    n = len(diff)

    t, p_t = stats.ttest_rel(a, b)
    # Wilcoxon: report both zero-handling conventions explicitly.
    if np.all(diff == 0):
        w_pratt = p_w_pratt = w_wilcox = p_w_wilcox = float('nan')
    else:
        try:
            w_pratt, p_w_pratt = stats.wilcoxon(a, b, zero_method='pratt')
        except ValueError:
            w_pratt, p_w_pratt = float('nan'), float('nan')
        nonzero = diff != 0
        if nonzero.sum() == 0:
            w_wilcox, p_w_wilcox = float('nan'), float('nan')
        else:
            try:
                w_wilcox, p_w_wilcox = stats.wilcoxon(a[nonzero], b[nonzero], zero_method='wilcox')
            except ValueError:
                w_wilcox, p_w_wilcox = float('nan'), float('nan')

    d_cohen = paired_cohens_d(diff)
    mean_diff = float(diff.mean())
    median_diff = float(np.median(diff))
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean_diff - t_crit * se, mean_diff + t_crit * se

    n_win = int((diff > 0).sum())
    n_tie = int((diff == 0).sum())
    n_loss = int((diff < 0).sum())

    result = {
        'name': name, 'n': n,
        'mean_diff_pp': mean_diff * 100, 'median_diff_pp': median_diff * 100,
        'ci95_pp': [ci_low * 100, ci_high * 100],
        'cohens_d': d_cohen, 'cohens_d_label': cohens_d_label(d_cohen),
        't': float(t), 'p_t': float(p_t),
        'wilcoxon_p_pratt': float(p_w_pratt), 'wilcoxon_p_dropzeros': float(p_w_wilcox),
        'win': n_win, 'tie': n_tie, 'loss': n_loss,
    }

    print(f"=== {name} ===")
    print(f"  n folds            : {n}  (win/tie/loss = {n_win}/{n_tie}/{n_loss})")
    print(f"  Mean diff          : {mean_diff*100:+.2f} pp")
    print(f"  Median diff        : {median_diff*100:+.2f} pp")
    print(f"  95% CI mean        : [{ci_low*100:+.2f}, {ci_high*100:+.2f}] pp")
    print(f"  Cohen's d (paired) : {d_cohen:.3f}  ({cohens_d_label(d_cohen)})")
    print(f"  paired t-test      : t={t:.3f}, p={p_t:.3e}")
    print(f"  Wilcoxon (Pratt)   : W={w_pratt:.3f}, p={p_w_pratt:.3e}  [keeps zero-diff pairs]")
    print(f"  Wilcoxon (drop 0)  : W={w_wilcox:.3f}, p={p_w_wilcox:.3e}  [n_nonzero={n - n_tie}]")
    print()
    return result


def analyze_within_file(name: str, json_glob: str):
    f = _latest(json_glob)
    if f is None:
        print(f"{name}: KHONG TIM THAY file khop '{json_glob}'\n")
        return None
    d = json.load(open(f))
    folds = sorted(d['folds'], key=lambda x: x['subject'])
    naive = np.array([x['baseline_acc'] for x in folds])
    collab = np.array([x['best_acc'] for x in folds])
    r = _paired_stats(f"{name}  [{f}]", collab, naive)
    r['file'] = f
    return r


def analyze_across_files(name: str, glob_a: str, glob_b: str):
    fa, fb = _latest(glob_a), _latest(glob_b)
    if fa is None or fb is None:
        print(f"{name}: KHONG TIM THAY file ('{glob_a}' hoac '{glob_b}')\n")
        return None
    da = json.load(open(fa))
    db = json.load(open(fb))
    folds_a = {x['subject']: x['best_acc'] for x in da['folds']}
    folds_b = {x['subject']: x['best_acc'] for x in db['folds']}
    common = sorted(set(folds_a) & set(folds_b))
    if len(common) != len(folds_a) or len(common) != len(folds_b):
        print(f"  [!] Subject mismatch: A has {len(folds_a)}, B has {len(folds_b)}, common={len(common)}")
    a = np.array([folds_a[s] for s in common])
    b = np.array([folds_b[s] for s in common])
    r = _paired_stats(f"{name}  [A={fa} | B={fb}]", a, b)
    r['file_a'], r['file_b'] = fa, fb
    return r


if __name__ == "__main__":
    print("=" * 78)
    print("PRIMARY FAMILY: collaborative (avg) vs naive ensemble, 4 Benchmark windows + BETA")
    print(" alpha_corrected = 0.05/5 = 0.01")
    print("=" * 78)
    primary = []
    primary.append(analyze_within_file("Benchmark 1.0s (avg vs naive)", "results/ensemble_1s_fixedwindow_round/collaborative_loso_*.json"))
    primary.append(analyze_within_file("Benchmark 2.0s (avg vs naive)", "results/ensemble_2s_fixedwindow_round/collaborative_loso_*.json"))
    primary.append(analyze_within_file("Benchmark 4.0s (avg vs naive)", "results/ensemble_4s_fixedwindow_round/collaborative_loso_*.json"))
    primary.append(analyze_within_file("Benchmark 5.0s MAIN (avg vs naive)", "results/ensemble_full_fixedwindow_round/collaborative_loso_*.json"))
    primary.append(analyze_within_file("BETA 2.0s (avg vs naive)", "results/ensemble_beta_fixedwindow_round/collaborative_loso_*.json"))

    print("=" * 78)
    print("SECONDARY: robustness checks at the 5.0s main configuration")
    print("=" * 78)
    secondary = []
    secondary.append(analyze_within_file("CE-only fine-tuning vs naive (training-fairness)", "results/ensemble_full_ce_baseline_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_within_file("label-smoothing fine-tuning vs naive", "results/ablation_labelsmooth_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_within_file("pairwise (original DML) vs naive", "results/ablation_distill_pairwise_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_within_file("kdcl_naive vs naive", "results/ablation_distill_kdcl_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_within_file("fixed_teacher vs naive", "results/ablation_fixedteacher_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_across_files("avg vs pairwise (original DML)", "results/ensemble_full_fixedwindow_round/collaborative_loso_*.json", "results/ablation_distill_pairwise_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_across_files("avg vs kdcl_naive", "results/ensemble_full_fixedwindow_round/collaborative_loso_*.json", "results/ablation_distill_kdcl_fixedwindow_round/collaborative_loso_*.json"))
    secondary.append(analyze_across_files("collaborative(avg) vs CE-only (isolate KD from extra training)", "results/ensemble_full_fixedwindow_round/collaborative_loso_*.json", "results/ensemble_full_ce_baseline_fixedwindow_round/collaborative_loso_*.json"))

    all_results = [r for r in primary + secondary if r is not None]

    ALPHA = 0.05
    N_PRIMARY = 5  # 4 Benchmark windows + BETA
    alpha_corr = ALPHA / N_PRIMARY

    print("=" * 78)
    print(f"SUMMARY — primary family Bonferroni alpha = {alpha_corr:.3f} (n={N_PRIMARY})")
    print("=" * 78)
    for r in primary:
        if r is None:
            continue
        flag = "PASS" if r['p_t'] < alpha_corr else "FAIL"
        print(f"  [{flag}] {r['name']:<45} d={r['cohens_d']:+.3f}  mean_diff={r['mean_diff_pp']:+.2f}pp  "
              f"p_t={r['p_t']:.2e}  win/tie/loss={r['win']}/{r['tie']}/{r['loss']}")

    print()
    print("SUMMARY — secondary robustness checks (uncorrected p, report as-is)")
    for r in secondary:
        if r is None:
            continue
        print(f"  [{'sig' if r['p_t'] < ALPHA else 'ns '}] {r['name']:<55} d={r['cohens_d']:+.3f}  "
              f"mean_diff={r['mean_diff_pp']:+.2f}pp  p_t={r['p_t']:.2e}  win/tie/loss={r['win']}/{r['tie']}/{r['loss']}")

    with open("results/effect_sizes_fixedwindow_summary.json", "w") as fh:
        json.dump({'primary': primary, 'secondary': secondary}, fh, indent=2)
    print("\nSaved: results/effect_sizes_fixedwindow_summary.json")
