# Harmonic-stacking multi-view ensemble for calibration-free SSVEP classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24075618.svg)](https://doi.org/10.5281/zenodo.24075618)

Code for the paper *"Lightweight calibration-free SSVEP classification with a
harmonic-stacking multi-view ensemble on embedded ARM processors"*
(T. M. T. Duong, P. H. Nguyen; submitted to Biomedical Signal Processing and Control).

Some in-code comments are written in Vietnamese shorthand; they do not affect execution.

## Layout

```
data/              Dataset loaders (Benchmark, BETA; other loaders are imported but unused here)
models/extractors/ TemporalCNN, SSVEPFreqNet (harmonic stacking), SpatialTemporalCNN,
                   plus the ablation baselines (harmonic_amp_linear, harmonic_stack_linear,
                   full_spectrum_cnn, shallow_conv_net)
training/          LOSO training of single extractors (Benchmark / BETA) and
                   collaborative distillation fine-tuning
evaluation/        Naive ensemble, ablations, FBCCA baseline, effect sizes, branch diversity
run_full.py        Main 5.0 s pipeline (3 extractors -> naive ensemble -> collaborative)
run_{1s,2s,4s}.py  Same pipeline at the shorter windows
compute_flops.py   Parameter / FLOP counts
experiments/       One shell script per experiment reported in the paper (see table below)
pi4_latency/       On-device latency + accuracy verification, Raspberry Pi 4 (CPU)
zcu104_latency/    Same, ZCU104 (ARM CPU)
figures/           Scripts that regenerate the paper figures from saved result JSONs
```

## Environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118   # pick a build for your GPU
pip install -r requirements.txt                                               # numpy<2, scipy
pip install matplotlib                                                        # figures only
```

All experiments use a fixed seed (42) unless the script states otherwise, and `--batch_size 16`.

## Data (not included)

- **Benchmark** (Wang et al., 2017): place `S1.mat ... S35.mat` in `data/raw/benchmark/`.
- **BETA** (Liu et al., 2020): place the 70 subject files in `data/raw/beta/`.

## Running

Every script in `experiments/` is self-contained: run it from anywhere with
`bash experiments/<name>.sh`. It changes into the repository root, writes a log to `logs/`,
and writes results to `results/`. Set `PY=/path/to/python` to override the interpreter.
Scripts that consume another script's output must be run after it (noted below).

| Paper content | Script(s) |
|---|---|
| Per-extractor performance and complexity; fusion progression; main Benchmark results (5.0 s) | `full.sh` (or `round5s_freq.sh` then `round5s_ens.sh`) |
| Branch combinations (1/2/3 branches), all windows and BETA | `python evaluation/branch_combos.py` per window (after the baseline runs), then `python evaluation/phase1_stats.py` |
| Four-branch ablation (branch removal; Supplementary S6) | `table4.sh` (after `full.sh`); `round5s_ablA.sh`; `table4_4s.sh` |
| FBCCA on BETA | `python evaluation/fbcca_beta.py --t_end 2.14` |
| Paired tests (ensemble vs branch, vs FBCCA), per-subject ITR, seed summaries | `python evaluation/phase1_stats.py` |
| Harmonic-stacking controls: parameter-matched full-spectrum CNN, H = 1-4, exact-DFT variant (post hoc) | `python training/train_extractors.py --extractor {full_spectrum_cnn_matched,freq_domain_cnn_h1..h4,freq_domain_cnn_dft} --seed 42 ...`, then `python evaluation/phase2_stats.py` |
| Training-free harmonic score | `python evaluation/harmonic_score_baseline.py` |
| Ensemble with the exact-DFT frequency branch (post hoc) | `python evaluation/dft_ensemble.py` |
| Harmonic-stacking ablation ladder and FBCCA, all windows | `novelty_baselines.sh`, `fbcca.sh`, `novelty_{1s,2s,4s}.sh` |
| Full-spectrum CNN control | `fullspectrum_control.sh` |
| Window-length effect (accuracy, ITR) | `1s.sh`, `2s.sh`, `4s.sh`, then `collab_retune_{1s,2s,4s}.sh` |
| Hyperparameter selection on validation folds (alpha x T) | `hp_search_{1s,2s,4s,5s}.sh` |
| FFT bin-resolution sensitivity at 1.0 s | `sens1s.sh` |
| Training-fairness check (CE-only, label smoothing) | `ce_baseline.sh`, `ce_baseline_4s.sh`, `labelsmooth_control.sh` |
| Soft-target construction (avg vs DML vs KDCL-Naive vs fixed teacher) | `ablation_distill*.sh`, `ablation_kdcl*.sh`, `fixedteacher_control.sh` |
| Per-extractor accuracy and branch diversity | `python evaluation/branch_diversity_benchmark.py` (after `full.sh`) |
| Variance decomposition (5 seeds) and baseline-stage seeds | `multiseed_avg.sh`, `multiseed_avg_4s.sh`, `multiseed_baseline_{1001,2002}.sh` |
| Effect sizes, confidence intervals, and all variant-vs-averaging p-values | `python evaluation/compute_effect_sizes_fixedwindow.py` |
| BETA: extractors, naive and collaborative ensemble, architecture check | `beta1.sh` then `beta2.sh`; `beta_collab_retune.sh` |
| BETA alpha x T sensitivity | `table6.sh` (after `beta1.sh`) |
| Parameters / MFLOPs | `python compute_flops.py` |
| Embedded latency (ensemble, reference and cached FBCCA, 1/4 threads, median/p99), memory, and trial-level ARM/GPU agreement (Table 6, S7, S8) | `board_phase3/`: copy to the board with `models/`, fold checkpoints and exported trials, run `bash run_all.sh <python> 7 5`, then `python board_phase3/aggregate.py` on the host |
| Earlier latency scripts | `pi4_latency/`, `zcu104_latency/` (see the scripts' headers) |
| Figures | `cd figures && python make_fig1.py && python make_data_figs.py` (Fig. 1 needs `rsvg-convert` for the PDF) |

## Notes

- Window length is defined from the true stimulus onset: `--t_start 0.14 --t_end 5.14`
  gives 1250 samples at 250 Hz, i.e. exactly 5.0 s (see the docstring of `run_full.py`).
- Benchmark trials are used as distributed (no extra filtering); BETA trials are
  band-pass filtered at 6-90 Hz in `data/beta_loader.py`. Both are z-normalized per trial.
- In every LOSO fold the test subject is never used for checkpoint, architecture or
  hyperparameter selection; validation subjects are drawn from the training subjects.
- The `export CUDA_MPS_PIPE_DIRECTORY` / `FI_PROVIDER` lines in the experiment scripts are
  workarounds for the shared GPU cluster used for the experiments and are harmless elsewhere.

## Citation

Repository: https://github.com/nguyenphuonghuy/SSVEP-Harmonic-Ensemble

Archived on Zenodo (all versions): https://doi.org/10.5281/zenodo.24075618. Version 1.0.0: https://doi.org/10.5281/zenodo.24075619. Version 1.1.0 adds the branch-combination, harmonic-stacking control, exact-DFT, and embedded latency/decision-agreement scripts used in the submitted manuscript.

See `CITATION.cff`. If you use this code, please cite the paper above.

## License

MIT, see `LICENSE`.
