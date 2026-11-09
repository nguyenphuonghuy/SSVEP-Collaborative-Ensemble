# Harmonic-stacking multi-view ensemble for calibration-free SSVEP classification

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
| Four-branch ablation (branch removal), Benchmark | `table4.sh` (after `full.sh`); `round5s_ablA.sh`; `table4_4s.sh` |
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
| Embedded latency, resources, cross-platform accuracy | `pi4_latency/`, `zcu104_latency/` (see the scripts' headers) |
| Figures | `cd figures && python make_fig2.py` (etc.) |

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

See `CITATION.cff`. If you use this code, please cite the paper above.

## License

MIT, see `LICENSE`.
