"""
training/collaborative_ensemble.py
====================================
Task 4: Collaborative Distillation Training.

Mỗi extractor học từ 2 nguồn đồng thời:
  1. Ground truth label  → CrossEntropy loss
  2. Soft predictions của 3 extractors còn lại → KL Divergence loss

Công thức loss cho extractor i:
    L_i = α × CE(logits_i, y) + (1-α) × KL(p_i/T || p_avg_others/T)

    p_i         = softmax(logits_i / T)
    p_avg_others = mean(softmax(logits_j / T) for j ≠ i)
    T           = temperature (làm mềm distribution)
    α           = balance CE vs KD

Kỳ vọng: Ensemble sau collaborative > naive ensemble (75.24%)
→ Criterion 4 PASS

Usage:
    python training/collaborative_ensemble.py \
        --data_path data/raw/benchmark \
        --init_checkpoint_dir results/baselines
"""

import sys
import json
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset
from data.beta_loader import BETADataset
from data.deap_loader import DEAPDataset
from data.eegmmidb_loader import EEGMMIDBDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ENSEMBLE_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    # 'shallow_conv_net',   # Ablation A: bỏ Ext3 → ensemble tốt hơn (+2.55%)
    'spatial_temporal_cnn',
]

# DEAP không có tần số kích thích để xếp chồng hài (freq_domain_cnn không áp
# dụng) — dùng band_power_cnn (differential entropy theo băng tần) thay thế,
# giống train_extractors_deap.py.
DEAP_ENSEMBLE_EXTRACTORS = [
    'temporal_cnn',
    'band_power_cnn',
    'spatial_temporal_cnn',
]

# EEGMMIDB (motor imagery): ca 3 nhanh dung nguyen ven khong sua kien truc -
# band_power_cnn (5 bang tan tong quat) da bao phu dai mu/beta (8-30Hz)
# can cho ERD/ERS, khong can nhanh chuyen biet kieu SSVEPFreqNet.
EEGMMIDB_ENSEMBLE_EXTRACTORS = [
    'temporal_cnn',
    'band_power_cnn',
    'spatial_temporal_cnn',
]


# ─────────────────────────────────────────────────────────────────────────────
# COLLABORATIVE DISTILLATION LOSS
# ─────────────────────────────────────────────────────────────────────────────

class CollaborativeLoss(nn.Module):
    """
    Loss function cho collaborative distillation.

    Ho tro 3 che do (Ablation: co che chung cat cong tac, xem README /
    experiments/ablation_distill.sh):
      - 'avg' (mac dinh, dung trong toan bo ket qua chinh cua bai bao):
        L_i = alpha*CE(logits_i,y) + (1-alpha)*T^2*KL(p_i/T || mean_j p_j/T)
        Muc tieu mem la TRUNG BINH cua cac extractor con lai (leave-one-out),
        gop truoc khi tinh 1 KL duy nhat.
      - 'pairwise' (DML nguyen ban, Zhang et al. 2018, dung eq:dml trong bai):
        L_i = CE(logits_i,y) + sum_{j!=i} KL(p_j/T || p_i/T), moi cap tinh
        KL rieng roi CONG lai, khong gop trung binh truoc. Khong co he so
        alpha/(1-alpha) hay T^2 theo dung cong thuc DML goc trong tai lieu.
      - 'kdcl_naive' (KDCL-Naive, Guo et al. 2020 CVPR, eq. 4 trong bai goc):
        muc tieu mem KHONG phai trung binh, ma la logit CUA MOT peer duy nhat
        (chon trong TAT CA K peer, ke ca ban than) co CE loss thap nhat trong
        batch hien tai — dung CHUNG cho moi student (khac 'avg' o cho khong
        loai tru ban than va khong gop trung binh nhieu peer). Cong thuc loss
        con lai giong het 'avg' (alpha*CE + (1-alpha)*T^2*KL). Viec chon
        "teacher" duoc thuc hien o train_collaborative_epoch (can logit cua
        tat ca K model truoc khi tinh loss tung model).

    Lý do nhân T^2 (che do 'avg'/'kdcl_naive'): compensate cho gradient
    magnitude khi dùng temperature scaling. Theo Hinton et al. 2015.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0,
                 distill_mode: str = 'avg', label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        # 'fixed_teacher' (fixed-teacher control):
        # cung cong thuc loss nhu 'avg' (alpha*CE + (1-alpha)*T^2*KL), nhung
        # soft target la trung binh softmax cua MOT bo teacher CO DINH (3
        # extractor baseline dong bang, khong cap nhat trong luc fine-tune),
        # dung CHUNG cho ca 3 student — khac 'avg' o cho target khong "song"
        # (khong dong tien hoa cung student), co lap rieng dong gop cua viec
        # "teacher co dinh" khoi "trao doi hai chieu giua cac peer dang hoc".
        assert distill_mode in ('avg', 'pairwise', 'kdcl_naive', 'fixed_teacher')
        self.distill_mode = distill_mode
        # label_smoothing>0 (label-smoothing control): dung cung voi alpha=1.0 (khong
        # KD) de tao doi chung "lam min nhan chuan, khong co thong tin tu
        # peer nao" — tach rieng hieu ung dieu chuan don thuan khoi hieu
        # ung trao doi tri thuc lien-mien cua soft target tu cac extractor.
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_i: torch.Tensor,
        soft_targets,
        # 'avg': torch.Tensor (batch, n_classes) - trung binh da gop truoc.
        # 'kdcl_naive': torch.Tensor (batch, n_classes) - logit cua 1 peer
        #   duy nhat (CE thap nhat), dung chung cho moi student.
        # 'pairwise': List[torch.Tensor] - soft prob rieng cua tung peer.
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Returns: (total_loss, ce_loss_val, kd_loss_val)
        """
        loss_ce = self.ce(logits_i, y_true)
        log_p_i = F.log_softmax(logits_i / self.T, dim=-1)

        if self.distill_mode == 'pairwise':
            # DML goc: tong cac KL rieng tung cap (soft_targets la list),
            # khong nhan T^2, khong co alpha (dung dung eq:dml).
            loss_kd = sum(
                F.kl_div(log_p_i, peer, reduction='batchmean')
                for peer in soft_targets
            )
            total = loss_ce + loss_kd
        else:
            loss_kd = F.kl_div(
                log_p_i, soft_targets, reduction='batchmean'
            ) * (self.T ** 2)
            total = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        loss_kd_val = loss_kd.item() if torch.is_tensor(loss_kd) else float(loss_kd)
        return total, loss_ce.item(), loss_kd_val


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_collaborative_epoch(
    models: List[nn.Module],
    loader: DataLoader,
    optimizers: List[optim.Optimizer],
    criterion: CollaborativeLoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    teacher_models: Optional[List[nn.Module]] = None
) -> Dict:
    """
    1 epoch collaborative training cho tất cả extractors đồng thời.
    Mỗi batch: forward pass song song, tính soft targets từ nhau, backward.

    teacher_models: chi dung khi criterion.distill_mode == 'fixed_teacher'
    (fixed-teacher control) — 3 extractor dong bang (eval(), khong gradient),
    khoi tao tu CUNG checkpoint voi models nhung khong bao gio duoc cap nhat;
    trung binh softmax cua bo nay la "naive ensemble dong bang" dung lam
    muc tieu mem CHUNG cho ca 3 student, thay vi muc tieu "song" tu cac
    peer dang cung hoc nhu che do 'avg'.
    """
    for m in models:
        m.train()

    total_losses = [0.0] * len(models)
    total_ce     = [0.0] * len(models)
    total_kd     = [0.0] * len(models)
    correct      = [0]   * len(models)
    total        = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        batch_size = len(y_batch)

        # ── Step 1: Forward pass tất cả extractors ──
        all_logits = []
        for model in models:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(X_batch)
            else:
                logits = model(X_batch)
            all_logits.append(logits)

        # ── Step 2: Tính soft targets cho từng extractor ──
        # Soft target của extractor i = mean softmax(logits_j/T) với j ≠ i
        T = criterion.T
        all_soft = [F.softmax(lg / T, dim=-1).detach() for lg in all_logits]

        # ── KDCL-Naive: chon 1 peer duy nhat (CE thap nhat trong batch, ke
        # ca ban than) lam "teacher" chung cho MOI student (Guo et al. 2020,
        # eq. 4). Tinh mot lan cho ca batch, dung ngoai vong lap per-student.
        kdcl_teacher = None
        if criterion.distill_mode == 'kdcl_naive':
            with torch.no_grad():
                ce_per_model = [
                    F.cross_entropy(lg, y_batch).item() for lg in all_logits
                ]
            k_star = int(np.argmin(ce_per_model))
            kdcl_teacher = all_soft[k_star]

        # ── 'fixed_teacher': trung binh softmax cua bo teacher dong bang,
        # tinh 1 lan cho ca batch, dung CHUNG cho ca 3 student (khong phu
        # thuoc i, khac han 'avg' noi moi student co target rieng tu 2 peer
        # con lai) ──
        fixed_teacher_soft = None
        if criterion.distill_mode == 'fixed_teacher':
            assert teacher_models is not None, "fixed_teacher mode can teacher_models"
            with torch.no_grad():
                teacher_logits = [tm(X_batch) for tm in teacher_models]
                teacher_soft = [F.softmax(lg / T, dim=-1) for lg in teacher_logits]
                fixed_teacher_soft = torch.stack(teacher_soft, dim=0).mean(dim=0)

        # ── Step 3: Backward cho từng extractor ──
        for i, (model, optimizer, logits) in enumerate(
            zip(models, optimizers, all_logits)
        ):
            if criterion.distill_mode == 'pairwise':
                # Soft target = cac extractor con lai (KHAC voi i)
                others = [all_soft[j] for j in range(len(models)) if j != i]
                soft_target = others  # List[Tensor], DML: tinh KL rieng tung cap
            elif criterion.distill_mode == 'kdcl_naive':
                soft_target = kdcl_teacher  # dung chung cho moi i, ke ca k_star
            elif criterion.distill_mode == 'fixed_teacher':
                soft_target = fixed_teacher_soft  # dung chung cho moi i, khong "song"
            else:
                others = [all_soft[j] for j in range(len(models)) if j != i]
                soft_target = torch.stack(others, dim=0).mean(dim=0)  # (batch, n_classes)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, ce_val, kd_val = criterion(logits, soft_target, y_batch)
                scaler.scale(loss).backward(retain_graph=(i < len(models) - 1))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, ce_val, kd_val = criterion(logits, soft_target, y_batch)
                loss.backward(retain_graph=(i < len(models) - 1))
                optimizer.step()

            total_losses[i] += loss.item() * batch_size
            total_ce[i]     += ce_val * batch_size
            total_kd[i]     += kd_val * batch_size
            correct[i]      += (logits.detach().argmax(1) == y_batch).sum().item()

        total += batch_size

    return {
        'losses':   [l / total for l in total_losses],
        'ce_losses':[l / total for l in total_ce],
        'kd_losses':[l / total for l in total_kd],
        'accs':     [c / total for c in correct],
    }


@torch.no_grad()
def evaluate_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate naive ensemble (avg softmax) trên tập test.

    Luôn tính thêm balanced_ensemble_acc (trung bình recall theo lớp) —
    trên Benchmark/BETA các lớp gần như cân bằng nên gần bằng ensemble_acc,
    không ảnh hưởng gì; trên DEAP (lớp mất cân bằng theo subject) đây mới
    là chỉ số phản ánh đúng việc model có học được tín hiệu hay chỉ sụp về
    đoán lớp đa số.
    """
    for m in models:
        m.eval()

    correct_per_ext = [0] * len(models)
    total = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        all_probs = []
        for i, model in enumerate(models):
            logits = model(X_batch)
            prob = F.softmax(logits, dim=-1)
            all_probs.append(prob)
            correct_per_ext[i] += (logits.argmax(1) == y_batch).sum().item()

        avg_prob = torch.stack(all_probs, dim=0).mean(dim=0)
        preds = avg_prob.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        total += len(y_batch)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    correct_ensemble = int((preds == labels).sum())

    recalls = []
    for c in np.unique(labels):
        mask = labels == c
        recalls.append((preds[mask] == c).mean())
    balanced_ensemble_acc = float(np.mean(recalls)) if recalls else 0.0

    return {
        'ensemble_acc': correct_ensemble / total,
        'balanced_ensemble_acc': balanced_ensemble_acc,
        'per_ext_accs': [c / total for c in correct_per_ext],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO COLLABORATIVE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_collaborative_loso(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    init_checkpoint_dir: Path,
    save_dir: Path,
    config: dict,
    device: torch.device,
    ensemble_extractors: List[str] = ENSEMBLE_EXTRACTORS,
) -> Dict:
    """
    LOSO collaborative distillation training.

    Mỗi fold:
    1. Load 4 pretrained extractors từ Task 2 checkpoints
    2. Fine-tune cùng lúc với collaborative loss
    3. Evaluate ensemble accuracy
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects  = len(unique_subjects)
    n_channels  = X.shape[1]
    n_samples   = X.shape[2]
    n_classes   = len(np.unique(y))

    criterion = CollaborativeLoss(
        alpha=config['alpha'],
        temperature=config['temperature'],
        distill_mode=config.get('distill_mode', 'avg'),
        label_smoothing=config.get('label_smoothing', 0.0)
    )

    logger.info(f"\n{'='*65}")
    logger.info(f"COLLABORATIVE DISTILLATION — LOSO Training")
    logger.info(f"  alpha={config['alpha']}, T={config['temperature']}, "
                f"distill_mode={config.get('distill_mode', 'avg')}, "
                f"seed={config.get('seed', 'unseeded')}")
    logger.info(f"  epochs={config['epochs']}, lr={config['lr']}")
    logger.info(f"  Init from: {init_checkpoint_dir}")
    logger.info(f"{'='*65}")

    # Resume-theo-fold: neu job bi ngat giua chung (walltime dai, LOSO nhieu
    # fold), doc lai ket qua tam thoi da luu va bo qua cac fold da hoan tat
    # thay vi chay lai tu dau.
    partial_path = save_dir / 'collaborative_partial_results.json'
    completed_by_subject = {}
    if partial_path.exists():
        try:
            with open(partial_path) as f:
                prev_folds = json.load(f).get('folds', [])
            completed_by_subject = {r['subject']: r for r in prev_folds}
            logger.info(f"[RESUME] Tim thay {len(completed_by_subject)} fold da hoan tat "
                        f"trong {partial_path.name}, se bo qua khi gap lai.")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[RESUME] File partial results loi ({e}), bo qua, chay lai tu dau.")

    fold_results = []
    # sel_metric chi phu thuoc config, khong phu thuoc fold -- gan truoc vong lap
    # de khong bi UnboundLocalError khi TAT CA fold deu da hoan tat (resume toan bo).
    sel_metric = config.get('selection_metric', 'ensemble_acc')

    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_num = fold_idx + 1
        fold_start = time.time()

        if int(test_subj) in completed_by_subject:
            r = completed_by_subject[int(test_subj)]
            fold_results.append(r)
            logger.info(
                f"  [RESUME] Fold {fold_num:2d} | S{test_subj:02d} | "
                f"Baseline={r['baseline_acc']:.4f} → Collab={r['best_acc']:.4f} | "
                f"(bỏ qua, đã có từ trước)"
            )
            continue

        # ── Data split ──
        # Test subject hoan toan tach biet khoi moi quyet dinh chon checkpoint.
        # Tach them validation theo SUBJECT tu cac subject train de chon
        # checkpoint cong tac tot nhat trong qua trinh fine-tune.
        train_pool = unique_subjects[unique_subjects != test_subj]
        rng = np.random.RandomState(1000 + fold_idx)
        shuffled = rng.permutation(train_pool)
        n_val = min(max(3, round(0.15 * len(train_pool))), max(1, len(train_pool) - 1))
        val_subjects = set(shuffled[:n_val].tolist())
        train_subjects = set(shuffled[n_val:].tolist())

        test_mask  = subject_ids == test_subj
        val_mask   = np.isin(subject_ids, list(val_subjects))
        train_mask = np.isin(subject_ids, list(train_subjects))

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'],
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=64,
            shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=64,
            shuffle=False, num_workers=2, pin_memory=True
        )

        # ── Load pretrained models ──
        models = []
        for ext_name in ensemble_extractors:
            ckpt = init_checkpoint_dir / f"{ext_name}_fold{fold_num}.pt"
            model = build_extractor(ext_name, n_channels, n_samples, n_classes)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model = model.to(device)
            models.append(model)

        # ── 'fixed_teacher' (fixed-teacher): nap THEM mot bo teacher dong
        # bang, cung checkpoint khoi tao voi models nhung khong bao gio duoc
        # toi uu (eval(), requires_grad_(False)) — dai dien cho "naive
        # ensemble dong bang" lam muc tieu mem co dinh xuyen suot fine-tune ──
        teacher_models = None
        if config.get('distill_mode') == 'fixed_teacher':
            teacher_models = []
            for ext_name in ensemble_extractors:
                ckpt = init_checkpoint_dir / f"{ext_name}_fold{fold_num}.pt"
                tm = build_extractor(ext_name, n_channels, n_samples, n_classes)
                tm.load_state_dict(torch.load(ckpt, map_location=device))
                tm = tm.to(device).eval()
                for p in tm.parameters():
                    p.requires_grad_(False)
                teacher_models.append(tm)

        # ── Optimizers — lower lr vì đang fine-tune ──
        optimizers = [
            optim.Adam(m.parameters(), lr=config['lr'],
                       weight_decay=config['weight_decay'])
            for m in models
        ]
        schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
            for opt in optimizers
        ]

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # ── Baseline: naive ensemble trước khi collaborative (đánh giá trên
        # test - đây là số liệu được báo cáo, không dùng để chọn checkpoint) ──
        baseline = evaluate_ensemble(models, test_loader, device)
        baseline_acc = baseline['ensemble_acc']
        baseline_bal_acc = baseline['balanced_ensemble_acc']

        # ── Training loop — checkpoint được chọn theo VALIDATION, test subject
        # không tham gia bất kỳ quyết định nào trong vòng lặp này. Tiêu chí
        # chọn (sel_metric) là 'ensemble_acc' cho Benchmark/BETA (lớp cân
        # bằng), 'balanced_ensemble_acc' cho DEAP (lớp mất cân bằng theo
        # subject — accuracy thô sẽ ưu tiên checkpoint sụp về đoán lớp đa
        # số, xem cùng lý do sửa trong train_extractors_deap.py) ──
        best_val_acc = evaluate_ensemble(models, val_loader, device)[sel_metric]
        best_epoch = 0
        # Lưu checkpoint khởi điểm (trước fine-tune) làm mốc so sánh
        for ext_name, model in zip(ensemble_extractors, models):
            torch.save(model.state_dict(), save_dir / f"{ext_name}_collab_fold{fold_num}.pt")

        for epoch in range(config['epochs']):
            train_stats = train_collaborative_epoch(
                models, train_loader, optimizers, criterion, device, scaler,
                teacher_models=teacher_models
            )
            for sch in schedulers:
                sch.step()

            val_stats = evaluate_ensemble(models, val_loader, device)
            val_sel = val_stats[sel_metric]

            if val_sel > best_val_acc:
                best_val_acc = val_sel
                best_epoch = epoch
                # Save best checkpoints
                for ext_name, model in zip(ensemble_extractors, models):
                    torch.save(
                        model.state_dict(),
                        save_dir / f"{ext_name}_collab_fold{fold_num}.pt"
                    )

            # Log mỗi 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_train_acc = np.mean(train_stats['accs'])
                avg_ce  = np.mean(train_stats['ce_losses'])
                avg_kd  = np.mean(train_stats['kd_losses'])
                logger.info(
                    f"  Fold {fold_num:2d}/{n_subjects} | "
                    f"Ep {epoch+1:3d}/{config['epochs']} | "
                    f"TrainAcc={avg_train_acc:.3f} | "
                    f"CE={avg_ce:.3f} KD={avg_kd:.3f} | "
                    f"Val={val_stats['ensemble_acc']:.4f} (bal={val_stats['balanced_ensemble_acc']:.4f}) | "
                    f"Best[{sel_metric}](val)={best_val_acc:.4f}"
                )

        # ── Đánh giá TRÊN TEST ĐÚNG MỘT LẦN DUY NHẤT, dùng checkpoint đã
        # chọn bởi validation ──
        for ext_name, model in zip(ensemble_extractors, models):
            ckpt = save_dir / f"{ext_name}_collab_fold{fold_num}.pt"
            model.load_state_dict(torch.load(ckpt, map_location=device))
        final_test = evaluate_ensemble(models, test_loader, device)
        best_acc = final_test['ensemble_acc']
        best_bal_acc = final_test['balanced_ensemble_acc']

        fold_time = time.time() - fold_start
        gain_vs_baseline = best_acc - baseline_acc
        gain_vs_naive    = gain_vs_baseline  # baseline_acc IS the naive ensemble acc for this fold
        gain_vs_baseline_bal = best_bal_acc - baseline_bal_acc

        fold_results.append({
            'subject':      int(test_subj),
            'fold':         fold_num,
            'baseline_acc': float(baseline_acc),   # naive ensemble trước fine-tune
            'baseline_balanced_acc': float(baseline_bal_acc),
            'best_balanced_acc':     float(best_bal_acc),
            'gain_vs_baseline_balanced': float(gain_vs_baseline_bal),
            'best_acc':     float(best_acc),        # collaborative ensemble
            'best_epoch':   best_epoch,
            'gain_vs_baseline': float(gain_vs_baseline),
            'time_sec':     fold_time,
        })

        # Lưu kết quả tạm sau mỗi fold để có thể resume nếu job bị ngắt
        # giữa chừng (walltime dài, LOSO nhiều fold — xem đầu hàm).
        with open(partial_path, 'w') as f:
            json.dump({'folds': fold_results}, f, indent=2)

        logger.info(
            f"  ✓ Fold {fold_num:2d} | S{test_subj:02d} | "
            f"Baseline={baseline_acc:.4f} → Collab={best_acc:.4f} | "
            f"Gain={gain_vs_baseline:+.4f} ({fold_time:.1f}s)"
        )

    # ── Summary ──
    collab_accs   = [r['best_acc']     for r in fold_results]
    baseline_accs = [r['baseline_acc'] for r in fold_results]
    gains         = [r['gain_vs_baseline'] for r in fold_results]
    collab_bal_accs   = [r['best_balanced_acc']     for r in fold_results]
    baseline_bal_accs = [r['baseline_balanced_acc'] for r in fold_results]
    gains_bal         = [r['gain_vs_baseline_balanced'] for r in fold_results]

    mean_collab   = np.mean(collab_accs)
    mean_baseline = np.mean(baseline_accs)
    mean_gain     = np.mean(gains)
    naive_mean    = mean_baseline  # baseline_acc per fold IS the naive ensemble acc
    mean_collab_bal   = np.mean(collab_bal_accs)
    mean_baseline_bal = np.mean(baseline_bal_accs)
    mean_gain_bal     = np.mean(gains_bal)

    result = {
        'method':           'collaborative_distillation',
        'selection_metric': sel_metric,
        'collab_mean_acc':  float(mean_collab),
        'collab_std_acc':   float(np.std(collab_accs)),
        'collab_min_acc':   float(np.min(collab_accs)),
        'collab_max_acc':   float(np.max(collab_accs)),
        'baseline_mean_acc': float(mean_baseline),
        'mean_gain_vs_baseline': float(mean_gain),
        'n_folds_positive_gain': int(sum(1 for g in gains if g > 0)),
        'naive_ensemble_mean': naive_mean,
        'gain_vs_naive': float(mean_collab - naive_mean),
        'collab_mean_balanced_acc':   float(mean_collab_bal),
        'collab_std_balanced_acc':    float(np.std(collab_bal_accs)),
        'baseline_mean_balanced_acc': float(mean_baseline_bal),
        'mean_gain_vs_baseline_balanced': float(mean_gain_bal),
        'n_folds_positive_gain_balanced': int(sum(1 for g in gains_bal if g > 0)),
        'config': config,
        'folds': fold_results,
    }

    # Criterion 4 dùng ĐÚNG chỉ số đã dùng để chọn checkpoint (sel_metric):
    # accuracy thô cho Benchmark/BETA, balanced accuracy cho DEAP — tránh
    # kết luận PASS/FAIL dựa trên một chỉ số khác với chỉ số thực sự điều
    # khiển quá trình huấn luyện.
    if sel_metric == 'balanced_ensemble_acc':
        c4_collab, c4_naive = mean_collab_bal, mean_baseline_bal
        c4_label = 'balanced accuracy'
    else:
        c4_collab, c4_naive = mean_collab, naive_mean
        c4_label = 'accuracy'

    logger.info(f"\n{'='*65}")
    logger.info(f"COLLABORATIVE DISTILLATION — RESULTS")
    logger.info(f"  Collaborative mean  : {mean_collab:.4f} ± {np.std(collab_accs):.4f} "
                f"(bal={mean_collab_bal:.4f} ± {np.std(collab_bal_accs):.4f})")
    logger.info(f"  Naive ensemble mean : {naive_mean:.4f} (bal={mean_baseline_bal:.4f})")
    logger.info(f"  Gain vs naive       : {mean_collab - naive_mean:+.4f} (bal={mean_gain_bal:+.4f})")
    logger.info(f"  Folds with gain > 0 : {result['n_folds_positive_gain']}/{n_subjects} "
                f"(bal={result['n_folds_positive_gain_balanced']}/{n_subjects})")
    logger.info(f"\n  Criterion 4 Check (theo {c4_label}, khớp sel_metric='{sel_metric}'):")
    c4 = c4_collab > c4_naive
    logger.info(f"  Collab ({c4_collab:.4f}) > Naive ({c4_naive:.4f}): "
                f"{'✓ PASS' if c4 else '✗ FAIL'}")
    logger.info(f"{'='*65}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Task 4: Collaborative Distillation Training'
    )
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark')
    parser.add_argument('--init_checkpoint_dir', type=str,
                        default='results/baselines',
                        help='Thư mục chứa checkpoints từ Task 2')
    parser.add_argument('--save_dir', type=str,
                        default='results/ensemble',
                        help='Thư mục lưu collaborative checkpoints')
    parser.add_argument('--n_subjects', type=int, default=None,
                        help='Số subjects. None = tự detect (35 Benchmark, 70 BETA)')
    parser.add_argument('--extractors', type=str, default=None,
                        help="Danh sách nhánh tuỳ chỉnh, phân tách bằng dấu "
                             "phẩy (vd: temporal_cnn,spatial_temporal_cnn), "
                             "ghi đè bộ nhánh mặc định theo --dataset. Dùng "
                             "cho ablation loại bỏ nhánh yếu. Checkpoint "
                             "baseline tương ứng phải đã tồn tại trong "
                             "--init_checkpoint_dir.")
    parser.add_argument('--dataset', type=str, default='benchmark',
                        choices=['benchmark', 'beta', 'deap', 'eegmmidb'],
                        help='Dataset để train: benchmark (35 subjects), '
                             'beta (70 subjects), deap (32 subjects) hoặc '
                             'eegmmidb (103 subjects, motor imagery)')
    parser.add_argument('--label_type', type=str, default='valence',
                        choices=['valence', 'arousal'],
                        help='Chỉ dùng khi --dataset deap')
    parser.add_argument('--window_sec', type=float, default=4.0,
                        help='Chỉ dùng khi --dataset deap')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Fine-tuning epochs (ít hơn Task 2 vì đã pretrained)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate nhỏ hơn Task 2 (fine-tuning)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Balance CE vs KD loss. 0=full KD, 1=full CE')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Softmax temperature. Cao hơn → softer distribution')
    parser.add_argument('--distill_mode', type=str, default='avg',
                        choices=['avg', 'pairwise', 'kdcl_naive', 'fixed_teacher'],
                        help="'avg' (mac dinh, dung trong ket qua chinh): "
                             "muc tieu mem = trung binh cac peer (leave-one-out). "
                             "'pairwise': DML goc, tong KL rieng tung cap. "
                             "'kdcl_naive': KDCL-Naive (Guo et al. 2020), muc "
                             "tieu mem = logit cua 1 peer duy nhat (CE thap "
                             "nhat trong batch, ke ca ban than), dung chung "
                             "cho moi student (Ablation co che distillation). "
                             "'fixed_teacher' (fixed-teacher control): muc tieu "
                             "mem = trung binh softmax cua bo teacher DONG BANG "
                             "(khong cap nhat trong fine-tune), dung chung cho "
                             "ca 3 student, co lap rieng dong gop cua 'teacher "
                             "co dinh' khoi 'trao doi hai chieu giua cac peer "
                             "dang hoc' trong che do 'avg'.")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="label-smoothing control: dung voi --alpha 1.0 de tao doi "
                             "chung 'lam min nhan chuan, khong thong tin tu "
                             "peer nao' — tach rieng hieu ung dieu chuan don "
                             "thuan khoi hieu ung trao doi tri thuc lien-mien.")
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=2.14)  # fixed window bug: onset-relative now, 2.14 -> 2.0s
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed dieu khien tinh ngau nhien cua qua "
                             "trinh fine-tune (thu tu batch, dropout mask). "
                             "Trong so extractor luon nap co dinh tu checkpoint, "
                             "khong bi anh huong boi seed nay.")
    args = parser.parse_args()

    # Seed: chi anh huong stochasticity cua qua trinh fine-tune (thu tu batch
    # qua DataLoader shuffle, dropout mask); khong anh huong split train/val
    # (da co RNG rieng, co dinh theo fold_idx) hay trong so extractor (nap tu
    # checkpoint pretrained, khong khoi tao ngau nhien trong script nay).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Seed: {args.seed}")

    # Config
    config = {
        'epochs':      args.epochs,
        'batch_size':  args.batch_size,
        'lr':          args.lr,
        'weight_decay': 1e-4,
        'alpha':       args.alpha,
        'temperature': args.temperature,
        'distill_mode': args.distill_mode,
        'label_smoothing': args.label_smoothing,
        'seed':        args.seed,
        # DEAP mat can bang lop theo subject -> chon checkpoint theo balanced
        # accuracy; EEGMMIDB can bang tot nhung van dung balanced_acc de
        # nhat quan; Benchmark/BETA lop can bang -> giu accuracy tho nhu cu.
        'selection_metric': (
            'balanced_ensemble_acc' if args.dataset in ('deap', 'eegmmidb')
            else 'ensemble_acc'
        ),
    }

    # Load data
    logger.info(f"Loading {args.dataset} dataset từ {args.data_path} ...")
    if args.dataset == 'beta':
        n_subjects = args.n_subjects or 70
        data_loader = BETADataset(
            data_path=args.data_path,
            t_start=args.t_start,
            t_end=args.t_end,
            verbose=False
        )
    elif args.dataset == 'deap':
        n_subjects = args.n_subjects or 32
        data_loader = DEAPDataset(
            data_path=args.data_path,
            label_type=args.label_type,
            window_sec=args.window_sec,
            verbose=False
        )
    elif args.dataset == 'eegmmidb':
        n_subjects = args.n_subjects or 103
        data_loader = EEGMMIDBDataset(
            data_path=args.data_path,
            verbose=False
        )
    else:
        n_subjects = args.n_subjects or 35
        data_loader = BenchmarkDataset(
            data_path=args.data_path,
            t_start=args.t_start,
            t_end=args.t_end,
            verbose=False
        )
    X, y, subject_ids = data_loader.load_all_subjects(n_subjects=n_subjects)
    logger.info(f"Data: X={X.shape}, subjects={len(np.unique(subject_ids))}")

    if args.extractors:
        ensemble_extractors = [e.strip() for e in args.extractors.split(',')]
        logger.info(f"[ABLATION] Dung bo nhanh tuy chinh: {ensemble_extractors}")
    elif args.dataset == 'deap':
        ensemble_extractors = DEAP_ENSEMBLE_EXTRACTORS
    elif args.dataset == 'eegmmidb':
        ensemble_extractors = EEGMMIDB_ENSEMBLE_EXTRACTORS
    else:
        ensemble_extractors = ENSEMBLE_EXTRACTORS

    # Dirs
    init_dir = Path(args.init_checkpoint_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train
    result = train_collaborative_loso(
        X=X, y=y, subject_ids=subject_ids,
        init_checkpoint_dir=init_dir,
        save_dir=save_dir,
        config=config,
        device=device,
        ensemble_extractors=ensemble_extractors,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = save_dir / f"collaborative_loso_{timestamp}_seed{args.seed}.json"
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved: {save_path}")

    # Final verdict — dùng đúng chỉ số đã dùng để chọn checkpoint
    # (selection_metric): accuracy thô cho Benchmark/BETA, balanced accuracy
    # cho DEAP (xem lý do trong train_collaborative_loso).
    sel_metric = result['selection_metric']
    use_bal = sel_metric == 'balanced_ensemble_acc'
    collab_mean = result['collab_mean_balanced_acc'] if use_bal else result['collab_mean_acc']
    collab_std  = result['collab_std_balanced_acc'] if use_bal else result['collab_std_acc']
    naive_mean  = result['baseline_mean_balanced_acc'] if use_bal else result['naive_ensemble_mean']
    gain        = result['mean_gain_vs_baseline_balanced'] if use_bal else result['gain_vs_naive']
    n_pos_gain  = result['n_folds_positive_gain_balanced'] if use_bal else result['n_folds_positive_gain']

    print("\n" + "="*65)
    print("COLLABORATIVE DISTILLATION — FINAL SUMMARY"
          + (" (balanced accuracy)" if use_bal else ""))
    print("="*65)
    print(f"  Collaborative mean : {collab_mean:.4f} ± {collab_std:.4f}")
    print(f"  Naive ensemble     : {naive_mean:.4f}")
    print(f"  Gain vs naive      : {gain:+.4f}")
    n_folds_total = len(np.unique(subject_ids))
    print(f"  Folds gain > 0     : {n_pos_gain}/{n_folds_total}")
    print()
    c4 = collab_mean > naive_mean
    print(f"  Criterion 4 (Collab > Naive):", end=" ")
    if c4:
        print("✓ PASS → Ensemble validated!")
    else:
        print("✗ FAIL → Xem xét tune alpha/temperature")
    print("="*65)


if __name__ == "__main__":
    main()
