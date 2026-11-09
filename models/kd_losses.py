"""
models/kd_losses.py
===================================================================
Ham mat mat cho CCKD-SSVEP.

Framing: PRIVILEGED-INFORMATION DISTILLATION.
  Teacher nhin 9 kenh (thong tin dac quyen chi co luc train); student chi nhin
  1/3 kenh. Teacher va student nhan CUNG trial nhung KHAC input space, nen day
  KHONG phai KD kinh dien (teacher/student cung input) ma la chung cat duoi
  thong tin dac quyen.

Hai thanh phan:
  (1) logit_kd_loss  — response-based (Hinton), truyen quan he giua 40 lop.
  (2) sp_kd_loss     — SIMILARITY-PRESERVING KD (Tung & Mori, ICCV 2019),
                       THAY cho cosine alignment cung. Khop CAU TRUC quan he
                       giua cac mau trong batch (mau nao giong mau nao theo
                       teacher) thay vi ep feature tuyet doi bang nhau -> phu
                       hop khi teacher/student o khong gian input khac nhau,
                       khong truyen thong tin ma student von khong quan sat duoc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_kd_loss(z_s, z_t, tau=4.0):
    """KL(softmax(z_t/tau) || softmax(z_s/tau)) * tau^2."""
    p_t = F.softmax(z_t / tau, dim=1)
    log_p_s = F.log_softmax(z_s / tau, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)


def sp_kd_loss(f_s, f_t):
    """Similarity-Preserving KD. f_*: [B, d] (hoac [B, ...] se duoc flatten).
    G = normalize_row(F F^T) ; loss = ||G_s - G_t||_F^2 / B^2."""
    f_s = f_s.flatten(1)
    f_t = f_t.flatten(1)
    G_s = F.normalize(f_s @ f_s.t(), p=2, dim=1)     # [B,B]
    G_t = F.normalize(f_t @ f_t.t(), p=2, dim=1)
    b = f_s.size(0)
    return (G_s - G_t).pow(2).sum() / (b * b)


class CCKDLoss(nn.Module):
    """Tong loss student: CE + lambda_kd * logit_kd + lambda_sp * sp_kd.

    Dung trong training loop:
        crit = CCKDLoss(tau=4, lambda_kd=1.0, lambda_sp=1.0)
        z_s, f_s = student(x_student, return_feat=True)
        with torch.no_grad():
            z_t, f_t = teacher(x_teacher, return_feat=True)   # teacher dong bang
        loss, parts = crit(z_s, f_s, z_t, f_t, y)
    """
    def __init__(self, tau=4.0, lambda_kd=1.0, lambda_sp=1.0):
        super().__init__()
        self.tau, self.lkd, self.lsp = tau, lambda_kd, lambda_sp

    def forward(self, z_s, feats_s, z_t, feats_t, y):
        l_ce = F.cross_entropy(z_s, y)
        l_kd = logit_kd_loss(z_s, z_t, self.tau)
        # SP da tang: trung binh sp_kd tren tung tang feature (list). Ho tro ca truong hop 1 tensor.
        if isinstance(feats_s, (list, tuple)):
            l_sp = sum(sp_kd_loss(a, b) for a, b in zip(feats_s, feats_t)) / len(feats_s)
        else:
            l_sp = sp_kd_loss(feats_s, feats_t)
        total = l_ce + self.lkd * l_kd + self.lsp * l_sp
        return total, {"ce": l_ce.item(), "kd": l_kd.item(), "sp": l_sp.item()}
