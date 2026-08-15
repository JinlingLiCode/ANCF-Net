"""CD, HD, and EMD definitions used in the manuscript experiments.

Adapted from HFCI-PU (Apache License 2.0). The formulas and default EMD
settings are intentionally preserved to reproduce the reported metrics.
"""

import torch
import torch.nn as nn

from pc_eval.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from pc_eval.emd_module.emd_module import emdModule


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.chamfer = chamfer_3DDist()
        self.emd = emdModule()

    def get_emd_loss(self, pred, gt, radius=1.0, eps=1.0, iters=512):
        distance, _ = self.emd(pred, gt, eps, iters)
        distance = torch.mean(torch.sqrt(distance), dim=1)
        return torch.mean(distance / radius)

    def get_cd_loss(self, pred, gt, radius=1.0):
        dist1, dist2, _, _ = self.chamfer(pred, gt)
        distance = torch.mean(dist1 + dist2, dim=1, keepdim=True)
        return torch.mean(distance / radius)

    def get_hd_loss(self, pred, gt, radius=1.0):
        dist1, dist2, _, _ = self.chamfer(pred, gt)
        distance = (
            torch.max(dist1, dim=1, keepdim=True)[0]
            + torch.max(dist2, dim=1, keepdim=True)[0]
        )
        return torch.mean(distance / radius)
