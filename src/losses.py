import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NTXentLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        return self.criterion(sim, labels)


class HardNegativeNTXentLoss(nn.Module):

    def __init__(self, temperature: float = 0.1,
                 hard_negative_weight: float = 2.0, beta: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_negative_weight
        self.beta = beta

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.mm(z, z.t()) / self.temperature

        self_mask = torch.eye(2 * batch_size, device=z.device).bool()

        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True

        neg_mask = ~self_mask & ~pos_mask

        pos_sim = sim[pos_mask].view(2 * batch_size, 1)

        neg_sim = sim.masked_fill(~neg_mask, -1e9)

        n_hard = max(1, int(self.beta * (2 * batch_size - 2)))
        hard_neg_vals, _ = neg_sim.topk(n_hard, dim=1)

        all_neg_logsumexp = torch.logsumexp(neg_sim.masked_fill(~neg_mask, -1e9), dim=1)
        hard_neg_logsumexp = torch.logsumexp(hard_neg_vals, dim=1)

        neg_score = torch.logaddexp(
            all_neg_logsumexp,
            hard_neg_logsumexp + torch.log(torch.tensor(self.hard_weight - 1.0, device=z.device))
        )

        loss = -pos_sim.squeeze() + neg_score

        return loss.mean()


class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)

        sim = torch.mm(features, features.t()) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)

        self_mask = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - self_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        n_positives = mask_pos.sum(dim=1)
        n_positives = torch.clamp(n_positives, min=1)

        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_positives

        loss = -mean_log_prob.mean()
        return loss
