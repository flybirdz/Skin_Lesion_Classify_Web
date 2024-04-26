import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CE_weight(nn.Module):
    def __init__(self, cls_num_list, E1=20, E2=50, E=100):
        super(CE_weight, self).__init__()
        self.cls_num_list = cls_num_list
        if torch.cuda.is_available():
            cls_num_list = torch.FloatTensor(cls_num_list).cuda()
        else:
            cls_num_list = torch.FloatTensor(cls_num_list)
        weight = 1.0 / cls_num_list
        self.weight = (weight / weight.sum()) * len(cls_num_list)
        self.E1 = E1
        self.E2 = E2
        self.E = E

    def forward(self, x, target, e, f1_score=[1, 1, 1, 1, 1, 1, 1]):
        if e <= self.E1:
            return F.cross_entropy(x, target)
        if e > self.E1 and e <= self.E2:
            now_power = (e - self.E1) / (self.E2 - self.E1)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            if torch.cuda.is_available():
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                per_cls_weights = torch.FloatTensor(per_cls_weights)
            return F.cross_entropy(x, target, weight=per_cls_weights)
        else:
            f1_score = torch.FloatTensor(f1_score).cuda() if torch.cuda.is_available() else torch.FloatTensor(f1_score)
            weight = 1.0 / f1_score
            self.weight = (weight / weight.sum()) * len(self.cls_num_list)
            now_power = (e - self.E2) / (self.E - self.E2)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            if torch.cuda.is_available():
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                per_cls_weights = torch.FloatTensor(per_cls_weights)
            return F.cross_entropy(x, target, weight=per_cls_weights)

class BHP(nn.Module):
    def __init__(self, cls_num_list=None, proxy_num_list=None, temperature=0.1):
        super(BHP, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list
        self.proxy_num_list = proxy_num_list

    def forward(self, proxy, features, targets):
        device = features.device
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)

        targets_proxy = torch.empty((0, 1), dtype=torch.int64, device=device)
        for i, num in enumerate(self.proxy_num_list):
            tmp_targets = torch.full([num, 1], i, dtype=torch.int64, device=device)
            targets_proxy = torch.cat((targets_proxy, tmp_targets), dim=0)

        targets_proxy = targets_proxy.view(-1, 1)

        targets = torch.cat([targets.repeat(2, 1), targets_proxy], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2 + int(np.array(self.proxy_num_list).sum()), device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, proxy], dim=0)
        logits = features.mm(features.T)
        logits = torch.div(logits, self.temperature)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size + int(np.array(self.proxy_num_list).sum()), 2 * batch_size + int(np.array(self.proxy_num_list).sum())) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
