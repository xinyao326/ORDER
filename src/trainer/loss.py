import torch
import torch.nn.functional as F
import torch.nn as nn


def cliploss(rep0, rep1, t=0.1):
    batch_size = rep0.shape[0]
    out_joint_mod = torch.cat([rep0, rep1], dim=0)
    sim_matrix_joint_mod = torch.exp(torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / t)
    mask_joint_mod = (torch.ones_like(sim_matrix_joint_mod) - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)).bool()
    sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

    pos_sim_joint_mod = torch.exp(torch.sum(rep0 * rep1, dim=-1) / t)
    pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
    loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))

    return loss_joint_mod


class OrderLoss(nn.Module):
    def __init__(self, mean, std, total_epoch, label_dis='l1', feature_dis='l2', label_norm='zscore', label_min=None, label_max=None, alpha=0.5, device='cuda:0'):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)
        if label_min is not None and label_max is not None:
            self.label_min = label_min.to(device)
            self.label_max = label_max.to(device)
        self.label_diff_fn = self._get_label_diff_fn(label_dis)
        self.feature_diff_fn = self._get_feature_sim_fn(feature_dis)
        self.t = 0.1
        self.t_rnc = torch.arange(1, 0.1, -0.9/total_epoch)
        self.alpha = alpha
        self.label_norm = label_norm
        self.device = device

    def _get_label_diff_fn(self, label_diff):
        if label_diff == 'l1':
            return lambda labels: torch.cdist(labels, labels, p=1)
        elif label_diff == 'l2':
            return lambda labels: torch.cdist(labels, labels, p=2)
        else:
            raise ValueError(f"Unsupported label_diff: {label_diff}")
    
    def _get_feature_sim_fn(self, feature_sim):
        if feature_sim == 'l2':
            return lambda features: -torch.cdist(features, features, p=2)
        elif feature_sim == 'l1':
            return lambda features: -torch.cdist(features, features, p=1)
        elif feature_sim == 'cosine':
            return lambda features: F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        elif feature_sim == 'product':
            return lambda features: features @ features.T    
        else:
            raise ValueError(f"Unsupported feature_sim: {feature_sim}")
        
    def normalize_labels(self, labels):
        if self.label_norm == 'zscore':
            return (labels - self.mean) / (self.std + 1e-8)
        elif self.label_norm == 'minmax':
            return (labels - self.label_min) / (self.label_max - self.label_min + 1e-8)
        else:
            return labels
        
    def compute_RNCloss(self, rep0, origin_labels, epoch=0):        
        features = rep0
        label_diffs = self.label_diff_fn(origin_labels)

        logits = self.feature_diff_fn(features) / self.t_rnc[epoch]
        
        exp_logits = logits.exp()
        
        n = logits.shape[0]

        mask = (1 - torch.eye(n, device=logits.device)).bool()
        logits = logits.masked_select(mask).view(n, n - 1)
        exp_logits = exp_logits.masked_select(mask).view(n, n - 1)
        label_diffs = label_diffs.masked_select(mask).view(n, n - 1)
        
        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]
            pos_label_diffs = label_diffs[:, k]

            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()

            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))            
            loss += -(pos_log_probs / (n * (n - 1))).sum()
        
        return loss

    def compute_CLIPloss(self, rep0, rep1):
        batch_size = rep0.shape[0]
        out_joint_mod = torch.cat([rep0, rep1], dim=0)
        sim_matrix_joint_mod = torch.exp(torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.t)
        mask_joint_mod = (torch.ones_like(sim_matrix_joint_mod) - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)).bool()
        sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

        pos_sim_joint_mod = torch.exp(torch.sum(rep0 * rep1, dim=-1) / self.t)
        pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
        loss_joint_mod = -torch.log(pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1))

        return loss_joint_mod

    def forward(self, rep0, rep1, origin_label, epoch_idx=0):
        label = self.normalize_labels(origin_label)
        loss_rnc_tab = self.compute_RNCloss(rep0, label, epoch=epoch_idx)
        loss_rnc_img = self.compute_RNCloss(rep1, label, epoch=epoch_idx)
        loss_CLIP = (self.compute_CLIPloss(rep0, rep1) + self.compute_CLIPloss(rep1, rep0)).mean()
        return loss_rnc_tab, loss_rnc_img, loss_CLIP


class RnCLoss(nn.Module):
    def __init__(self, mean, std, device, total_epoch, temperature=2, label_diff='l1', feature_sim='product', label_norm='zscore'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.label_diff_fn = self._get_label_diff_fn(label_diff)
        self.feature_sim_fn = self._get_feature_sim_fn(feature_sim)
        self.t_rnc = torch.arange(1, 0.1, -0.9/total_epoch)
        self.label_norm = label_norm

    def _get_label_diff_fn(self, label_diff):
        if label_diff == 'l1':
            return lambda labels: torch.cdist(labels, labels, p=1)
        elif label_diff == 'l2':
            return lambda labels: torch.cdist(labels, labels, p=2)
        else:
            raise ValueError(f"Unsupported label_diff: {label_diff}")
    
    def _get_feature_sim_fn(self, feature_sim):
        if feature_sim == 'l2':
            return lambda features: -torch.cdist(features, features, p=2)
        elif feature_sim == 'l1':
            return lambda features: -torch.cdist(features, features, p=1)
        elif feature_sim == 'cosine':
            return lambda features: F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        elif feature_sim == 'product':
            return lambda features: features @ features.T    
        else:
            raise ValueError(f"Unsupported feature_sim: {feature_sim}")
        
    def normalize_labels(self, labels):
        if self.label_norm == 'zscore':
            return (labels - self.mean) / (self.std + 1e-8)
        elif self.label_norm == 'minmax':
            return (labels - self.label_min) / (self.label_max - self.label_min + 1e-8)
        else:
            raise NotImplementedError()

    def forward(self, features, origin_label, epoch):
        labels = self.normalize_labels(origin_label)
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features) / self.t_rnc[epoch]
        exp_logits = logits.exp()

        n = logits.shape[0]

        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


class DistanceWeightedContrastiveLoss(nn.Module):
    def __init__(self, mean, std, sigma=1.0, temperature=0.1, device='cuda:0'):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.sigma = sigma
        self.temperature = temperature
        self.device = device

    def normalize_labels(self, labels):
        return (labels - self.mean) / (self.std + 1e-8)

    def forward_single(self, features, labels):
        N = features.shape[0]

        label_dists = torch.cdist(labels, labels, p=1)
        weights = torch.exp(-label_dists / self.sigma)
        mask_diag = 1.0 - torch.eye(N, device=features.device)
        weights = weights * mask_diag
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        sim = (features @ features.T) / self.temperature
        sim_masked = sim.masked_fill(torch.eye(N, device=features.device).bool(), -1e9)
        log_probs = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

        loss = -(weights * log_probs).sum(dim=1).mean()
        return loss


class PropertyTripletLoss(nn.Module):
    def __init__(self, mean, std, base_margin=0.2, device='cuda:0'):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.base_margin = base_margin
        self.device = device

    def normalize_labels(self, labels):
        return (labels - self.mean) / (self.std + 1e-8)

    def forward_single(self, features, labels):
        N = features.shape[0]

        D = torch.cdist(labels, labels, p=1)

        S = features @ features.T

        gap = D.unsqueeze(1) - D.unsqueeze(2)

        sim_diff = S.unsqueeze(1) - S.unsqueeze(2)

        eye = torch.eye(N, device=features.device)
        valid = (gap > 0).float()
        valid = valid * (1.0 - eye).unsqueeze(2)
        valid = valid * (1.0 - eye).unsqueeze(1)
        valid = valid * (1.0 - eye).unsqueeze(0)

        margin = self.base_margin * gap.clamp(min=0)

        triplet_loss = torch.clamp(sim_diff + margin, min=0)

        loss = (valid * triplet_loss).sum() / (valid.sum() + 1e-8)
        return loss


class BaselineLoss(nn.Module):
    def __init__(self, mean, std, loss_type='dwcl', device='cuda:0',
                 sigma=1.0, base_margin=0.2, temperature=0.1):
        super().__init__()
        self.device = device
        self.t = temperature

        if loss_type == 'dwcl':
            self.property_loss = DistanceWeightedContrastiveLoss(
                mean=mean, std=std, sigma=sigma, temperature=temperature, device=device)
        elif loss_type == 'triplet':
            self.property_loss = PropertyTripletLoss(
                mean=mean, std=std, base_margin=base_margin, device=device)
        else:
            raise ValueError(f"Unknown baseline loss_type: {loss_type}. Choose 'dwcl' or 'triplet'.")

        self.mean = mean.to(device)
        self.std = std.to(device)

    def normalize_labels(self, labels):
        return (labels - self.mean) / (self.std + 1e-8)

    def _clip_loss(self, rep0, rep1):
        batch_size = rep0.shape[0]
        out = torch.cat([rep0, rep1], dim=0)
        sim = torch.exp(torch.mm(out, out.t()) / self.t)
        mask = (torch.ones_like(sim) - torch.eye(2 * batch_size, device=sim.device)).bool()
        sim = sim.masked_select(mask).view(2 * batch_size, -1)
        pos = torch.exp((rep0 * rep1).sum(dim=-1) / self.t)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos / sim.sum(dim=-1))

    def forward(self, rep0, rep1, origin_label, epoch_idx=0):
        labels = self.normalize_labels(origin_label)
        loss_tab = self.property_loss.forward_single(rep0, labels)
        loss_img = self.property_loss.forward_single(rep1, labels)
        clip_loss = (self._clip_loss(rep0, rep1) + self._clip_loss(rep1, rep0)).mean()
        return loss_tab, loss_img, clip_loss


if __name__ == "__main__":
    pass
