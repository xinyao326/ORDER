from .loss import OrderLoss


class PerTaskOrderLoss(OrderLoss):
    def forward(self, rep0, rep1, origin_label, epoch_idx=0):
        n_tasks = origin_label.shape[1]
        loss_rnc_tab = 0.
        loss_rnc_img = 0.

        for k in range(n_tasks):
            label_k = (origin_label[:, k:k+1] - self.mean[k]) / (self.std[k] + 1e-8)
            loss_rnc_tab += self.compute_RNCloss(rep0, label_k, epoch=epoch_idx)
            loss_rnc_img += self.compute_RNCloss(rep1, label_k, epoch=epoch_idx)

        loss_rnc_tab /= n_tasks
        loss_rnc_img /= n_tasks

        loss_CLIP = (
            self.compute_CLIPloss(rep0, rep1) +
            self.compute_CLIPloss(rep1, rep0)
        ).mean()

        return loss_rnc_tab, loss_rnc_img, loss_CLIP
