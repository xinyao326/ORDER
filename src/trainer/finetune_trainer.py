import torch
import numpy as np
import os
from src.models import OrderModel


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator_1, final_evaluator_2, result_tracker,
                 summary_writer, device, model_name, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.final_evaluator_1 = final_evaluator_1
        self.final_evaluator_2 = final_evaluator_2
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.early_stop = (args.dataset == 'composite')

    def _forward_epoch(self, model, batched_data):
        (feat, labels) = batched_data
        feat = feat.to(self.device)
        labels = labels.to(self.device)
        predictions = model(feat)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            loss = self.loss_fn(predictions, labels)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

    def fit(self, model, train_loader, val_loader, test_loader):
        best_test_result, best_val_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0
        test_r2 = 0
        for epoch in range(1, self.args.n_epochs + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)
                try:
                    self.args.logger.info(f"Epoch: {epoch}\ttrain: {train_result:.3f}\tval: {val_result:.3f}\ttest: {test_result:.3f}")
                except:
                    print(f"Epoch: {epoch}\ttrain: {train_result:.3f}\tval: {val_result:.3f}\ttest: {test_result:.3f}")
                if self.early_stop:
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        test_r2 = self.eval(model, test_loader, is_test=True)
                else:
                    test_result = test_result
                    val_result = val_result
                    train_result = train_result
                    best_epoch = epoch
                    test_r2 = self.eval(model, test_loader, is_test=True)
                if epoch - best_epoch >= 20:
                    break
        return train_result, val_result, test_result, test_r2

    def eval(self, model, dataloader, is_test=False):
        model.eval()
        predictions_all = []
        labels_all = []

        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        if is_test:
            result = [self.final_evaluator_1.eval(torch.cat(labels_all), torch.cat(predictions_all)),
                      self.final_evaluator_2.eval(torch.cat(labels_all), torch.cat(predictions_all))]
        else:
            result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result


class FusionTrainer(Trainer):
    def _forward_epoch(self, model, batched_data):
        (idx, target, x_tabular, x_img) = batched_data
        x_tabular = x_tabular.to(self.device)
        x_img = x_img.to(self.device)
        labels = target.to(self.device)
        predictions = model(x_tabular, x_img)
        return predictions, labels
