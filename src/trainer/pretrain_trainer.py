import torch
import matplotlib.pyplot as plt
import numpy as np
from ..utils import *


class Trainer:
    def __init__(self, args, optimizer, lr_scheduler, gmc_loss_fn, summary_writer, device, model_name="model"):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gmc_loss_fn = gmc_loss_fn
        self.summary_writer = summary_writer
        self.device = device
        self.minloss = 9999999
        self.maxres = 0

    def _forward_epoch(self, model, batched_data):
        (idx, target, x_tabular, x_img) = batched_data
        x_tabular = x_tabular.to(self.device)
        x_img = x_img.to(self.device)
        batch_repr = model.forward_unsupervised(x_tabular, x_img)
        return batch_repr, target

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        self.rnc_tab, self.rnc_img, self.clip_lss = 0, 0, 0
        self.train_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch_repr, _ = self._forward_epoch(model, batched_data)   
            loss = self.gmc_loss_fn(batch_repr, temperature=0.1, batch_size=32).mean()
            self.train_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
        self.train_loss /= batch_idx

    def print_info(self, epoch):
        strin = f"Epoch: {epoch}\tTrainLoss={self.train_loss.item():.3f}"
        try:
            self.args.logger.info(strin)
        except:
            print(strin)

    def fit(self, model, train_loader, test_loader):
        self.train_acc, self.val_acc = [], []
        for epoch in range(self.args.n_epochs):
            self.train_epoch(model, train_loader, epoch)
            self.test_result = self.eval(model, test_loader, epoch)
            self.train_result = self.eval(model, train_loader, epoch)
            self.train_acc.append(self.train_result)
            self.val_acc.append(self.test_result)            
            self.print_info(epoch)
            torch.save(model.state_dict(), f'{self.args.weightpth}-final.pth')

        return self.train_result, self.test_result

    def eval(self, model, dataloader, epoch=0):
        model.eval()
        loss_all = []
        tab_fea, img_fea = None, None
        with torch.no_grad():
            for batched_data in dataloader:
                batch_repr, _ = self._forward_epoch(model, batched_data)
                batch_size = batch_repr[0].size()[0]
                tab_fea = batch_repr[0] if tab_fea is None else torch.concat([tab_fea, batch_repr[0]])
                img_fea = batch_repr[1] if img_fea is None else torch.concat([img_fea, batch_repr[1]])

        similarity_matrix = torch.matmul(img_fea, tab_fea.T)
        labels = torch.arange(tab_fea.shape[0], device=similarity_matrix.device)
            
        img_to_tab_pred = similarity_matrix.argmax(dim=1)
        img_to_tab_acc = (img_to_tab_pred == labels).float().mean()
        
        tab_to_img_pred = similarity_matrix.T.argmax(dim=1)
        tab_to_img_acc = (tab_to_img_pred == labels).float().mean()

        return (img_to_tab_acc+tab_to_img_acc) / 2


class OrderTrainer(Trainer):
    def _forward_epoch(self, model, batched_data):
        (idx, target, x_tabular, x_img) = batched_data
        x_tabular = x_tabular.to(self.device)
        x_img = x_img.to(self.device)
        tab_rep = model.encode(x_tabular, 'tab')
        img_rep = model.encode(x_img, 'image')
        return img_rep, tab_rep, target.to(self.device)
    
    def print_info(self, epoch):
        self.args.logger.info(f"Epoch: {epoch}  rnc_tab_loss={self.rnc_tab.item():.3f}   rnc_img_loss={self.rnc_img.item():.3f}  clip_loss={self.clip_lss.item():.3f}")

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        self.train_loss, cnt = 0, 0
        self.rnc_tab, self.rnc_img, self.clip_lss = 0, 0, 0
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            img, tab, target = self._forward_epoch(model, batched_data)  
            rnc_tab, rnc_img, clip_loss = self.gmc_loss_fn(tab, img, target, epoch_idx)    
            loss = self.args.alpha * (rnc_img + rnc_tab) + (1-self.args.alpha) * clip_loss 
            self.train_loss += loss
            self.rnc_tab += rnc_tab
            self.rnc_img += rnc_img
            self.clip_lss += clip_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()

        self.train_loss /= batch_idx
        self.rnc_tab /= batch_idx
        self.rnc_img /= batch_idx
        self.clip_lss /= batch_idx

    def eval(self, model, dataloader, epoch=0, k=1):
        model.eval()
        loss_all = []
        tab_fea, img_fea, targets = None, None, None
        with torch.no_grad():
            for batched_data in dataloader:
                img_rep, tab_rep, target = self._forward_epoch(model, batched_data)
                tab_fea = tab_rep if tab_fea is None else torch.concat([tab_fea, tab_rep])
                img_fea = img_rep if img_fea is None else torch.concat([img_fea, img_rep])
                targets = target if targets is None else torch.concat([targets, target])
        
        similarity_matrix = torch.matmul(img_fea, tab_fea.T)
        labels = torch.arange(tab_fea.shape[0], device=similarity_matrix.device)

        _, img_to_tab_topk_idx = similarity_matrix.topk(k, dim=1)
        img_to_tab_correct = torch.any(img_to_tab_topk_idx == labels.unsqueeze(1), dim=1)
        img_to_tab_acc = img_to_tab_correct.float().mean()
        
        _, tab_to_img_topk_idx = similarity_matrix.T.topk(k, dim=1)
        tab_to_img_correct = torch.any(tab_to_img_topk_idx == labels.unsqueeze(1), dim=1)
        tab_to_img_acc = tab_to_img_correct.float().mean()

        return (img_to_tab_acc+tab_to_img_acc) / 2
    

class BaselineTrainer(OrderTrainer):
    def __init__(self, args, optimizer, lr_scheduler, loss_fn,
                 summary_writer, device, loss_type='dwcl'):
        super().__init__(args, optimizer, lr_scheduler, loss_fn, summary_writer, device)
        self.loss_type = loss_type

    def print_info(self, epoch):
        self.args.logger.info(
            f"Epoch: {epoch}  {self.loss_type}_tab_loss={self.rnc_tab.item():.3f}"
            f"   {self.loss_type}_img_loss={self.rnc_img.item():.3f}"
            f"   clip_loss={self.clip_lss.item():.3f}"
        )


class DynTrainer:
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, device):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.minloss = 9999999
        self.maxres = 0
        self.epo_lp = EPO_LP(m=2)

    def _forward_epoch(self, model, batched_data):
        (idx, target, x_tabular, x_img) = batched_data
        x_tabular = x_tabular.to(self.device)
        x_img = x_img.to(self.device)
        target = target.to(self.device)
        img_rep = model.encode(x_img, 'image')
        tab_rep = model.encode(x_tabular, 'tab')
        return img_rep, tab_rep, target

    def train_epoch(self, model, train_loader, val_loader, epoch_idx):
        model.train()
        self.losses = {
            'clip': 0, 'rnc': 0, 'val': 0, 'cnt': 0
        }
        val_iter = iter(val_loader)
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            img_rep, tab_rep, target = self._forward_epoch(model, batched_data)
            clip_loss = (self.loss_fn['clip'](img_rep, tab_rep) + self.loss_fn['clip'](tab_rep, img_rep)).mean()
            rnc_loss = self.loss_fn['rnc'](img_rep, target, epoch_idx) + self.loss_fn['rnc'](tab_rep, target, epoch_idx)
            g_clip = compute_gradient_vector(clip_loss, model)
            g_rnc = compute_gradient_vector(rnc_loss, model)
            try:
                val_data = next(val_iter)
            except:
                val_iter = iter(val_loader)
                val_data = next(val_iter)
            val_img_rep, val_tab_rep, val_target = self._forward_epoch(model, val_data)
            loss_val = (self.loss_fn['clip'](val_img_rep, val_tab_rep) + self.loss_fn['clip'](val_tab_rep, val_img_rep)).mean()
            g_val = compute_gradient_vector(loss_val, model)
            G = torch.stack([g_clip, g_rnc])
            try:
                alpha = self.epo_lp.get_alpha(G=G.cpu().numpy(), G_val=g_val.cpu().numpy(), loss_val=loss_val)
                if self.epo_lp.last_move == "dom":
                  descent += 1
            except Exception as e:
                print(e)
                alpha = np.array([0.5,0.5])
            alpha = torch.from_numpy(alpha).float().to(self.device)
            self.optimizer.zero_grad()
            img_rep, tab_rep, target = self._forward_epoch(model, batched_data)
            clip_loss = (self.loss_fn['clip'](img_rep, tab_rep) + self.loss_fn['clip'](tab_rep, img_rep)).mean()
            rnc_loss = self.loss_fn['rnc'](img_rep, target, epoch_idx) + self.loss_fn['rnc'](tab_rep, target, epoch_idx)
            self.losses['val'] += loss_val.item()
            self.losses['clip'] += clip_loss.item()
            self.losses['rnc'] += rnc_loss.item()
            self.losses['cnt'] += 1
            total_loss = alpha[0]*clip_loss + alpha[1]*rnc_loss 
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()

    def plot(self, savepth):
        x = list(range(len(self.train_acc)))
        plt.plot(x, self.train_acc, label='train')
        plt.plot(x, self.val_acc, label='val')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.savefig(savepth, bbox_inches='tight')

    def print_info(self, epoch):
        loss_str = ' '.join([f"{k}:{self.losses[k]/self.losses['cnt']:.2f}" for k in self.losses])
        self.args.logger.info(f"Epoch: {epoch} "+loss_str+self.savestr)

    def fit(self, model, train_loader, val_loader, test_loader):
        self.train_acc, self.val_acc = [], []
        for epoch in range(self.args.n_epochs):
            self.train_epoch(model, train_loader, val_loader, epoch)
            self.test_result = self.eval(model, test_loader, epoch)
            self.train_result = self.eval(model, train_loader, epoch)
            self.savestr = ' '
            self.train_acc.append(self.train_result)
            self.val_acc.append(self.test_result)            
            self.print_info(epoch)
            torch.save(model.state_dict(), f'{self.args.weightpth}-final.pth')

        return self.train_result, self.test_result

    def eval(self, model, dataloader, epoch=0, k=1):
        model.eval()
        loss_all = []
        tab_fea, img_fea, targets = None, None, None
        with torch.no_grad():
            for batched_data in dataloader:
                img_rep, tab_rep, target = self._forward_epoch(model, batched_data)
                tab_fea = tab_rep if tab_fea is None else torch.concat([tab_fea, tab_rep])
                img_fea = img_rep if img_fea is None else torch.concat([img_fea, img_rep])
                targets = target if targets is None else torch.concat([targets, target])

        similarity_matrix = torch.matmul(img_fea, tab_fea.T)
        labels = torch.arange(tab_fea.shape[0], device=similarity_matrix.device)

        _, img_to_tab_topk_idx = similarity_matrix.topk(k, dim=1)
        img_to_tab_correct = torch.any(img_to_tab_topk_idx == labels.unsqueeze(1), dim=1)
        img_to_tab_acc = img_to_tab_correct.float().mean()
        
        _, tab_to_img_topk_idx = similarity_matrix.T.topk(k, dim=1)
        tab_to_img_correct = torch.any(tab_to_img_topk_idx == labels.unsqueeze(1), dim=1)
        tab_to_img_acc = tab_to_img_correct.float().mean()

        return (img_to_tab_acc+tab_to_img_acc) / 2
    
