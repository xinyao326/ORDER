import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('..')
from src.data.singlemodal_dataset import (
    CompositeTableDataset, CompositeImageDataset,
    TableDataset, FibreImageDataset,
)
from src.model_config import config_dict
from src.models.order import OrderModel
from src.trainer.evaluator import Evaluator
from src.trainer.finetune_trainer import Trainer
from src.trainer.result_tracker import Result_Tracker
from src.trainer.scheduler import PolynomialDecayLR
from src.utils import create_logger, set_random_seed, print_arg

from torchvision.models import vit_b_16, ViT_B_16_Weights


class _SGPTViT(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Linear(768, out_dim)

    def forward(self, _, x_img):
        return self.vit(x_img)


class _Encoder(nn.Module):
    def __init__(self, common_dim=128, latent_dim=128):
        super().__init__()
        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)


class SGPTCompatImageNet(nn.Module):
    def __init__(self, cond_dim, cardinality):
        super().__init__()
        from src.models import TableTransformerWrapper
        self.table_processor  = TableTransformerWrapper(
            in_dim=cond_dim, out_dim=128, dropout=0, cardinality=cardinality)
        self.vision_processor = _SGPTViT(out_dim=128)
        self.encoder          = _Encoder(128, 128)

    def encode(self, x, modal):
        if modal == 'tab':
            return self.encoder(self.table_processor(x, x))
        elif modal == 'image':
            return self.encoder(self.vision_processor(x, x))
        raise ValueError(f"Unknown modal: {modal}")


class SGPTCompatCLIP(nn.Module):
    def __init__(self, cond_dim, cardinality):
        super().__init__()
        from src.models import TableTransformerWrapper
        from src.models.myclip import peftCLIP
        self.table_processor  = TableTransformerWrapper(
            in_dim=cond_dim, out_dim=128, dropout=0, cardinality=cardinality)
        self.vision_processor = peftCLIP(
            model_name='openai/clip-vit-base-patch16', output_dim=128, lora_r=8)
        self.encoder          = _Encoder(128, 128)

    def encode(self, x, modal):
        if modal == 'tab':
            return self.encoder(self.table_processor(x, x))
        elif modal == 'image':
            return self.encoder(self.vision_processor(x, x))
        raise ValueError(f"Unknown modal: {modal}")


def build_mlp_head(latent_dim, hidden_dim, out_dim, num_layers=3, dropout=0.2):
    layers = []
    layers += [nn.Linear(latent_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def extract_features(encoder, dataset, device, modal, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    encoder.eval()
    for batch in loader:
        x, y = batch
        x = x.to(device)
        feats.append(encoder.encode(x, modal).cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--n_epochs",     type=int,   default=50)
    parser.add_argument("--dataset",      type=str,   default='composite',
                        choices=['composite', 'fiber'])
    parser.add_argument("--modal",        type=str,   default='tab',
                        choices=['tab', 'image'])
    parser.add_argument("--metric",       type=str,   default='rmse')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--split_suffix", type=str,   default='_clean')
    parser.add_argument("--log",          type=str,   default='log')

    parser.add_argument("--prefix",       type=str,   default='order_dyn',
                        choices=['order_dyn', 'order_alpha', 'matmcl_imagenet', 'matmcl_clip',
                                 'baseline_dwcl', 'baseline_triplet'])
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="Alpha value for order_alpha prefix")
    parser.add_argument("--backbone",     type=str,   default='CLIP_ViT-B/16',
                        help="Backbone name (for ORDER models)")
    parser.add_argument("--r",            type=int,   default=8)
    parser.add_argument("--config",       type=str,   default='order')

    parser.add_argument("--matmcl_ckpt",  type=str,   default=None,
                        help="Path to MatMCL pretrained checkpoint (required for matmcl_* prefixes)")

    args = parser.parse_args()

    ss = args.split_suffix
    if args.prefix == 'order_dyn':
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', 'order_dyn',
                                    args.backbone.replace('/', '_'), setting)
    elif args.prefix == 'order_alpha':
        setting = f'seed{args.seed}_Alpha{args.alpha}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', 'order_alpha',
                                    args.backbone.replace('/', '_'), setting)
    elif args.prefix in ('baseline_dwcl', 'baseline_triplet'):
        loss_type = args.prefix.split('_', 1)[1]  # 'dwcl' or 'triplet'
        setting = f'seed{args.seed}_{loss_type}_Alpha{args.alpha}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix,
                                    args.backbone.replace('/', '_'), setting)
    else:
        setting = f'seed{args.seed}{ss}_{args.dataset}'
        args.savepth = os.path.join('save', args.prefix, setting)
        os.makedirs(args.savepth, exist_ok=True)

    if args.prefix in ('order_dyn', 'order_alpha', 'baseline_dwcl', 'baseline_triplet'):
        if not os.path.exists(args.savepth):
            raise RuntimeError(f'Checkpoint directory not found: {args.savepth}')
        args.weightpth = os.path.join(args.savepth, 'weight-final.pth')
    else:
        if args.matmcl_ckpt is None:
            raise ValueError('--matmcl_ckpt required for matmcl_* prefixes')
        args.weightpth = args.matmcl_ckpt

    args.logger = create_logger(args.savepth, f'{args.log}-predict_{args.modal}')
    args.logger.info(print_arg(args))
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ss = args.split_suffix

    if args.dataset == 'composite':
        rootpath   = '../datasets_composite'
        fea_col    = ['NumFibers', 'MMA', 'Vf', 'A11', 'A12', 'A13', 'A22', 'A23', 'A33']
        tar_col    = ['Yield strength', 'Elongation']
        cardin     = []
        cond_dim   = len(fea_col)
        n_task     = 2
        use_norm   = (args.prefix in ('matmcl_imagenet', 'matmcl_clip') or
                      args.backbone == 'ViT-B/16')

        if args.modal == 'tab':
            train_set = CompositeTableDataset(f'train_pred{ss}', rootpath, fea_col, tar_col, None, None,
                                              use_normalize=use_norm)
            val_set   = CompositeTableDataset(f'val_pred{ss}',   rootpath, fea_col, tar_col, None,
                                              train_set.scaler, use_normalize=use_norm)
            test_set  = CompositeTableDataset(f'test_pred{ss}',  rootpath, fea_col, tar_col, None,
                                              train_set.scaler, use_normalize=use_norm)
        else:
            processed = os.path.join(rootpath, 'processed')
            train_set = CompositeImageDataset(f'train_pred{ss}', rootpath, processed, feature_cols=fea_col, target_cols=tar_col)
            val_set   = CompositeImageDataset(f'val_pred{ss}',   rootpath, processed, feature_cols=fea_col, target_cols=tar_col)
            test_set  = CompositeImageDataset(f'test_pred{ss}',  rootpath, processed, feature_cols=fea_col, target_cols=tar_col)

    else:  # fiber
        data_path  = '../datasets_fiber/'
        rootpath   = '../datasets_fiber/table/mech'
        image_dir  = '../datasets_fiber/images/preprocessed'
        fea_col    = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        tar_col    = ['fracture', 'elongation', 'elastic modulus', 'tangent modulus', 'yield']
        cardin     = [2]
        cond_dim   = len(fea_col)
        n_task     = 5

        if args.modal == 'tab':
            train_set = TableDataset('mech', f'train_pred{ss}', data_path, scaler=None)
            val_set   = TableDataset('mech', f'val_pred{ss}',   data_path, scaler=train_set.scaler)
            test_set  = TableDataset('mech', f'test_pred{ss}',  data_path, scaler=train_set.scaler)
        else:
            train_set = FibreImageDataset(f'train_pred{ss}', rootpath, image_dir,
                                          feature_cols=fea_col, target_cols=tar_col, id_col='ID',
                                          imagenet_norm=True)
            val_set   = FibreImageDataset(f'val_pred{ss}',   rootpath, image_dir,
                                          feature_cols=fea_col, target_cols=tar_col, id_col='ID',
                                          imagenet_norm=True)
            test_set  = FibreImageDataset(f'test_pred{ss}',  rootpath, image_dir,
                                          feature_cols=fea_col, target_cols=tar_col, id_col='ID',
                                          imagenet_norm=True)

    train_mean = train_set.mean.numpy()
    train_std  = train_set.std.numpy()

    cfg = config_dict[args.config]

    if args.prefix in ('order_dyn', 'order_alpha', 'baseline_dwcl', 'baseline_triplet'):
        encoder = OrderModel(
            cond_dim=cond_dim,
            hidden_dim=cfg['hidden_dim'],
            common_dim=cfg['common_dim'],
            latent_dim=cfg['latent_dim'],
            dropout=cfg['dropout'],
            backbone=args.backbone,
            lora_r=args.r,
            cardinality=cardin,
        ).to(device)
        encoder.load_state_dict(torch.load(args.weightpth, map_location=device), strict=False)

    elif args.prefix == 'matmcl_imagenet':
        encoder = SGPTCompatImageNet(cond_dim=cond_dim, cardinality=cardin).to(device)
        ckpt = torch.load(args.weightpth, map_location=device)
        relevant = {k: v for k, v in ckpt.items()
                    if k.startswith(('table_processor', 'vision_processor', 'encoder'))}
        missing, unexpected = encoder.load_state_dict(relevant, strict=False)
        args.logger.info(f'MatMCL ImageNet: missing={missing}, unexpected={unexpected[:3]}...')

    elif args.prefix == 'matmcl_clip':
        encoder = SGPTCompatCLIP(cond_dim=cond_dim, cardinality=cardin).to(device)
        ckpt = torch.load(args.weightpth, map_location=device)
        relevant = {k: v for k, v in ckpt.items()
                    if k.startswith(('table_processor', 'vision_processor', 'encoder'))}
        missing, unexpected = encoder.load_state_dict(relevant, strict=False)
        args.logger.info(f'MatMCL CLIP: missing={missing}, unexpected={unexpected[:3]}...')

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    args.logger.info(f'Extracting {args.modal} features ...')
    tr_feat, tr_y = extract_features(encoder, train_set, device, args.modal)
    va_feat, va_y = extract_features(encoder, val_set,   device, args.modal)
    te_feat, te_y = extract_features(encoder, test_set,  device, args.modal)
    args.logger.info(f'  train={tr_feat.shape}, val={va_feat.shape}, test={te_feat.shape}')

    lat_dim = tr_feat.shape[1]
    train_loader = DataLoader(TensorDataset(tr_feat, tr_y), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(va_feat, va_y), batch_size=32, shuffle=False)
    test_loader  = DataLoader(TensorDataset(te_feat, te_y), batch_size=32, shuffle=False)

    model = build_mlp_head(lat_dim, cfg['hidden_dim'], n_task,
                           num_layers=3, dropout=args.dropout).to(device)

    optimizer    = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.n_epochs * len(tr_y) // 32 // 10,
        tot_updates=args.n_epochs * len(tr_y) // 32,
        lr=args.lr, end_lr=1e-9, power=1)

    label_mean = torch.tensor(train_mean).to(device) if train_mean is not None else None
    label_std  = torch.tensor(train_std).to(device)  if train_std  is not None else None

    evaluator        = Evaluator('MLP', args.metric,      n_task, mean=train_mean, std=train_std)
    final_eval_rmse  = Evaluator('MLP', 'rmse_split',     n_task, mean=train_mean, std=train_std)
    final_eval_r2    = Evaluator('MLP', 'r2',             n_task, mean=train_mean, std=train_std)
    result_tracker   = Result_Tracker(args.metric)

    trainer = Trainer(args, optimizer, lr_scheduler, nn.MSELoss(),
                      evaluator, final_eval_rmse, final_eval_r2, result_tracker,
                      summary_writer=None, device=device, model_name='MLP',
                      label_mean=label_mean, label_std=label_std)

    best_train, best_val, best_test, test_final = trainer.fit(
        model, train_loader, val_loader, test_loader)
    args.logger.info(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")

    avg_r2 = 0.0
    for i in range(n_task):
        rmse_fmt = f"{test_final[0][i]:.4f}" if args.dataset == 'composite' else f"{test_final[0][i]:.3f}"
        args.logger.info(f"{tar_col[i]}: test rmse: {rmse_fmt}"
                         f"\ttest r2: {test_final[1][i]:.3f}")
        avg_r2 += test_final[1][i]
    args.logger.info(f"Avg R2: {avg_r2 / n_task:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
